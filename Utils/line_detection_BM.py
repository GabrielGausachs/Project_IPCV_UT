import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN

# Global variables for tracking
tracking_points = None
H = None

# Function to create the field mask
def mask_field(frame, num=10):
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([im_hsv], [0], None, [180], [0, 180])
    peak_idx = np.argmax(hue_hist)
    min_hue = max(peak_idx - num, 0)
    max_hue = min(peak_idx + num, 179)

    lower_green = np.array([min_hue, 50, 50])
    upper_green = np.array([max_hue, 255, 255])
    field_mask = cv2.inRange(im_hsv, lower_green, upper_green)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    field_mask = (labels == largest_label).astype(np.uint8)

    masked_frame = cv2.bitwise_and(frame, frame, mask=field_mask)
    return masked_frame

# Detect edges and lines
def detect_lines(frame):
    masked_frame = mask_field(frame)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    blurred_small = cv2.GaussianBlur(gray, (3, 3), 1)
    blurred_large = cv2.GaussianBlur(gray, (5, 5), 3)

    edges_small = cv2.Canny(blurred_small, 71, 99)
    edges_large = cv2.Canny(blurred_large, 81, 109)

    combined_edges = cv2.bitwise_or(edges_small, edges_large)
    kernel = np.ones((5, 5), np.uint8)
    final_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)

    return final_edges

# Calculate intersection between two line segments
def intersection(o1, p1, o2, p2):
    o1, p1, o2, p2 = map(np.array, [o1, p1, o2, p2])
    d1 = p1 - o1
    d2 = p2 - o2
    x = o2 - o1
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-8:
        return None
    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = o1 + d1 * t1
    if (min(o1[0], p1[0]) <= r[0] <= max(o1[0], p1[0]) and 
        min(o1[1], p1[1]) <= r[1] <= max(o1[1], p1[1]) and 
        min(o2[0], p2[0]) <= r[0] <= max(o2[0], p2[0]) and 
        min(o2[1], p2[1]) <= r[1] <= max(o2[1], p2[1])):
        return (int(round(r[0])), int(round(r[1])))
    return None

# Cluster close intersection points and return centroids
def cluster_intersection_points(intersection_points, eps=10, min_samples=1):
    points = np.array([p for p in intersection_points if p is not None])

    if len(points) == 0:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    centroids = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue
        cluster_points = points[clustering.labels_ == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append((int(round(centroid[0])), int(round(centroid[1]))))

    return centroids

# Neighbor check for goal line points
def filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=30):
    filtered_points = []
    for i, pt1 in enumerate(centroids):
        x1, y1 = pt1
        for j, pt2 in enumerate(centroids):
            if i != j:
                x2, y2 = pt2
                if abs(x1 - x2) <= max_x_dist and abs(y1 - y2) <= max_y_dist:
                    filtered_points.append(pt1)
                    break
    return filtered_points

# Order goal line points in the required sequence
def order_goal_line_points(points):
    # Sort points by y-coordinate to separate top and bottom points
    sorted_points = sorted(points, key=lambda p: p[1])
    
    # Bottom points (lower y value), then sorted by x for left and right
    bottom_points = sorted(sorted_points[:2], key=lambda p: p[0])
    
    # Top points (higher y value), then sorted by x for left and right
    top_points = sorted(sorted_points[2:], key=lambda p: p[0])
    
    # Return points in desired order: [Bottom-left, Bottom-right, Top-left, Top-right]
    return [bottom_points[0], bottom_points[1], top_points[0], top_points[1]]

# Track points using optical flow
def track_points(prev_frame, current_frame, points):
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points, dtype="float32").reshape(-1, 1, 2)
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None)
    new_points = [tuple(pt.ravel()) for pt, s in zip(current_points, status) if s == 1]
    return new_points if len(new_points) == len(points) else None

# Overlay advertisement on the field using homography matrix
def overlay_advertisement(frame, ad_image, homography_matrix):
    # Rotate the ad image by 90 degrees
    rotated_ad_image = cv2.rotate(ad_image, cv2.ROTATE_90_CLOCKWISE)

    # Define the new target location in real-world coordinates, moving it slightly up and left
    # Adjust the values to place the ad outside the field, near the top-left boundary
    ad_points_real_world = np.array([
        [8.0, -2.0],  # Adjust this to move left or right along the sideline
        [11.0, -2.0],  
        [8.0, -0.25],  
        [11.0, -0.25]   
    ], dtype="float32")

    # GOAL POST ads location
    # Real-world coordinates for the ad placement
    # ad_points_real_world = np.array([
    #     [1.0, -2.5],   # Bottom-left of ad near the sideline
    #     [5.0, -2.5],  # Bottom-right of ad near the sideline
    #     [1.0, -0.25],  # Top-left of ad near the goal line
    #     [5.0, -0.25]  # Top-right of ad near the goal line
    # ], dtype="float32")

    # Transform the real-world ad points to video coordinates using the homography matrix
    ad_points_video = cv2.perspectiveTransform(np.array([ad_points_real_world]), homography_matrix)[0]

    # Define the corners of the rotated ad image
    ad_height, ad_width = rotated_ad_image.shape[:2]
    ad_corners = np.array([
        [0, 0],                      # Top-left corner
        [ad_width, 0],               # Top-right corner
        [0, ad_height],              # Bottom-left corner
        [ad_width, ad_height]        # Bottom-right corner
    ], dtype="float32")

    # Compute the perspective transform to place the rotated ad onto the field
    transform_matrix = cv2.getPerspectiveTransform(ad_corners, ad_points_video)
    warped_ad = cv2.warpPerspective(rotated_ad_image, transform_matrix, (frame.shape[1], frame.shape[0]))

    # Create a mask to blend the advertisement into the frame
    ad_mask = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
    _, ad_mask = cv2.threshold(ad_mask, 1, 255, cv2.THRESH_BINARY)

    # Remove the background area where the ad will be placed
    frame_background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(ad_mask))

    # Overlay the advertisement onto the frame
    result_frame = cv2.add(frame_background, warped_ad)

    return result_frame

# Function to detect intersections, filter for goal line points, and track them
def detecting_lines_intersection_points(frame, prev_frame, ad_image):
    global tracking_points, H

    # Perform goal line detection only on the first frame if tracking_points is None
    if tracking_points is None:
        # Detect edges and lines
        edges = detect_lines(frame)
        vertical_lines = []
        horizontal_lines = []
        frame_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=95, minLineLength=130, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if -10 < angle < 10:
                    horizontal_lines.append((x1, y1, x2, y2, length, angle))
                    cv2.line(frame_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    vertical_lines.append((x1, y1, x2, y2, length, angle))
                    cv2.line(frame_colored, (x1, y1), (x2, y2), (255, 0, 0), 2)

        intersection_points = []
        for h in horizontal_lines:
            for v in vertical_lines:
                inters = intersection((h[0], h[1]), (h[2], h[3]), (v[0], v[1]), (v[2], v[3]))
                intersection_points.append(inters)

        # Cluster intersection points and calculate centroids
        centroids = cluster_intersection_points(intersection_points, eps=50, min_samples=1)

        # Filter and order goal line points based on neighbor checking
        goal_line_points = filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=30)
        tracking_points = order_goal_line_points(goal_line_points)

        # Print ordered goal line points
        print("Ordered Goal Line Points:")
        for point in tracking_points:
            print(point)

        # Compute homography with the ordered points
        H = compute_homography(tracking_points)

    # If previous frame exists, use optical flow to track points
    if prev_frame is not None and tracking_points is not None:
        tracked_points = track_points(prev_frame, frame, tracking_points)
        if tracked_points:
            tracking_points = tracked_points  # Update tracked points
            H = compute_homography(tracking_points)  # Recalculate homography matrix

    # Overlay the advertisement if homography is valid
    if H is not None:
        frame = overlay_advertisement(frame, ad_image, H)

    return frame

# Compute homography matrix based on selected or tracked points
def compute_homography(video_points):
    field_points = np.array([
        [0, 0], [7.32, 0], [0, 5.5], [7.32, 5.5]
    ], dtype="float32")

    # # Real-world coordinates for the goal post front face (vertical plane)
    # field_points = np.array([
    #     [0, 0],         # Point 5: Bottom-left of front face of goal post
    #     [7.32, 0],      # Point 6: Bottom-right of front face of goal post
    #     [0, 2.44],      # Point 7: Top-left of front face of goal post
    #     [7.32, 2.44]    # Point 8: Top-right of front face of goal post
    # ], dtype="float32")

    video_points = np.array(video_points, dtype="float32")
    H, status = cv2.findHomography(field_points, video_points, cv2.RANSAC)
    return H
