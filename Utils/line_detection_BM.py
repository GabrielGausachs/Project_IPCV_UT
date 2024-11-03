import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN

# Global variables for tracking
tracking_points = None
tracking_gp_points = None
H = None

# Real-world coordinates for the goal post front face (3D reference)
goal_post_points_3D = np.array([
    [0, 0, 0],         # Point 5: Bottom-left of goal post
    [7.32, 0, 0],      # Point 6: Bottom-right of goal post
    [0, 2.44, 0],      # Point 7: Top-left of goal post
    [7.32, 2.44, 0]    # Point 8: Top-right of goal post
], dtype="float32")

# Real-world coordinates for the ad placement (3D reference)
ad_points_3D = np.array([
    [1.0, -3.5, 0],    # Bottom-left of the ad
    [5.0, -3.5, 0],    # Bottom-right of the ad
    [1.0, -1.25, 0],   # Top-left of the ad
    [5.0, -1.25, 0]    # Top-right of the ad
], dtype="float32")

goal_line_points_3D = np.array([
    [0, 0, 0],          # Leftmost point of goal line
    [0, 5.5, 0],        # Left point, 5.5 meters back
    [18.32, 0, 0],      # Rightmost point of goal line
    [18.32, 5.5, 0]     # Right point, 5.5 meters back
], dtype="float32")

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

    edges_small = cv2.Canny(blurred_small, 16, 37)
    edges_large = cv2.Canny(blurred_large, 26, 47)

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
def filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=50):
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
    if len(points) < 4:
        print("Error: Expected at least 4 points for ordering, but found", len(points))
        return []

    # If more than 4 points, select the best 4 (e.g., based on spatial distribution)
    if len(points) > 4:
        # You can implement a method to select the best 4 points
        # For simplicity, let's select the 4 points with the lowest y-values (closest to the bottom)
        points = sorted(points, key=lambda p: p[1])[:4]

    # Now proceed to order the 4 points
    sorted_points = sorted(points, key=lambda p: p[1])
    bottom_points = sorted(sorted_points[:2], key=lambda p: p[0])
    top_points = sorted(sorted_points[2:], key=lambda p: p[0])

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

    ad_points_real_world = np.array([ # [0, 0], [0, 5.5], [18.32, 0], [18.32, 5.5]
        [-6.0, 6.6],    # Bottom-Left
        [-6.0, 8.5],    # Top-Left
        [ 0.0, 6.6],    # Bottom-right
        [0.0, 8.5]      # Top-right
    ], dtype="float32")

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

# Main function to detect intersections, filter for goal line points, and apply tracking
def detecting_lines_intersection_points(frame, prev_frame, ad_image):
    global tracking_points, H, tracking_gp_points

    # Perform goal line detection only on the first frame if tracking_points is None
    if tracking_points is None:
        edges = detect_lines(frame)
        vertical_lines = []
        horizontal_lines = []

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=95, minLineLength=145, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                if -10 < angle < 10:
                    horizontal_lines.append((x1, y1, x2, y2))
                else:
                    vertical_lines.append((x1, y1, x2, y2))

        intersection_points = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                inters = intersection(
                    (h_line[0], h_line[1]), (h_line[2], h_line[3]),
                    (v_line[0], v_line[1]), (v_line[2], v_line[3])
                )
                if inters is not None:
                    intersection_points.append(inters)

        # Cluster intersection points and calculate centroids
        centroids = cluster_intersection_points(intersection_points, eps=50, min_samples=1)
        goal_line_points = filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=50)
        tracking_points = order_goal_line_points(goal_line_points)
        H = compute_homography(tracking_points)
        # Use finding_goalpost to get bottom and top points of the goal post
        goal_post_points = finding_goalpost(frame, H)
        print("Goal post points:", goal_post_points)
        mark_real_world_points(frame, goal_post_points)
        tracking_gp_points = goal_post_points

        # Calculate the projection matrix with the combined points
        P = compute_projection_matrix(goal_post_points, goal_post_points_3D)
        
        # P = compute_projection_matrix(tracking_gp_points, goal_post_points_3D)
        # compute_goal_post_reprojection_error(P, normalize_points_2D(goal_post_points), normalize_points_3D(goal_post_points_3D))
        # K, R, t = decompose_projection_matrix(P)

        # print("Intrinsic Matrix (K):\n", K)
        # print("Rotation Matrix (R):\n", R)
        # print("Translation Vector (t):\n", t)
       
    # If previous frame exists, use optical flow to track points
    if prev_frame is not None and tracking_points is not None and tracking_gp_points is not None:
        tracked_points = track_points(prev_frame, frame, tracking_points)
        tracked_gp_points = track_points(prev_frame, frame, tracking_gp_points)
        if tracked_points:
            tracking_points = tracked_points  # Update tracked points
            H = compute_homography(tracking_points)  # Recalculate homography matrix
        if tracked_gp_points:
            tracking_gp_points = np.array(tracked_gp_points, dtype="float32")
            # mark_real_world_points(frame, tracking_gp_points)
            P = compute_projection_matrix(tracking_gp_points, goal_post_points_3D)
            compute_goal_post_reprojection_error(P, goal_post_points_3D, tracking_gp_points)
        
    # if H is not None:
    #     frame = overlay_advertisement(frame, ad_image, H)

    if P is not None:
        frame = project_ad(frame, ad_image, ad_points_3D, P)
    
    return frame

# Compute homography matrix based on selected or tracked points
def compute_homography(video_points):
    field_points = np.array([
        [0, 0], [0, 5.5], [18.32, 0], [18.32, 5.5]
    ], dtype="float32")

    video_points = np.array(video_points, dtype="float32")
    H, status = cv2.findHomography(field_points, video_points, cv2.RANSAC)
    return H

def finding_goalpost(frame, homography_matrix):
    # Define the specific real-world points for the goalpost area and bottom points
    goalpostarea_world_points = np.array([
        [5.0, 5.5],   # First point to mark
        [14.0, 5.5]   # Second point to mark
    ], dtype="float32")

    goalpost_bottom_world_points = np.array([
        [5.7, 5.5],   # First point to mark
        [13.1, 5.5]   # Second point to mark
    ], dtype="float32")

    # Convert the real-world points to image coordinates
    video_points = cv2.perspectiveTransform(np.array([goalpostarea_world_points], dtype="float32"), homography_matrix)[0]
    bottom_goalpost_points = cv2.perspectiveTransform(np.array([goalpost_bottom_world_points], dtype="float32"), homography_matrix)[0]

    # Cropping the area between the bottom goal points to the top of the image
    min_x = int(min(video_points[0][0], video_points[1][0]))
    max_x = int(max(video_points[0][0], video_points[1][0]))
    bottom_y = int(video_points[1][1])

    # Define the cropping rectangle: from bottom of goal line to the top of the frame
    cropped_frame = frame[:bottom_y, min_x:max_x]

    # Edge detection on cropped image
    gray_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray_cropped, 200, 255, cv2.THRESH_BINARY)
    white_regions = cv2.bitwise_and(gray_cropped, gray_cropped, mask=white_mask)
    blurred_small = cv2.GaussianBlur(white_regions, (3, 3), 1)
    edges = cv2.Canny(blurred_small, 70, 90)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=75, minLineLength=20, maxLineGap=10)
    
    # Find vertical lines near the goalpost bottom points
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if 70 < abs(angle) < 100:
                vertical_lines.append((x1, y1, x2, y2))

    # Identify top points for goalposts
    top_points = []
    for bottom_point in bottom_goalpost_points:
        closest_line = None
        min_distance = float('inf')
        highest_y = bottom_y

        for x1, y1, x2, y2 in vertical_lines:
            line_mid_x = (x1 + x2) / 2
            distance = abs(line_mid_x - (bottom_point[0] - min_x))
            if distance < min_distance:
                min_distance = distance
                highest_y = min(y1, y2)
                closest_line = (x1, y1, x2, y2)

        if closest_line:
            adjusted_top_point = (closest_line[0] + min_x, highest_y + (bottom_y - cropped_frame.shape[0]))
            top_points.append(adjusted_top_point)
    

    # cv2.imshow("Line and Intersection Detection on Cropped Goal Area", cropped_frame)
    # cv2.waitKey(0)

    # Combine bottom and top points into a single array
    goal_post_points = np.array([
        bottom_goalpost_points[0],  # Bottom-left
        top_points[0],              # Top-left
        bottom_goalpost_points[1],  # Bottom-right
        top_points[1]               # Top-right
    ], dtype="float32")

    return goal_post_points

# Function to mark real-world points on the video frame
def mark_real_world_points(frame, points):
    
    # Draw marks on the frame at the transformed coordinates
    for point in points:
        x, y = int(point[0]), int(point[1])
        # Draw a circle at the transformed point
        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red circle
        # Alternatively, draw an X mark if preferred
        cv2.line(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 0), 2)  # Green "X" mark
        cv2.line(frame, (x + 5, y - 5), (x - 5, y + 5), (0, 255, 0), 2)  # Green "X" mark

def compute_projection_matrix(goal_post_points, goal_post_points_3D):
    """
    Calculate the camera projection matrix P using the 3D coordinates of the goal post 
    and their corresponding 2D image coordinates.
    """
    num_points = goal_post_points.shape[0]
    A = []
    
    for i in range(num_points):
        X, Y, Z = goal_post_points_3D[i]
        u, v = goal_post_points[i]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # Use SVD to solve for P
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  # The last row of Vt gives the solution
    
    return P

def decompose_projection_matrix(P):
    """
    Decompose the projection matrix P to retrieve intrinsic matrix K, 
    rotation matrix R, and translation vector t.
    """
    # Extract the intrinsic and extrinsic parameters from P
    M = P[:, :3]
    K, R = np.linalg.qr(np.linalg.inv(M))
    
    # Ensure the diagonal of K has positive values
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Normalize K so that K[2, 2] = 1
    K /= K[2, 2]
    
    # Compute translation vector t
    t = np.linalg.inv(K) @ P[:, 3]
    
    return K, R, t

def project_ad(frame, ad_image, ad_points_3D, P):
    """
    Project the ad onto the scene using the projection matrix P.
    """

    
    # Rotate the ad image 90 degrees clockwise
    ad_image = cv2.rotate(ad_image, cv2.ROTATE_90_CLOCKWISE)

    # Convert 3D ad points to homogeneous coordinates
    ad_points_3D_homogeneous = np.hstack((ad_points_3D, np.ones((ad_points_3D.shape[0], 1))))
    
    # Project the 3D ad points to 2D using P
    ad_points_2D_homogeneous = P @ ad_points_3D_homogeneous.T
    ad_points_2D = (ad_points_2D_homogeneous[:2] / ad_points_2D_homogeneous[2]).T  # Convert to 2D
    
       # Debugging: Print projected ad points in 2D
    print("Projected ad points (2D):", ad_points_2D)

    # Prepare ad image corners for perspective transformation
    ad_img_height, ad_img_width = ad_image.shape[:2]
    ad_corners_src = np.array([
        [0, 0],
        [ad_img_width, 0],
        [0, ad_img_height],
        [ad_img_width, ad_img_height]
    ], dtype="float32")
    
    # Compute the perspective transform to fit the ad in the scene
    transform_matrix = cv2.getPerspectiveTransform(ad_corners_src, ad_points_2D.astype("float32"))
    warped_ad = cv2.warpPerspective(ad_image, transform_matrix, (frame.shape[1], frame.shape[0]))
    
    # Create a mask for blending the ad
    ad_mask = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
    _, ad_mask = cv2.threshold(ad_mask, 1, 255, cv2.THRESH_BINARY)
    frame_background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(ad_mask))
    frame_with_ad = cv2.add(frame_background, warped_ad)

    return frame_with_ad

def compute_goal_post_reprojection_error(P, goal_post_points_3D, goal_post_points_2D):
    """
    Calculate the reprojection error using only the goal post points to verify calibration accuracy.
    """
    # Convert 3D goal post points to homogeneous coordinates
    goal_post_3D_homogeneous = np.hstack((goal_post_points_3D, np.ones((goal_post_points_3D.shape[0], 1))))
    
    # Project the goal post 3D points to 2D using P
    goal_post_2D_homogeneous = P @ goal_post_3D_homogeneous.T
    goal_post_2D_projected = (goal_post_2D_homogeneous[:2] / goal_post_2D_homogeneous[2]).T  # Convert to 2D
    
    # Calculate reprojection error for each goal post point
    error = np.linalg.norm(goal_post_points_2D - goal_post_2D_projected, axis=1)
    mean_reprojection_error = np.mean(error)
    
    print("Goal post projected points (2D):", goal_post_2D_projected)
    print("Reprojection error for each goal post point:", error)
    print("Mean reprojection error (goal post points):", mean_reprojection_error)
    
    return mean_reprojection_error
