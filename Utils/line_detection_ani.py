import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN

def mask_field(frame, num=10):
    # Convert the image to the HSV color space
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for the hue channel
    hue_hist = cv2.calcHist([im_hsv], [0], None, [180], [0, 180])
    peak_idx = np.argmax(hue_hist)  # Get the dominant hue index

    # Determine hue range around the peak index
    min_hue = max(peak_idx - num, 0)
    max_hue = min(peak_idx + num, 179)

    # # Define range for green color (field)
    lower_green = np.array([min_hue, 50, 50])  # Adjust these values as needed
    upper_green = np.array([max_hue, 255, 255])
    field_mask = cv2.inRange(im_hsv, lower_green, upper_green)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Ignore background
    field_mask = (labels == largest_label).astype(np.uint8)

    # Apply the combined mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=field_mask)

    return masked_frame


def detect_lines(frame):
    # Convert to HSV and mask green areas
    masked_frame = mask_field(frame)

    # Convert frame to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurs with different sigma values
    blurred_small = cv2.GaussianBlur(gray, (3, 3), 1)  # Small sigma
    blurred_large = cv2.GaussianBlur(gray, (5, 5), 3)  # Larger sigma

    # Perform edge detection on each blurred image
    edges_small = cv2.Canny(blurred_small, 40, 70)
    edges_large = cv2.Canny(blurred_large, 80, 100)

    # Combine edges from both scales
    combined_edges = cv2.bitwise_or(edges_small, edges_large)

    # Use morphological operations to clean up the edges
    kernel = np.ones((5, 5), np.uint8)
    final_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    final_edges = cv2.erode(final_edges, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    final_edges = cv2.dilate(final_edges, kernel, iterations=2)
    final_edges = cv2.erode(final_edges, kernel, iterations=2)

    final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel)

    return final_edges


def intersection(o1, p1, o2, p2):
    """Find the intersection of two line segments (o1, p1) and (o2, p2).

    Args:
        o1: Starting point of the first line segment (x1, y1).
        p1: Ending point of the first line segment (x2, y2).
        o2: Starting point of the second line segment (x3, y3).
        p2: Ending point of the second line segment (x4, y4).

    Returns:
        A tuple (x, y) of the intersection point if it exists,
        or None if the line segments do not intersect.
    """
    # Convert points to numpy arrays
    o1, p1, o2, p2 = map(np.array, [o1, p1, o2, p2])

    # Direction vectors
    d1 = p1 - o1  # Vector for the first line segment
    d2 = p2 - o2  # Vector for the second line segment
    x = o2 - o1  # Vector from the start of line 1 to the start of line 2

    # Calculate the cross product
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    # Check if lines are parallel
    if abs(cross) < 1e-8:
        return None  # Lines are parallel, no intersection

    # Calculate t1 parameter
    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross

    # Calculate intersection point
    r = o1 + d1 * t1
    if (
        min(o1[0], p1[0]) <= r[0] <= max(o1[0], p1[0])
        and min(o1[1], p1[1]) <= r[1] <= max(o1[1], p1[1])
        and min(o2[0], p2[0]) <= r[0] <= max(o2[0], p2[0])
        and min(o2[1], p2[1]) <= r[1] <= max(o2[1], p2[1])
    ):
        return (int(round(r[0])), int(round(r[1])))
    return None


def is_duplicate_angle(angle, existing_lines, angle_tolerance=3.5):

    for existing in existing_lines:
        existing_angle = existing[5]

        # Check if the angles are within the specified tolerance
        if abs(existing_angle - angle) < angle_tolerance:
            return True
    return False

def cluster_intersection_points(intersection_points, eps=10, min_samples=1):
    """Cluster intersection points that are close to each other and return centroids."""
    # Remove None values from intersection_points
    points = np.array([p for p in intersection_points if p is not None])

    if len(points) == 0:
        return []  # No valid intersection points

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Calculate centroids of each cluster
    centroids = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue  # Skip noise points

        # Find points in the current cluster
        cluster_points = points[clustering.labels_ == cluster_id]

        # Calculate the centroid
        centroid = cluster_points.mean(axis=0)
        centroids.append((int(round(centroid[0])), int(round(centroid[1]))))


    return centroids


def detecting_lines_intersection_points(frame):
    edges = detect_lines(frame)

    vertical_lines = []
    horizontal_lines = []

    frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=80, maxLineGap=10
    )
    # Draw the detected lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the angle of the line in degrees and the length
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if -10 < angle < 10:  # Tolerance for horizontal (close to 0 degrees)
                horizontal_lines.append((x1, y1, x2, y2, length, angle))
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Classify as vertical if angle is close to 90 or -90 degrees, otherwise horizontal
            else:  # Tolerance for vertical (close to 90 degrees)
                vertical_lines.append((x1, y1, x2, y2, length, angle))
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    horizontal_lines_sorted = sorted(
        horizontal_lines, key=lambda line: line[4], reverse=True
    )
    vertical_lines_sorted = sorted(
        vertical_lines, key=lambda line: line[4], reverse=True
    )
    intersection_points = []
    for h in horizontal_lines_sorted:
        for v in vertical_lines_sorted:
            inters = intersection(
                (h[0], h[1]), (h[2], h[3]), (v[0], v[1]), (v[2], v[3])
            )
            intersection_points.append(inters)
            #if inters:
            #    cv2.circle(frame, inters, radius=10, color=(0, 0, 255), thickness=1)

    centroids = cluster_intersection_points(intersection_points,50,1)
    print('Number of centroids:',len(centroids))
    #for cent in centroids:
    #    cv2.circle(frame, cent, radius=5, color=(0, 0, 255), thickness=10)

    centroids_sorted = sorted(centroids, key=lambda point: point[1])
    for cent in centroids_sorted[:4]:
        cv2.circle(frame, cent, radius=5, color=(0, 0, 255), thickness=10)
        # Prepare the text for coordinates
        text = f"({cent[0]}, {cent[1]})"
        
        # Define the position for the text
        text_position = (cent[0] + 10, cent[1] - 10)  # Slightly offset from the circle
        
        # Put the text on the image
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=3, color=(255, 255, 255), thickness=3)  # White text

    print(centroids_sorted)
    y_range = 100
    x_range = 700

    for i, (x1,y1) in enumerate(centroids_sorted):
        print(i)
        # Define the rectangle boundaries
        found_nearby = False
        for j, (x2, y2) in enumerate(centroids_sorted):
            if i != j:  # Skip the same point
                if abs(y2-y1) < 50:
                    if 1000 > abs(x2-x1) > 300:
                        found_nearby = True
                        break 

    image_height, image_width = frame.shape[:2]
    if found_nearby:
        # Calculate rectangle coordinates with boundary checks
        top_left = (max(0, x1 - x_range), max(0, y1 - y_range))
        bottom_right = (min(image_width - 1, x1 + x_range), min(image_height - 1, y1 + y_range))
        
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
    return frame


# def detect_lines(frame):

#     # Convert to HSV and mask green areas
#     masked_frame = mask_field(frame)

#     # Convert masked image to grayscale
#     gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

#     # Step 4: Combine Laplacian and Canny edge detection with Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#     # # Laplacian edge detection
#     # laplacian_edges = cv2.Laplacian(blurred, cv2.CV_8U, delta=1, ksize=1)

#     # Canny edge detection
#     low_threshold = 40
#     high_threshold = 70
#     processed_frame = cv2.Canny(blurred, low_threshold, high_threshold)

#     # Convert Laplacian and Canny results to 3-channel (BGR) if needed
#     if len(processed_frame.shape) == 2:  # Convert single-channel Canny to BGR
#         processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

#     # Combine Laplacian edges with Canny edges
#     # processed_frame = cv2.addWeighted(processed_frame, 0.5, laplacian_edges, 0.5, 0)

#     return processed_frame
