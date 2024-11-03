import cv2
import numpy as np
import math


def mask_field(frame, num=10):
    # Convert the image to the HSV color space
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for the hue channel
    hue_hist = cv2.calcHist([im_hsv], [0], None, [180], [0, 180])
    peak_idx = np.argmax(hue_hist)  # Get the dominant hue index

    # Determine hue range around the peak index
    min_hue = max(peak_idx - num, 0)
    max_hue = min(peak_idx + num, 179)

    print(min_hue, max_hue)

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
    kernel = np.ones((3, 3), np.uint8)
    final_edges = cv2.morphologyEx(
        combined_edges, cv2.MORPH_CLOSE, kernel, iterations=1
    )

    kernel = np.ones((3, 3), np.uint8)
    final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return final_edges


def get_hough_lines(edge_frame: np.ndarray, original_frame: np.ndarray) -> tuple:
    vertical_lines = []
    horizontal_lines = []

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        edge_frame, 1, np.pi / 180, 120, minLineLength=80, maxLineGap=8
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Adjusted angle threshold for better classification
            if -10 < angle < 10:  # Horizontal lines
                horizontal_lines.append((x1, y1, x2, y2, length, angle))
                cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(edge_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif (
                (80 < abs(angle) < 100)
                or (angle > 10 and angle < 80)
                or (angle < -10 and angle > -80)
            ):  # Allow near-vertical lines
                vertical_lines.append((x1, y1, x2, y2, length, angle))
                cv2.line(original_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.line(edge_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return vertical_lines, horizontal_lines, original_frame, edge_frame
