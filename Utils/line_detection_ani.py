import cv2
import numpy as np


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
    lower_green = np.array([38, 45, 0])  # Adjust these values as needed
    upper_green = np.array([55, 255, 220])
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

    # Convert masked image to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Step 4: Combine Laplacian and Canny edge detection with Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # # Laplacian edge detection
    # laplacian_edges = cv2.Laplacian(blurred, cv2.CV_8U, delta=1, ksize=1)

    # Canny edge detection
    low_threshold = 40
    high_threshold = 70
    processed_frame = cv2.Canny(blurred, low_threshold, high_threshold)

    # Convert Laplacian and Canny results to 3-channel (BGR) if needed
    if len(processed_frame.shape) == 2:  # Convert single-channel Canny to BGR
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Combine Laplacian edges with Canny edges
    # processed_frame = cv2.addWeighted(processed_frame, 0.5, laplacian_edges, 0.5, 0)

    return processed_frame
