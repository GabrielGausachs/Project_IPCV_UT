import cv2
import numpy as np
import math

# from skimage.morphology import skeletonize


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
    edges_small = cv2.Canny(blurred_small, 40, 60)
    edges_large = cv2.Canny(blurred_large, 40, 60)

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

    # Display intermediate results
    # cv2.imshow("Dilated Edges", dilated_edges)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(final_edges, cv2.COLOR_GRAY2BGR)
    edge_frame = np.copy(cdst)

    linesP = cv2.HoughLinesP(
        final_edges, 1, np.pi / 180, 120, minLineLength=100, maxLineGap=20
    )

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(
                edge_frame, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA
            )

    # cv2.imshow("Frame", frame)
    # cv2.imshow("Edge Frame", edge_frame)

    return frame, edge_frame


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
