import cv2


def detect_lines(frame):
    # Step 4: Combine Laplacian and Canny edge detection with Gaussian blur
    blurred = cv2.GaussianBlur(frame, (1, 1), 3)

    # Laplacian edge detection
    laplacian_edges = cv2.Laplacian(blurred, cv2.CV_8U, delta=3, ksize=5)

    # Canny edge detection
    low_threshold = 40
    high_threshold = 70
    canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Convert Laplacian and Canny results to 3-channel (BGR) if needed
    if len(canny_edges.shape) == 2:  # Convert single-channel Canny to BGR
        canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

    # Combine Laplacian edges with Canny edges
    processed_frame = cv2.addWeighted(canny_edges, 0.5, laplacian_edges, 0.5, 0)

    return processed_frame
