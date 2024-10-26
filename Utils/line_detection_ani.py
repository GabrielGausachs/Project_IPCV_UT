import cv2


def line_detection(frame):
    # Step 4: Combine Laplacian and Canny edge detection with Gaussian blur
    blurred = cv2.GaussianBlur(frame, (3, 3), 2)

    # Laplacian edge detection
    laplacian_edges = cv2.Laplacian(blurred, cv2.CV_8U, delta=1)

    # Canny edge detection
    low_threshold = 100
    high_threshold = 150
    canny_edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Convert Laplacian and Canny results to 3-channel (BGR) if needed
    if len(canny_edges.shape) == 2:  # Convert single-channel Canny to BGR
        canny_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

    # Combine Laplacian edges with Canny edges
    processed_frame = cv2.addWeighted(canny_edges, 0.5, laplacian_edges, 0.5, 0)

    return processed_frame
