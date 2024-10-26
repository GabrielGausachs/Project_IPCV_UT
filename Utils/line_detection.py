import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def field_mask(frame):
    # Assuming `I` is your input image in RGB format (e.g., loaded using matplotlib or OpenCV and converted to RGB)
    # Convert to HSV
    im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Calculate histogram for the Hue channel (index 0)
    # `im_hsv[:,:,0]` extracts the Hue channel
    counts, bin_edges = np.histogram(im_hsv[:, :, 0], bins=256, range=(0, 256))

    lower_green = np.array([50, 40, 40])   # Adjusted lower bound based on the histogram
    upper_green = np.array([80, 255, 255]) # Adjusted upper bound

    mask = cv2.inRange(im_hsv, lower_green, upper_green)

    num_labels, labels_im = cv2.connectedComponents(mask)

    # Initialize variables to find the largest component
    largest_area = 0
    largest_label = -1

    # Iterate through all labels to find the largest component
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        area = np.sum(labels_im == label)
        if area > largest_area:
            largest_area = area
            largest_label = label

    largest_mask = np.zeros_like(mask)  # Create a blank mask

    # Fill the largest connected component in white
    largest_mask[labels_im == largest_label] = 255
        
    # Define a kernel for dilation
    #kernel = np.ones((17, 17), np.uint8)  # You can adjust the kernel size as needed

    # Apply dilation to the mask
    #field_mask = cv2.dilate(largest_mask, kernel)

    # Optional: Plot histogram to visualize
    #plt.plot(bin_edges[:-1], counts)  # bin_edges[:-1] gives the centers of bins
    #plt.title("Hue Histogram")
    #plt.xlabel("Hue value")
    #plt.ylabel("cvtColorCount")
    #plt.show()

    field_mask = cv2.cvtColor(largest_mask, cv2.COLOR_GRAY2BGR)

    return field_mask

def edge_detection(frame):
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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring element for morphological operations
    processed_frame = cv2.morphologyEx(processed_frame, cv2.MORPH_CLOSE, kernel)

    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    print(processed_frame.shape)

    return processed_frame


def line_detection(frame):
    # Assume field_mask is defined elsewhere
    fieldmask = field_mask(frame)
    
    # Extract only the field using the mask
    only_field = cv2.bitwise_and(frame, fieldmask)

    #gray = cv2.cvtColor(only_field, cv2.COLOR_BGR2GRAY)

    frame = edge_detection(only_field)
    _, frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring element for morphological operations
    #frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 1)

    # Apply Canny Edge Detection with adjusted thresholds
    edges = cv2.Canny(blurred, 30, 120)  # Try lowering these values if needed

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring element for morphological operations
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    """
    
    # THRESHOLDING
    #_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring element for morphological operations
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    
    # Use Hough Line Transformation to detect straight lines
    lines = cv2.HoughLinesP(frame, rho=1, theta=np.pi/180, threshold=50, minLineLength=80, maxLineGap=30)
    print(lines.shape)
    # Draw the detected lines on the original image
    line_image = only_field.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return line_image

