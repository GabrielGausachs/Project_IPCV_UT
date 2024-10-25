import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    kernel = np.ones((17, 17), np.uint8)  # You can adjust the kernel size as needed

    # Apply dilation to the mask
    field_mask = cv2.dilate(largest_mask, kernel)

    # Optional: Plot histogram to visualize
    #plt.plot(bin_edges[:-1], counts)  # bin_edges[:-1] gives the centers of bins
    #plt.title("Hue Histogram")
    #plt.xlabel("Hue value")
    #plt.ylabel("Count")
    #plt.show()

    #plt.figure(figsize=(8, 8))
    #plt.title('Green Mask')
    #plt.imshow(field_mask, cmap='gray')  # Display the mask in grayscale
    #plt.axis('off')  # Hide axes
    #plt.show()

    field_mask = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)

    return field_mask
