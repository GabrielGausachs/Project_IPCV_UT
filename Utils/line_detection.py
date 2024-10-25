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
    #kernel = np.ones((17, 17), np.uint8)  # You can adjust the kernel size as needed

    # Apply dilation to the mask
    #field_mask = cv2.dilate(largest_mask, kernel)

    # Optional: Plot histogram to visualize
    #plt.plot(bin_edges[:-1], counts)  # bin_edges[:-1] gives the centers of bins
    #plt.title("Hue Histogram")
    #plt.xlabel("Hue value")
    #plt.ylabel("Count")
    #plt.show()

    field_mask = cv2.cvtColor(largest_mask, cv2.COLOR_GRAY2BGR)

    return field_mask

import cv2
import numpy as np

def line_detection(frame):
    # Assume field_mask is defined elsewhere
    fieldmask = field_mask(frame)
    
    # Extract only the field using the mask
    only_field = cv2.bitwise_and(frame, fieldmask)


    gray = cv2.cvtColor(only_field, cv2.COLOR_BGR2GRAY)

    """
    # Initialize parameters
    tau = 8  # Distance in pixels to check
    threshold = 15  # Brightness threshold for classification

    # Create a binary image initialized to black
    binary_image = np.zeros_like(gray)

    # Get dimensions of the image
    height, width = gray.shape

    # Loop through each pixel in the image
    for y in range(tau, height - tau):
        for x in range(tau, width - tau):
            candidate_pixel = gray[y, x]
            surrounding_brightness = []
            
            # Check the surrounding pixels and add valid ones to the list
            if y - tau >= 0:  # Check above
                surrounding_brightness.append(gray[y - tau, x])
            if y + tau < height:  # Check below
                surrounding_brightness.append(gray[y + tau, x])
            if x - tau >= 0:  # Check left
                surrounding_brightness.append(gray[y, x - tau])
            if x + tau < width:  # Check right
                surrounding_brightness.append(gray[y, x + tau])

            # Only proceed if there are surrounding pixels to compare with
            if surrounding_brightness:
                avg_surrounding_brightness = np.mean(surrounding_brightness)

                # Check if the candidate pixel is significantly brighter than the average
                if candidate_pixel > avg_surrounding_brightness + threshold:
                    binary_image[y, x] = 255  # Classify as white pixel
                else:
                    binary_image[y, x] = 0    # Classify as black pixel
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    """
    
    # THRESHOLDING
    #_, binary = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Structuring element for morphological operations
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # CANNY EDGE DETECTION

    



    return gray

