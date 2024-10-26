import cv2
import numpy as np

# Load the image
image = cv2.imread('sample_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

# Apply Canny Edge Detection with adjusted thresholds
edges = cv2.Canny(blurred, 30, 100)  # Try lowering these values if needed

# Use Hough Line Transformation to detect straight lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Draw the detected lines on the original image
line_image = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show results
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()