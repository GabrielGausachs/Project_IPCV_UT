import cv2
import numpy as np
import math

# Global variables for tracking
clicked_points = []
tracking_points = None
H = None

# Function to select points for initial homography calculation
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Point {len(clicked_points)}: ({x}, {y})")

# Track points using optical flow
def track_points(prev_frame, current_frame, points):
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(points, dtype="float32").reshape(-1, 1, 2)
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None)
    new_points = [tuple(pt.ravel()) for pt, s in zip(current_points, status) if s == 1]
    return new_points if len(new_points) == len(points) else None

# Overlay advertisement on the field using homography matrix
def overlay_advertisement(frame, ad_image, homography_matrix):
    # Rotate the ad image by 90 degrees
    rotated_ad_image = cv2.rotate(ad_image, cv2.ROTATE_90_CLOCKWISE)

    # Define the new target location in real-world coordinates, moving it slightly up and left
    # Adjust the values to place the ad outside the field, near the top-left boundary
    ad_points_real_world = np.array([
        [8.0, -2.0],  # Adjust this to move left or right along the sideline
        [11.0, -2.0],  
        [8.0, -0.25],  
        [11.0, -0.25]   
    ], dtype="float32")

    # Enable this to use goal post as reference real-world coordinates for the ad placement
    # ad_points_real_world = np.array([
    #     [1.0, -2.5],   # Bottom-left of ad near the sideline
    #     [5.0, -2.5],  # Bottom-right of ad near the sideline
    #     [1.0, -0.25],  # Top-left of ad near the goal line
    #     [5.0, -0.25]  # Top-right of ad near the goal line
    # ], dtype="float32")

    # Transform the real-world ad points to video coordinates using the homography matrix
    ad_points_video = cv2.perspectiveTransform(np.array([ad_points_real_world]), homography_matrix)[0]

    # Define the corners of the rotated ad image
    ad_height, ad_width = rotated_ad_image.shape[:2]
    ad_corners = np.array([
        [0, 0],                      # Top-left corner
        [ad_width, 0],               # Top-right corner
        [0, ad_height],              # Bottom-left corner
        [ad_width, ad_height]        # Bottom-right corner
    ], dtype="float32")

    # Compute the perspective transform to place the rotated ad onto the field
    transform_matrix = cv2.getPerspectiveTransform(ad_corners, ad_points_video)
    warped_ad = cv2.warpPerspective(rotated_ad_image, transform_matrix, (frame.shape[1], frame.shape[0]))

    # Create a mask to blend the advertisement into the frame
    ad_mask = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
    _, ad_mask = cv2.threshold(ad_mask, 1, 255, cv2.THRESH_BINARY)

    # Remove the background area where the ad will be placed
    frame_background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(ad_mask))

    # Overlay the advertisement onto the frame
    result_frame = cv2.add(frame_background, warped_ad)

    return result_frame

# Detect lines, calculate homography, and overlay ad with updated homography
def detecting_lines_intersection_points(frame, prev_frame, ad_image):
    global tracking_points, H

    # Track points if previous frame and points are available
    if prev_frame is not None and tracking_points is not None:
        tracked_points = track_points(prev_frame, frame, tracking_points)
        if tracked_points:
            tracking_points = tracked_points  # Update tracked points
            H = compute_homography(tracking_points)  # Recalculate homography matrix
            # Overlay the advertisement if homography is valid
            if H is not None:
                frame = overlay_advertisement(frame, ad_image, H)
    
    else:
        # Display initial points selection for homography calculation
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', select_points)
        
        # Wait for four points to be selected
        while len(clicked_points) < 4:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Once points are selected, calculate the initial homography and start tracking
        if len(clicked_points) == 4:
            tracking_points = clicked_points[:]
            H = compute_homography(tracking_points)
            if H is not None:
                frame = overlay_advertisement(frame, ad_image, H)

    return frame

# Compute homography matrix based on selected or tracked points
def compute_homography(video_points):
    field_points = np.array([
        [0, 0], [7.32, 0], [0, 5.5], [7.32, 5.5]
    ], dtype="float32")

    # Enable this to use Real-world coordinates for the goal post front face (vertical plane)
    # field_points = np.array([
    #     [0, 0],         # Point 5: Bottom-left of front face of goal post
    #     [7.32, 0],      # Point 6: Bottom-right of front face of goal post
    #     [0, 2.44],      # Point 7: Top-left of front face of goal post
    #     [7.32, 2.44]    # Point 8: Top-right of front face of goal post
    # ], dtype="float32")
    video_points = np.array(video_points, dtype="float32")
    H, status = cv2.findHomography(field_points, video_points, cv2.RANSAC)
    return H
