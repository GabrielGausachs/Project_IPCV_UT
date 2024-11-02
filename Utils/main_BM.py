import cv2
import sys
import numpy as np
import os
from Utils.config import (
    INPUT_FOLDER,
    INPUT_NAME_VIDEO,
    OUTPUT_FOLDER,
    OUTPUT_NAME_VIDEO
)
from Utils.line_detection import line_detection
from Utils.line_detection_BM import detecting_lines_intersection_points

def main(input_video_file: str, output_video_file: str):
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Load the advertisement image
    ad_image = cv2.imread('ads3.png')  # Replace with the actual path to your ad image
    if ad_image is None:
        print("Error: Advertisement image not found.")
        sys.exit(1)

    # Create a resizable window
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    prev_frame = None

    # Main loop for processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale for tracking
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect lines, intersections, and compute homography to overlay ad
        frame_with_ad = detecting_lines_intersection_points(frame, prev_frame, ad_image)

        # Display the frame with the ad overlay and write to output
        cv2.imshow('Frame', frame_with_ad)
        out.write(frame_with_ad)

        # Update previous frame for optical flow tracking
        prev_frame = gray_frame.copy()

        # Press 'q' to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video objects and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video = os.path.join(INPUT_FOLDER, INPUT_NAME_VIDEO)
    output_video = os.path.join(OUTPUT_FOLDER, OUTPUT_NAME_VIDEO)
    main(input_video, output_video)
