import cv2
import numpy as np


def overlay_ad_on_frame(frame, rotated_ad_image, ad_points_video):
    # Define the ad corner points from the ad image
    ad_height, ad_width = rotated_ad_image.shape[:2]
    print("Ad Height:", ad_height)
    print("Ad Width:", ad_width)

    ad_corners = np.array(
        [[0, 0], [ad_width, 0], [0, ad_height], [ad_width, ad_height]],
        dtype="float32",
    )

    # Compute the perspective transform to place the rotated ad onto the field
    transform_matrix = cv2.getPerspectiveTransform(ad_corners, ad_points_video)
    warped_ad = cv2.warpPerspective(
        rotated_ad_image, transform_matrix, (frame.shape[1], frame.shape[0])
    )

    # Create a mask to blend the advertisement into the frame
    ad_mask = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
    _, ad_mask = cv2.threshold(ad_mask, 1, 255, cv2.THRESH_BINARY)

    # Remove the background area where the ad will be placed
    frame_background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(ad_mask))

    # Overlay the advertisement onto the frame
    result_frame = cv2.add(frame_background, warped_ad)

    return result_frame
