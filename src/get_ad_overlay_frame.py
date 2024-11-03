import cv2
import numpy as np


def overlay_ad_on_frame(frame, ad_image, ad_position_video):
    # Step 1: Define the source points from the ad image
    ad_height, ad_width = ad_image.shape[:2]
    print("Ad Height:", ad_height)
    print("Ad Width:", ad_width)

    ad_corners = np.array(
        [[0, 0], [ad_width, 0], [ad_width, ad_height], [0, ad_height]], dtype="float32"
    )

    # Step 2: Define destination points where we want to place the ad
    ad_position_video = np.array(ad_position_video, dtype="float32")

    # Step 3: Compute the homography to warp ad_image to the destination points in the frame
    homography_matrix, _ = cv2.findHomography(ad_corners, ad_position_video)

    # Step 4: Warp the ad image to the frame's perspective
    warped_ad = cv2.warpPerspective(
        ad_image, homography_matrix, (frame.shape[1], frame.shape[0])
    )

    # Step 5: Create a mask from the warped ad for blending
    ad_mask = cv2.cvtColor(warped_ad, cv2.COLOR_BGR2GRAY)
    _, ad_mask = cv2.threshold(ad_mask, 1, 255, cv2.THRESH_BINARY)

    # Check shapes before combining
    print("Frame shape:", frame.shape)
    print("Warped Ad shape:", warped_ad.shape)
    print("Ad Mask shape:", ad_mask.shape)

    # Step 6: Resize the mask to match the frame size if necessary
    if ad_mask.shape[:2] != frame.shape[:2]:
        ad_mask = cv2.resize(ad_mask, (frame.shape[1], frame.shape[0]))

    # Use the mask to remove the background from the frame where the ad will be placed
    frame_background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(ad_mask))

    # Step 7: Combine the background and the warped ad
    result_frame = cv2.add(frame_background, warped_ad)

    return result_frame
