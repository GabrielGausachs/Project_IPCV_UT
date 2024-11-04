import cv2
import numpy as np


def get_ad_position_field(
    ad_image,
    field_scale=1.0,
    field_size=(105, 68),
    field_padding=5,
    ad_side="right",
    banner_scale=0.75,
    v_padding=10,
    h_padding=0.1,
):
    goal_height = 7.3 * field_scale
    field_length, field_width = field_size
    field_length = field_length * field_scale
    field_width = field_width * field_scale

    # Set padding
    v_padding = v_padding * field_scale
    h_padding = h_padding * field_scale

    # Load the banner image using OpenCV
    width, height = ad_image.shape[:2]

    # Calculate aspect ratio and new dimensions
    aspect_ratio = height / width
    banner_width = field_padding * field_scale
    banner_height = banner_width * aspect_ratio  # Maintain the banner's aspect ratio
    banner_height = banner_height * banner_scale
    banner_width = banner_width * banner_scale

    # Position the banner outside the specified goal line
    if ad_side == "left":
        rotated_ad_image = cv2.rotate(ad_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Outside the left goal line
        x0, x1 = -banner_width - h_padding, -h_padding
    else:
        rotated_ad_image = cv2.rotate(ad_image, cv2.ROTATE_90_CLOCKWISE)
        # Outside the right goal line
        x0, x1 = field_length + h_padding, field_length + banner_width + h_padding

    y0, y1 = (field_width / 2) + (goal_height / 2) + banner_height + v_padding, (
        field_width / 2
    ) + (goal_height / 2) + v_padding

    # Place the image on the field with automatic scaling
    points = [
        [x0, y0],
        [x1, y0],
        [x0, y1],
        [x1, y1],
    ]

    return (
        np.array(points, dtype="float32"),
        rotated_ad_image,
    )


def get_ad_position_video(ad_position_field, H):
    # Project to video coordinates using the inverse homography
    ad_position_video = cv2.perspectiveTransform(np.array([ad_position_field]), H)[0]

    return np.array(
        ad_position_video,
        dtype="float32",
    )
