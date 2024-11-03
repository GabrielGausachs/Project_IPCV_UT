import cv2
from matplotlib import pyplot as plt
import numpy as np

from get_lines import get_lines
from get_intersections import get_intersections
from get_points import get_points
from get_calibration import get_calibration
from get_ad_position import get_ad_position_frame, get_ad_position_field
from get_ad_overlay_frame import overlay_ad_on_frame
from get_tracked_points import track_frame_points

# Load the video and read the first frame for line detection
# video_path = "football_videos/video_example1.mp4"
video_path = "football_videos/video5.mp4"
# video_path = "football_videos/soccer_video_example1.mp4"
output_path = "football_videos/output/output_ani.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
# Get width and height
frame_size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
ad_image = cv2.imread("banners/jumbo_banner.png", cv2.IMREAD_COLOR)
ad_side = "right"


field_size = (105, 68)

# Your original dimensions
frame_width, frame_height = frame_size
field_width, field_height = field_size

# Get the ad position in the field plane
ad_position_field, rotated_ad_image = get_ad_position_field(
    ad_image, field_padding=5, h_padding=1, side=ad_side, banner_scale=0.5
)
print("Ad Position in Field:", ad_position_field)


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
prev_frame = None
field_points = None
frame_points = None
ad_position_buffer = []  # Initialize a buffer to store ad positions
buffer_size = 10  # Adjust buffer size as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get lines and intersections
    (
        vertical_lines,
        horizontal_lines,
        annotated_frame,
        annotated_edge_frame,
        edge_frame,
    ) = get_lines(frame)
    unique_intersections = get_intersections(vertical_lines, horizontal_lines)

    # Visualize the intersections with red dot
    for pt in unique_intersections:
        cv2.circle(annotated_frame, pt, 5, (0, 0, 255), -1)
    cv2.imshow("Intersections", annotated_frame)
    cv2.imshow("Annotated Edge Frame", annotated_edge_frame)

    # Get field points and frame points for calibration
    # Modify the number of points as needed, minimum is 4
    if field_points is None or frame_points is None:
        field_points, frame_points = get_points(frame, 4)
        print("Field points:", field_points)
        print("Frame points:", frame_points)
    else:
        frame_points, status = track_frame_points(
            prev_frame, annotated_edge_frame, frame_points
        )

    # Update frame and points for the next iteration
    prev_frame = annotated_edge_frame.copy()

    # # Get calibration and reprojected points
    rotation_vector, translation_vector, reprojected_points = get_calibration(
        field_points, frame_points, frame_width, frame_height
    )

    # Print the reprojected points for validation
    for original, reprojected in zip(frame_points, reprojected_points):
        print(f"Original: {original}, Reprojected: {reprojected}")

    # Calculate reprojection error
    error = np.sqrt(
        np.sum((frame_points - reprojected_points) ** 2) / len(frame_points)
    )
    print(f"Reprojection Error: {error}")

    # Visualize the reprojected points
    for original, reprojected in zip(frame_points, reprojected_points):
        cv2.circle(annotated_frame, tuple(original.astype(int)), 5, (0, 255, 255), 4)
        cv2.circle(annotated_frame, tuple(reprojected.astype(int)), 5, (255, 120, 0), 4)
    cv2.imshow("Points Validation", annotated_frame)

    # Calculate homography
    H, status = cv2.findHomography(field_points, frame_points, cv2.RANSAC)

    # Get the ad position in the video plane
    ad_position_video = get_ad_position_frame(
        ad_position_field,
        H,
    )
    print("Ad Position in Video Frame:", ad_position_video)

    # Append the current ad position to the buffer
    ad_position_buffer.append(ad_position_video)
    if len(ad_position_buffer) > buffer_size:
        ad_position_buffer.pop(0)  # Remove oldest entry if buffer is full

    # Calculate the average position
    average_ad_position = np.mean(ad_position_buffer, axis=0)

    # Visualize the ad position in the video plane
    for point in average_ad_position:
        cv2.circle(
            annotated_frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1
        )
    cv2.imshow("Frame", annotated_frame)

    # Overlay the ad on the video
    result_frame = overlay_ad_on_frame(frame, rotated_ad_image, average_ad_position)
    cv2.imshow("Frame", result_frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
