import cv2
from matplotlib import pyplot as plt
import numpy as np

from get_lines import get_lines
from get_intersections import get_intersections
from get_points import get_points
from get_calibration import get_calibration
from get_ad_position import get_ad_position_frame, get_ad_position_field
from get_ad_overlay_frame import overlay_ad_on_frame

# Load the video and read the first frame for line detection
# video_path = "football_videos/video_example1.mp4"
video_path = "football_videos/video5.mp4"
cap = cv2.VideoCapture(video_path)

# Get width and height
frame_size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
field_size = (105, 68)

ad_image = cv2.imread("banners/jumbo_banner.png", cv2.IMREAD_COLOR)

# Your original dimensions
frame_width, frame_height = frame_size
field_width, field_height = field_size

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()

# Get lines and intersections
vertical_lines, horizontal_lines, annotated_frame = get_lines(frame)
unique_intersections = get_intersections(vertical_lines, horizontal_lines)

# Visualize the intersections
for pt in unique_intersections:
    cv2.circle(annotated_frame, pt, 5, (0, 0, 255), -1)
cv2.imshow("Intersections", annotated_frame)


# Get field points and frame points for calibration
# Modify the number of points as needed, minimum is 4
field_points, frame_points = get_points(frame, 4)
print("Field points:", field_points)
print("Frame points:", frame_points)

# Get calibration and reprojected points
rotation_vector, translation_vector, reprojected_points = get_calibration(
    field_points, frame_points, frame_width, frame_height
)

# Print the reprojected points for validation
for original, reprojected in zip(frame_points, reprojected_points):
    print(f"Original: {original}, Reprojected: {reprojected}")

# Calculate reprojection error
error = np.sqrt(np.sum((frame_points - reprojected_points) ** 2) / len(frame_points))
print(f"Reprojection Error: {error}")

# Visualize the reprojected points
for original, reprojected in zip(frame_points, reprojected_points):
    cv2.circle(annotated_frame, tuple(original.astype(int)), 5, (0, 255, 255), 4)
    cv2.circle(annotated_frame, tuple(reprojected.astype(int)), 5, (255, 120, 0), 4)
cv2.imshow("Points Validation", annotated_frame)

# Calculate homography from video plane to field plane
H, status = cv2.findHomography(field_points, reprojected_points, cv2.RANSAC)

# Get the ad position in the field plane
ad_position_field, rotated_ad_image = get_ad_position_field(
    ad_image, field_padding=4, h_padding=0.5
)
print("Ad Position in Field:", ad_position_field)

# Get the ad position in the video plane
ad_position_video = get_ad_position_frame(
    ad_position_field,
    H,
)
print("Ad Position in Video Frame:", ad_position_video)

# Visualize the ad position in the video plane
for point in ad_position_video:
    cv2.circle(annotated_frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)
cv2.imshow("Projected Ad Position", annotated_frame)

# Overlay the ad on the video
result_frame = overlay_ad_on_frame(frame, rotated_ad_image, ad_position_video)
cv2.imshow("Ad Overlay", result_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
