from typing import Literal
import cv2
from matplotlib import pyplot as plt
import numpy as np

from get_detected_points import get_detected_points
from get_lines import detect_lines, get_hough_lines
from get_intersections import get_intersections
from get_points import get_points
from get_calibration import get_calibration
from get_ad_position import get_ad_position_frame, get_ad_position_field
from get_ad_overlay_frame import overlay_ad_on_frame
from get_tracked_points import track_frame_points


def main(
    input_video_path: str,
    output_video_path: str,
    ad_image_path: str,
    ad_side: Literal["left", "right"],
    automatic: bool = False,
    num_points: int = 4,
    field_size: tuple = (105, 68),
    field_padding: float = 5,
    field_scale: int = 1.0,
    h_padding: float = 0.2,
    v_padding: float = 10,
    banner_scale: float = 0.5,
):
    """
    Main function to overlay advertisement on a video.
    These parameters only works for manual points selection: *num_points*, *field_padding*, *field_scale*, *h_padding*, *v_padding*, *banner_scale*


    Parameters
    ----------
    input_video_path : str
        Path to the input video file.
    output_video_path : str
        Path to the output video file.
    ad_image_path : str
        Path to the advertisement image file.
    ad_side : Literal["left", "right"]
        Side of the field to place the advertisement.
    automatic : bool, optional
        Whether to automatically detect the points for homography or not.
    num_points: int, optional
        Number of points to use for homography estimation. Default is 4.
    field_padding : float, optional
        Padding around the field boundary in meters. Default is 5.
        This depends on field's touchline area.
    field_scale : int, optional
        Scale factor for the field image. Default is 1.
    field_size : tuple, optional
        Size of the field in meters. Default is (105, 68).
    h_padding : float, optional
        Padding between the ad image and the goal line.
        Note: Higher values will shift the ad banner towards the spectator area.
    v_padding : float, optional
        Padding between the ad image and the top side line.
        Note: Higher values will shift the ad banner towards the goal post.
    banner_scale : float, optional
        Scale factor for the advertisement image.
    """
    cv2.namedWindow("Overlay Frame", cv2.WINDOW_NORMAL)

    # Load video and get video properties
    cap = cv2.VideoCapture(input_video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Load ad image
    ad_image = cv2.imread(ad_image_path, cv2.IMREAD_COLOR)

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Get the ad position in the field plane
    ad_position_field, rotated_ad_image = get_ad_position_field(
        ad_image,
        field_scale=field_scale,
        field_size=field_size,
        field_padding=field_padding,
        ad_side=ad_side,
        h_padding=h_padding,
        v_padding=v_padding,
        banner_scale=banner_scale,
    )
    print("Ad Position in Field:", ad_position_field)

    # Your original dimensions
    frame_width, frame_height = frame_size
    field_width, field_height = field_size

    field_points = None
    frame_points = None
    ad_position_buffer = []  # Initialize a buffer to store ad positions
    buffer_size = 10  # Adjust buffer size as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Copy the original frame
        original_frame = frame.copy()

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the edges
        edge_frame = detect_lines(frame)

        # Get the hough lines
        v_lines, h_lines, annotated_frame, annotated_edge_frame = get_hough_lines(
            edge_frame, frame
        )

        # Get the unique intersections, i.e. detect the corners
        field_corners = get_intersections(v_lines, h_lines)

        # Visualize the intersections with red dot
        for pt in field_corners:
            cv2.circle(annotated_frame, pt, 5, (0, 0, 255), -1)
            cv2.circle(annotated_edge_frame, pt, 5, (0, 0, 255), -1)

        cv2.imshow("Corners Detected Frame", annotated_frame)

        # Get field points and frame points for homography
        if field_points is None or frame_points is None:
            if automatic:
                # Get the automatically detected points for homography
                # Here, Field points are thegoal area points, and will be our 2D plane
                field_points = np.array(
                    [[0, 0], [7.32, 0], [0, 5.5], [7.32, 5.5]], dtype="float32"
                )

                # Frame points are the goal area points detected automatically from the frame
                frame_points = get_detected_points(h_lines, v_lines)

                # Here, ad position are selected in our 2D plane w.r.t goal area points.
                ad_position_field = np.array(
                    [
                        [8.0, -2.0],
                        [11.0, -2.0],
                        [8.0, -0.25],
                        [11.0, -0.25],
                    ],
                    dtype="float32",
                )

            else:
                # Get the manually selected field points and frame points
                # We have to select one point from field and then corresponding point from frame
                # Minimum points required to select for homography is 4
                # To select more points, change the value of num_points
                field_points, frame_points = get_points(
                    annotated_frame,
                    num_points=num_points,
                    field_size=field_size,
                    field_padding=field_padding,
                    field_scale=field_scale,
                )

            print("Field points:", field_points)
            print("Frame points:", frame_points)
        else:
            frame_points, status = track_frame_points(
                prev_frame, gray_frame, frame_points
            )

        # Update frame and points for the next iteration
        prev_frame = gray_frame.copy()

        # OPTIONAL- Perform calibration
        # # Get calibration and reprojected points
        # rotation_vector, translation_vector, reprojected_points = get_calibration(
        #     field_points, frame_points, frame_width, frame_height
        # )

        # # Print the reprojected points for validation
        # for original, reprojected in zip(frame_points, reprojected_points):
        #     print(f"Original: {original}, Reprojected: {reprojected}")

        # # Calculate reprojection error
        # error = np.sqrt(
        #     np.sum((frame_points - reprojected_points) ** 2) / len(frame_points)
        # )
        # print(f"Reprojection Error: {error}")

        # # Visualize the reprojected points
        # for original, reprojected in zip(frame_points, reprojected_points):
        #     cv2.circle(annotated_frame, tuple(original.astype(int)), 5, (0, 255, 255), 4)
        #     cv2.circle(annotated_frame, tuple(reprojected.astype(int)), 5, (255, 120, 0), 4)
        # cv2.imshow("Points Validation", annotated_frame)

        # print(field_points.shape)
        # print(frame_points.shape)
        ## OPTIONAL - Till here ##

        # Calculate homography
        # Use reprojected points if you want to use calibrated points
        H, status = cv2.findHomography(field_points, frame_points, cv2.RANSAC)

        # Get the ad position in the video plane
        ad_position_video = get_ad_position_frame(
            ad_position_field,
            H,
        )
        print("Ad Position in Video Frame:", ad_position_video)

        # Calculate the average position in the video plane
        # Append the current ad position to the buffer
        ad_position_buffer.append(ad_position_video)
        if len(ad_position_buffer) > buffer_size:
            ad_position_buffer.pop(0)  # Remove oldest entry if buffer is full

        ## OPTIONAL - Use buffer to average ad position ##
        # average_ad_position = np.mean(ad_position_buffer, axis=0)
        ## OPTIONAL - Till here ##

        # Visualize the ad position in the video plane
        for point in ad_position_video:
            cv2.circle(
                annotated_frame, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1
            )
        cv2.imshow("Ad Position Frame", annotated_frame)

        # Overlay the ad on the video
        overlay_frame = overlay_ad_on_frame(
            original_frame, rotated_ad_image, ad_position_video
        )

        cv2.imshow("Overlay Frame", overlay_frame)
        out.write(overlay_frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ad_path = "banners/jumbo_banner.png"

    # video_path = "football_videos/video_example1.mp4"

    # video_path = "football_videos/video5.mp4"
    # output_path = "football_videos/output/output_video5.mp4"

    video_path = "football_videos/soccer_video_example1.mp4"
    output_path = "football_videos/output/output_soccer_video_example1.mp4"

    main(
        video_path,
        output_path,
        ad_path,
        ad_side="left",
        automatic=False,
    )
