from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import cv2
import numpy as np

from config import (
    INPUT_FOLDER,
    INPUT_NAME_VIDEO,
    OUTPUT_FOLDER,
    OUTPUT_NAME_VIDEO,
)

# from Utils.line_detection import line_detection
from line_detection_ani import detect_lines


def create_football_field(pitch_width=68, pitch_length=105, ax_field=None):
    pitch = Pitch(
        pitch_type="metricasports",
        goal_type="line",
        pitch_width=pitch_width,
        pitch_length=pitch_length,
        spot_scale=0.01,
    )
    return pitch.draw(ax=ax_field)


def refine_points_with_subpixel_accuracy(points, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert points to np.float32 for subpixel accuracy
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    refined_points = cv2.cornerSubPix(
        gray_frame,
        points,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
    )
    return refined_points.reshape(-1, 2)


def select_points_on_combined_plot(fig, ax_field, ax_frame, frame, num_points=5):
    points_collected = 0
    selected_points_field = []
    selected_points_frame = []

    # Display the football field on the left subplot
    ax_field.set_title("Football Field")
    create_football_field(ax_field=ax_field)  # Generate the field plot on ax_field

    # Display the video frame on the right subplot
    ax_frame.set_title("Video Frame")
    ax_frame.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Click event handler
    def on_click(event):
        nonlocal points_collected

        if event.inaxes == ax_field and len(selected_points_field) < num_points:
            # Clicks on the football field
            selected_points_field.append((event.xdata, event.ydata))
            ax_field.plot(event.xdata, event.ydata, "o", color="red")
            print(f"Field point selected: ({event.xdata}, {event.ydata})")
            points_collected += 1
        elif event.inaxes == ax_frame and len(selected_points_frame) < num_points:
            # Clicks on the video frame
            selected_points_frame.append((event.xdata, event.ydata))
            ax_frame.plot(event.xdata, event.ydata, "o", color="blue")
            print(f"Frame point selected: ({event.xdata}, {event.ydata})")
            points_collected += 1

        fig.canvas.draw()

        # Close the figure if enough points have been collected
        if points_collected >= 2 * num_points:
            plt.close(fig)

    # Set up plot and event handling
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    return selected_points_field, selected_points_frame


def scale_field_points(field_points, field_size=(1050, 680)):
    field_width, field_height = field_size
    return [(x * field_width, y * field_height) for x, y in field_points]


def get_points(frame, num_points=9):
    # Create football field image and side-by-side subplots
    fig, (ax_field, ax_frame) = plt.subplots(1, 2, figsize=(16, 8))

    # Select points on both the field and video frame
    frame = cv2.resize(frame, (1050, 680))  # Resize for better alignment
    field_points, frame_points = select_points_on_combined_plot(
        fig, ax_field, ax_frame, frame, num_points=num_points
    )

    # Refine frame points to subpixel accuracy
    refined_frame_points = refine_points_with_subpixel_accuracy(frame_points, frame)

    return field_points, refined_frame_points


def apply_homography(frame, field_points, frame_points):
    # Homography matrix computation
    if len(field_points) == len(frame_points) and len(field_points) >= 4:
        field_points_np = np.array(scale_field_points(field_points), dtype=np.float32)
        frame_points_np = np.array(frame_points, dtype=np.float32)
        H, status = cv2.findHomography(frame_points_np, field_points_np, cv2.RANSAC)
        print("Homography Matrix:\n", H)
    else:
        print("Insufficient points or mismatched points for homography.")
        return frame

    # Apply homography transformation to the frame
    warped_frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
    return warped_frame


def apply_homography_with_grid(frame, field_points, frame_points, grid_size=10):
    # Calculate the homography matrix
    if len(field_points) == len(frame_points) and len(field_points) >= 4:
        field_points_np = np.array(scale_field_points(field_points), dtype=np.float32)
        frame_points_np = np.array(frame_points, dtype=np.float32)
        H, status = cv2.findHomography(frame_points_np, field_points_np, cv2.RANSAC)
        print("Homography Matrix:\n", H)
    else:
        print("Insufficient points or mismatched points for homography.")
        return frame

    # Apply homography transformation to the frame
    warped_frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

    # Create a grid and transform it using homography matrix
    for x in range(0, frame.shape[1], grid_size):
        for y in range(0, frame.shape[0], grid_size):
            # Define a point in the source image
            pt = np.array([[x, y]], dtype="float32")
            pt = np.array([pt])
            # Transform point using homography matrix
            transformed_pt = cv2.perspectiveTransform(pt, H)
            tx, ty = transformed_pt[0][0]
            # Draw small circles to mark grid points
            cv2.circle(warped_frame, (int(tx), int(ty)), 2, (0, 255, 0), -1)

    return warped_frame


def main(input_video_file: str, output_video_file: str):
    # Open video file
    cap = cv2.VideoCapture(input_video_file)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    print(
        "Navigate through the video using keys: 'n' for next frame, 'p' for previous frame, and 's' to select the current frame."
    )

    frame = None
    current_frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop to navigate frames
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        if not ret:
            print("No more frames or failed to retrieve frame.")
            break

        # Display the current frame
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Frame {current_frame_index + 1}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Select Frame", display_frame)

        # Wait for key input to control navigation
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):  # Next frame
            current_frame_index += 1
            if current_frame_index >= total_frames:
                current_frame_index = total_frames - 1
        elif key == ord("p"):  # Previous frame
            current_frame_index -= 1
            if current_frame_index < 0:
                current_frame_index = 0
        elif key == ord("s"):  # Select current frame
            print(f"Selected frame {current_frame_index + 1}")
            break
        elif key == ord("q"):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            return

    # Ensure a frame was selected
    if frame is None:
        print("No frame selected.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Process selected frame
    processed_frame, _ = detect_lines(frame)
    cv2.imshow("Line Frame", processed_frame)

    # Select points on the football field and frame
    field_points, frame_points = get_points(processed_frame, num_points=9)

    # Print collected points for verification
    print("Field Points:", field_points)
    print("Frame Points:", frame_points)

    # Calculate homography and transform frame
    transformed_frame = apply_homography(processed_frame, field_points, frame_points)
    cv2.imshow("Transformed Frame", transformed_frame)

    # Wait for user to close windows
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = INPUT_FOLDER + "/" + INPUT_NAME_VIDEO
    output_video = OUTPUT_FOLDER + "/" + OUTPUT_NAME_VIDEO
    main(input_video, output_video)
