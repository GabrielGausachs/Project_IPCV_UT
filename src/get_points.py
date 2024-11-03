import matplotlib.pyplot as plt
import cv2
import numpy as np

from get_football_field import create_field


def refine_points(points, frame):
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


def select_points(fig, ax_field, ax_frame, num_points):
    points_collected = 0
    selected_points_field = []
    selected_points_frame = []

    # Click event handler
    def on_click(event):
        nonlocal points_collected

        if event.inaxes == ax_field and len(selected_points_field) < num_points:
            # Clicks on the football field
            selected_points_field.append((event.xdata, event.ydata))
            ax_field.plot(event.xdata, event.ydata, "o", color="red")
            points_collected += 1
        elif event.inaxes == ax_frame and len(selected_points_frame) < num_points:
            # Clicks on the video frame
            selected_points_frame.append((event.xdata, event.ydata))
            ax_frame.plot(event.xdata, event.ydata, "o", color="blue")
            points_collected += 1

        fig.canvas.draw()

        # Close the figure if enough points have been collected
        if points_collected >= 2 * num_points:
            plt.close(fig)

    # Set up plot and event handling
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)

    return selected_points_field, selected_points_frame


def get_points(
    frame, num_points, field_size=(105, 68), field_padding=5, field_scale=1.0
):
    # Create football field image and side-by-side subplots in a single figure
    # fig, (ax_field, ax_frame) = plt.subplots(1, 2, figsize=(12, 6))
    # Create a figure with a specific size
    fig = plt.figure(figsize=(16, 8), tight_layout=True)  # Adjust figure size
    # Use gridspec to manage subplot layout and reduce padding
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])  # Two equal-width subplots

    # Create subplots with reduced padding
    ax_field = fig.add_subplot(gs[0])
    ax_frame = fig.add_subplot(gs[1])

    # Set titles for the subplots
    ax_field.set_title("Football Field")
    ax_frame.set_title("Video Frame")

    # Draw the field on the left subplot
    create_field(
        ax_field,
        field_size=field_size,
        field_padding=field_padding,
        field_scale=field_scale,
    )

    # Display the video frame on the right subplot
    ax_frame.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Set aspect ratio to equal for accurate representation of the field
    ax_field.set_aspect(1)

    # Call the point selection function (you will need to define this)
    field_points, frame_points = select_points(fig, ax_field, ax_frame, num_points)

    # Refine frame points to subpixel accuracy (you will need to define this)
    refined_frame_points = refine_points(frame_points, frame)
    refined_field_points = np.array(field_points).astype(np.float32)

    plt.show()  # Ensure the plot is displayed

    return refined_field_points, refined_frame_points
