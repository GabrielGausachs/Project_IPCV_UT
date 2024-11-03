import cv2
import numpy as np


def track_frame_points(prev_frame, current_frame, current_frame_points):
    """
    Track points from the current frame to the next frame using the
    Lucas-Kanade optical flow algorithm.

    Parameters
    ----------
    prev_frame : numpy.ndarray
        The previous frame in the video.
    current_frame : numpy.ndarray
        The current frame in the video.
    current_frame_points : numpy.ndarray
        The points from the current frame to track.

    Returns
    -------
    new_frame_points : numpy.ndarray
        The tracked points in the next frame.
    status : numpy.ndarray
        A boolean array indicating the success of tracking for each point.
    """

    # Convert frames to grayscale for optical flow tracking
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if len(current_frame.shape) == 3:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Reshape old_frame_points if needed
    old_frame_points = current_frame_points
    if current_frame_points.shape[-1] == 2:
        old_frame_points = current_frame_points.reshape(-1, 1, 2)

    # Define parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Track points
    new_frame_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_frame, current_frame, old_frame_points, None, **lk_params
    )

    # Filter and reshape results to (N, 2) format
    new_frame_points = new_frame_points[status == 1].reshape(-1, 2)
    status = status.reshape(-1)

    # If less than 4 points are tracked, use the old frame points
    if len(new_frame_points) < 4:
        new_frame_points = current_frame_points

    return new_frame_points, status
