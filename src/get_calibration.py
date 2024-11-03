import cv2
import numpy as np


def get_calibration(field_points, frame_points, frame_width, frame_height):

    field_points_3d = np.hstack((field_points, np.zeros((field_points.shape[0], 1))))

    # Convert field points to a NumPy array (if not already)
    field_points_3d = np.array(field_points_3d, dtype=np.float32)  # Shape: (N, 3)
    frame_points = np.array(frame_points, dtype=np.float32)  # Shape: (N, 2)

    # Assuming frame_width and frame_height are defined
    fx = 6400  # Focal length in x (as a float)
    fy = 6400  # Focal length in y (as a float)
    cx = frame_width / 2.0  # Principal point x (image center)
    cy = frame_height / 2.0  # Principal point y (image center)

    # Camera intrinsic matrix
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Solve PnP to find camera pose
    success, rotation_vector, translation_vector = cv2.solvePnP(
        field_points_3d,  # 3D points in the world
        frame_points,  # 2D image points
        intrinsic_matrix,  # Camera intrinsic matrix
        None,  # No distortion coefficients
    )

    reprojected_points = None

    if success:
        print("Rotation Vector:\n", rotation_vector)
        print("Translation Vector:\n", translation_vector)
        # Project the 3D points back onto the 2D image plane
        reprojected_points, _ = cv2.projectPoints(
            field_points_3d,  # 3D points
            rotation_vector,  # Rotation vector
            translation_vector,  # Translation vector
            intrinsic_matrix,  # Camera intrinsic matrix
            None,  # No distortion coefficients
        )

        # Convert to the same format for comparison
        reprojected_points = reprojected_points.reshape(-1, 2)
    else:
        print("Camera pose estimation failed.")

    return rotation_vector, translation_vector, reprojected_points
