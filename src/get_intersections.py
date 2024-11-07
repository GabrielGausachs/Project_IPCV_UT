import math
import cv2
from sklearn.cluster import DBSCAN
import numpy as np


def intersection(o1, p1, o2, p2, tolerance=1e-5):
    o1, p1, o2, p2 = map(np.array, [o1, p1, o2, p2])
    d1 = p1 - o1
    d2 = p2 - o2
    x = o2 - o1
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-8:
        return None

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = o1 + d1 * t1

    # Check if the intersection point is within the bounds of both lines with a tolerance
    if (
        min(o1[0], p1[0]) - tolerance <= r[0] <= max(o1[0], p1[0]) + tolerance
        and min(o1[1], p1[1]) - tolerance <= r[1] <= max(o1[1], p1[1]) + tolerance
        and min(o2[0], p2[0]) - tolerance <= r[0] <= max(o2[0], p2[0]) + tolerance
        and min(o2[1], p2[1]) - tolerance <= r[1] <= max(o2[1], p2[1]) + tolerance
    ):
        return (int(round(r[0])), int(round(r[1])))
    return None


def extend_line(line, extension_length=20):
    """Extend a line segment (x1, y1, x2, y2) in both directions."""
    x1, y1, x2, y2 = line[:4]
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)

    # Normalize direction
    dx /= length
    dy /= length

    # Extend the line
    x1_extended = int(x1 - dx * extension_length)
    y1_extended = int(y1 - dy * extension_length)
    x2_extended = int(x2 + dx * extension_length)
    y2_extended = int(y2 + dy * extension_length)

    return (x1_extended, y1_extended, x2_extended, y2_extended)


def find_intersections(vertical_lines, horizontal_lines, extension_length=20):
    intersections = []

    # Extend lines
    extended_vertical_lines = [
        extend_line(v_line, extension_length) for v_line in vertical_lines
    ]
    extended_horizontal_lines = [
        extend_line(h_line, extension_length) for h_line in horizontal_lines
    ]

    for v_line in extended_vertical_lines:
        for h_line in extended_horizontal_lines:
            x1, y1, x2, y2 = v_line[:4]
            x3, y3, x4, y4 = h_line[:4]

            # Use the intersection function to find the intersection point
            result = intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            if result is not None:
                intersections.append(result)  # Add the intersection point to the set

    return list(intersections)  # Convert the set back to a list before returning


def cluster_intersections(intersections, eps=20, min_samples=2):
    """
    Groups nearby intersection points using DBSCAN clustering.
    Args:
        intersections: List of (x, y) intersection points.
        eps: Maximum distance between points to be considered in the same cluster.
        min_samples: Minimum number of points required to form a cluster.
    Returns:
        List of clustered intersection points as (x, y) tuples.
    """
    if len(intersections) == 0:
        return []  # No intersections to cluster

    # Convert to numpy array for clustering
    points = np.array(intersections)

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Calculate the average location of each cluster to get unique intersection points
    unique_intersections = []
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points if any

        # Find points in this cluster
        cluster_points = points[labels == label]
        # Calculate the centroid of the cluster
        centroid = cluster_points.mean(axis=0)
        unique_intersections.append((int(centroid[0]), int(centroid[1])))

    return unique_intersections


def draw_points(
    frame,
    points,
    color=(0, 255, 0),
    marker_type=cv2.MARKER_CROSS,
    marker_size=35,
    thickness=4,
):
    """
    Draws custom markers at each corner point on the frame.

    Parameters:
    - frame: The image on which to draw the points.
    - points: List of tuples (x, y) for each corner point.
    - color: Color of the marker (default is green).
    - marker_type: Type of marker, e.g., cv2.MARKER_CROSS.
    - marker_size: Size of the marker.
    - thickness: Thickness of the marker lines.
    """
    for x, y in points:
        cv2.drawMarker(
            frame, (int(x), int(y)), color, marker_type, marker_size, thickness
        )
    return frame

def get_unique_intersections(vertical_lines, horizontal_lines):
    intersections = find_intersections(vertical_lines, horizontal_lines)
    unique_intersections = cluster_intersections(intersections)
    return unique_intersections
