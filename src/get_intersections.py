from sklearn.cluster import DBSCAN
import numpy as np


def find_intersections(vertical_lines, horizontal_lines):
    intersections = []

    for v_line in vertical_lines:
        for h_line in horizontal_lines:
            x1, y1, x2, y2 = v_line[:4]
            x3, y3, x4, y4 = h_line[:4]

            # Calculate denominator for the intersection formula
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue  # Lines are parallel and do not intersect

            # Calculate intersection point (px, py)
            px = (
                (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
            ) / denom
            py = (
                (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
            ) / denom

            # Check if the intersection point is within the bounds of both lines
            if (
                min(x1, x2) <= px <= max(x1, x2)
                and min(y1, y2) <= py <= max(y1, y2)
                and min(x3, x4) <= px <= max(x3, x4)
                and min(y3, y4) <= py <= max(y3, y4)
            ):
                intersections.append(
                    (int(px), int(py))
                )  # Append the intersection point as an integer tuple

    return intersections


def cluster_intersections(intersections, eps=100, min_samples=2):
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


def get_intersections(vertical_lines, horizontal_lines):
    intersections = find_intersections(vertical_lines, horizontal_lines)
    unique_intersections = cluster_intersections(intersections)
    return unique_intersections
