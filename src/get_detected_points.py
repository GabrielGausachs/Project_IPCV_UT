import numpy as np
from sklearn.cluster import DBSCAN


def cluster_intersection_points(intersection_points, eps=10, min_samples=1):
    points = np.array([p for p in intersection_points if p is not None])

    if len(points) == 0:
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    centroids = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:
            continue
        cluster_points = points[clustering.labels_ == cluster_id]
        centroid = cluster_points.mean(axis=0)
        centroids.append((int(round(centroid[0])), int(round(centroid[1]))))

    return centroids


# Neighbor check for goal line points
def filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=30):
    filtered_points = []
    for i, pt1 in enumerate(centroids):
        x1, y1 = pt1
        for j, pt2 in enumerate(centroids):
            if i != j:
                x2, y2 = pt2
                if abs(x1 - x2) <= max_x_dist and abs(y1 - y2) <= max_y_dist:
                    filtered_points.append(pt1)
                    break
    return filtered_points


# Order goal line points in the required sequence
def order_goal_line_points(points):
    # Sort points by y-coordinate to separate top and bottom points
    sorted_points = sorted(points, key=lambda p: p[1])

    # Bottom points (lower y value), then sorted by x for left and right
    bottom_points = sorted(sorted_points[:2], key=lambda p: p[0])

    # Top points (higher y value), then sorted by x for left and right
    top_points = sorted(sorted_points[2:], key=lambda p: p[0])

    # Return points in desired order: [Bottom-left, Bottom-right, Top-left, Top-right]
    # return [bottom_points[0], bottom_points[1], top_points[0], top_points[1]]
    return np.array(
        [
            [bottom_points[0][0], bottom_points[0][1]],
            [bottom_points[1][0], bottom_points[1][1]],
            [top_points[0][0], top_points[0][1]],
            [top_points[1][0], top_points[1][1]],
        ],
        dtype="float32",
    )


# Calculate intersection between two line segments
def intersection(o1, p1, o2, p2):
    o1, p1, o2, p2 = map(np.array, [o1, p1, o2, p2])
    d1 = p1 - o1
    d2 = p2 - o2
    x = o2 - o1
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-8:
        return None
    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    r = o1 + d1 * t1
    if (
        min(o1[0], p1[0]) <= r[0] <= max(o1[0], p1[0])
        and min(o1[1], p1[1]) <= r[1] <= max(o1[1], p1[1])
        and min(o2[0], p2[0]) <= r[0] <= max(o2[0], p2[0])
        and min(o2[1], p2[1]) <= r[1] <= max(o2[1], p2[1])
    ):
        return (int(round(r[0])), int(round(r[1])))
    return None


def get_detected_points(horizontal_lines, vertical_lines):
    intersection_points = []
    for h in horizontal_lines:
        for v in vertical_lines:
            inters = intersection(
                (h[0], h[1]), (h[2], h[3]), (v[0], v[1]), (v[2], v[3])
            )
            intersection_points.append(inters)

    # Cluster intersection points and calculate centroids
    centroids = cluster_intersection_points(intersection_points, eps=50, min_samples=1)

    # Filter and order goal line points based on neighbor checking
    goal_line_points = filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=30)
    tracking_points = order_goal_line_points(goal_line_points)
    return tracking_points
