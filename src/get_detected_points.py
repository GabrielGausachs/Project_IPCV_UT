import numpy as np
from sklearn.cluster import DBSCAN


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


def get_detected_points(centroids):
    # Filter and order goal line points based on neighbor checking
    goal_line_points = filter_goal_line_points(centroids, max_x_dist=300, max_y_dist=30)
    tracking_points = order_goal_line_points(goal_line_points)
    return tracking_points
