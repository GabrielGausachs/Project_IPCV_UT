from matplotlib import patches


def create_field(
    ax,
    field_size=(105, 68),
    field_padding=5,
    field_scale=1.0,
):
    """
    Create a football pitch plot with various elements.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the plot on.
    field_size : tuple, optional
        The size of the football field. Default is (105, 68).
    field_padding : float, optional
        Padding around the field boundary in meters. Default is 5.
    field_scale : float, optional
        Scale factor for the plot. Default is 1.0.

    Returns
    -------
    None
    """
    field_length, field_width = field_size
    field_padding = field_padding * field_scale
    length = field_length * field_scale
    width = field_width * field_scale
    goal_length = 7.3 * field_scale
    penalty_area_length = 40.3 * field_scale
    penalty_area_width = 16.5 * field_scale
    goal_area_extension = 5.5 * field_scale
    goal_area_length = goal_area_extension + goal_length + goal_area_extension
    centre_circle_radius = 9.15 * field_scale
    penalty_mark_length = 11 * field_scale
    penalty_arc_radius = 9.15 * field_scale

    # Draw field outline
    ax.plot(
        [
            -field_padding,
            -field_padding,
            length + field_padding,
            length + field_padding,
            -field_padding,
        ],
        [
            -field_padding,
            width + field_padding,
            width + field_padding,
            -field_padding,
            -field_padding,
        ],
        color="grey",
    )

    # field Outline & Centre Line
    ax.plot([0, 0, length, length, 0], [0, width, width, 0, 0], color="black")

    # Centre Circle
    centre_circle = patches.Circle(
        (length / 2, width / 2),
        centre_circle_radius,
        color="black",
        fill=False,
    )
    ax.add_patch(centre_circle)

    # Plot centre
    ax.plot(length / 2, width / 2, marker="o", markersize=6, color="black")

    # Halfway Line
    ax.plot([length / 2, length / 2], [0, width], color="black")

    # Goal Lines
    half_goal_length = goal_length / 2
    ax.plot(
        [0, 0],
        [width / 2 - half_goal_length, width / 2 + half_goal_length],
        color="red",
        linewidth=2,
    )  # Left goal line
    ax.plot(
        [length, length],
        [width / 2 - half_goal_length, width / 2 + half_goal_length],
        color="red",
        linewidth=2,
    )  # Right goal line

    # Penalty Areas
    half_penalty_area_length = penalty_area_length / 2
    ax.plot(
        [0, penalty_area_width, penalty_area_width, 0],
        [
            width / 2 - half_penalty_area_length,
            width / 2 - half_penalty_area_length,
            width / 2 + half_penalty_area_length,
            width / 2 + half_penalty_area_length,
        ],
        color="green",
    )
    ax.plot(
        [length, length - penalty_area_width, length - penalty_area_width, length],
        [
            width / 2 - half_penalty_area_length,
            width / 2 - half_penalty_area_length,
            width / 2 + half_penalty_area_length,
            width / 2 + half_penalty_area_length,
        ],
        color="green",
    )

    # Goal Areas
    half_goal_area_length = goal_area_length / 2
    ax.plot(
        [0, goal_area_extension, goal_area_extension, 0],
        [
            width / 2 - half_goal_area_length,
            width / 2 - half_goal_area_length,
            width / 2 + half_goal_area_length,
            width / 2 + half_goal_area_length,
        ],
        color="blue",
    )
    ax.plot(
        [length, length - goal_area_extension, length - goal_area_extension, length],
        [
            width / 2 - half_goal_area_length,
            width / 2 - half_goal_area_length,
            width / 2 + half_goal_area_length,
            width / 2 + half_goal_area_length,
        ],
        color="blue",
    )

    # Penalty Marks and Arcs
    ax.plot(
        penalty_mark_length,
        width / 2,
        marker="o",
        markersize=6,
        color="black",
        fillstyle="none",
    )
    ax.plot(
        length - penalty_mark_length,
        width / 2,
        marker="o",
        markersize=6,
        color="black",
        fillstyle="none",
    )

    left_penalty_arc = patches.Arc(
        (penalty_mark_length, width / 2),
        penalty_arc_radius * 2,
        penalty_arc_radius * 2,
        theta1=308,
        theta2=52,
        color="black",
    )
    ax.add_patch(left_penalty_arc)

    right_penalty_arc = patches.Arc(
        (length - penalty_mark_length, width / 2),
        penalty_arc_radius * 2,
        penalty_arc_radius * 2,
        theta1=128,
        theta2=232,
        color="black",
    )
    ax.add_patch(right_penalty_arc)

    # Corner Circles
    corner_radius = 0.9
    corners = [
        (0, 0, 0),  # Bottom-left corner (0 degrees)
        (0, width, 270),  # Top-left corner (90 degrees)
        (length, width, 180),  # Top-right corner (180 degrees)
        (length, 0, 90),  # Bottom-right corner (270 degrees)
    ]
    for x, y, angle in corners:
        corner_circle = patches.Arc(
            (x, y),
            corner_radius * 2,
            corner_radius * 2,
            angle=angle,
            theta1=0,
            theta2=90,
            color="black",
        )
        ax.add_patch(corner_circle)

    # Set limits and aspect ratio
    ax.set_xlim(-field_padding, length + field_padding)
    ax.set_ylim(-field_padding, width + field_padding)
    ax.set_aspect("equal")
    ax.axis("off")
