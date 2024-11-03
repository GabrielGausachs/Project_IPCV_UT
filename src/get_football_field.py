from matplotlib import patches


def create_field(
    ax,
    field_size=(105, 68),
    padding=5,
    scale=1.0,
):
    field_length, field_width = field_size
    padding = padding * scale
    length = field_length * scale
    width = field_width * scale
    goal_length = 7.3 * scale
    penalty_area_length = 40.3 * scale
    penalty_area_width = 16.5 * scale
    goal_area_extension = 5.5 * scale
    goal_area_length = goal_area_extension + goal_length + goal_area_extension
    centre_circle_radius = 9.15 * scale
    penalty_mark_length = 11 * scale
    penalty_arc_radius = 9.15 * scale

    # Draw field outline
    ax.plot(
        [-padding, -padding, length + padding, length + padding, -padding],
        [-padding, width + padding, width + padding, -padding, -padding],
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
    ax.set_xlim(-padding, length + padding)
    ax.set_ylim(-padding, width + padding)
    ax.set_aspect("equal")
    ax.axis("off")
