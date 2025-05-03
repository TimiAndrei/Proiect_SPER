import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import random
import matplotlib.animation as animation
import sys
import time
from collections import deque
import argparse


def create_map(size, seed=None, num_obstacles=None):
    """
    Create a map with randomly placed obstacles

    Args:
        size: Size of the square map
        seed: Random seed for reproducibility
        num_obstacles: Number of obstacles to place (if None, uses a random value)

    Returns:
        map_grid: 2D numpy array with obstacles (1) and free space (0)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    map_grid = np.zeros((size, size))

    # Determine number of obstacles
    if num_obstacles is None:
        num_obstacles = random.randint(3, max(3, size // 3))
    else:
        num_obstacles = min(num_obstacles, size * size //
                            4)  # Limit to 25% of map area

    max_obs_size = max(2, size // 5)
    for _ in range(num_obstacles):
        obs_size = random.randint(2, max_obs_size)
        start_row = random.randint(0, size - obs_size)
        start_col = random.randint(0, size - obs_size)
        map_grid[start_row: start_row + obs_size,
                 start_col: start_col + obs_size] = 1
    return map_grid


def calc_connectivity(slice):
    """Calculate the connectivity of free space in a slice"""
    connectivity = 0
    last_data = 0
    open_part = False
    connective_parts = []
    start_point = 0
    for i, data in enumerate(slice):
        if last_data == 0 and data == 1:
            open_part = True
            start_point = i
        elif last_data == 1 and data == 0 and open_part:
            open_part = False
            connectivity += 1
            connective_parts.append((start_point, i))
        last_data = data
    if open_part:
        connectivity += 1
        connective_parts.append((start_point, len(slice)))
    return connectivity, connective_parts


def get_adjacency_matrix(parts_left, parts_right):
    """Get the adjacency matrix between parts"""
    adjacency_matrix = np.zeros([len(parts_left), len(parts_right)])
    for l, lparts in enumerate(parts_left):
        for r, rparts in enumerate(parts_right):
            if min(lparts[1], rparts[1]) - max(lparts[0], rparts[0]) > 0:
                adjacency_matrix[l, r] = 1
    return adjacency_matrix


def animate_decomposition_cells(map_grid, separate_img, num_cells):
    """
    Animate the Boustrophedon decomposition process cell by cell.
    Args:
        map_grid: Original map with obstacles
        separate_img: Final cell decomposition image
        num_cells: Number of cells
    Returns:
        True if Enter was pressed to continue, False if window closed
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.colormaps["tab20"].resampled(num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)

    # Prepare cell-by-cell reveal images
    cell_labels = [cell for cell in np.unique(separate_img) if cell != 0]
    reveal_imgs = []
    revealed = np.zeros_like(separate_img)
    for cell in cell_labels:
        revealed = revealed.copy()
        revealed[separate_img == cell] = cell
        reveal_imgs.append(revealed.copy())

    # Animation state
    finished = [False]
    proceed = [False]

    def init():
        ax.clear()
        ax.set_title("Decomposition Progress (Cell by Cell)")
        return []

    def animate(i):
        ax.clear()
        masked_img = np.ma.masked_where(reveal_imgs[i] == 0, reveal_imgs[i])
        ax.imshow(masked_img, cmap=cmap, norm=norm, origin="lower")
        obstacle_img = np.zeros((*map_grid.shape, 4))
        obstacle_img[map_grid == 1, 3] = 1
        ax.imshow(obstacle_img, origin="lower")
        ax.set_title(f"Decomposition Progress: Cell {i+1}/{len(reveal_imgs)}")
        if i == len(reveal_imgs) - 1:
            finished[0] = True
        return []

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(reveal_imgs),
        interval=300,
        blit=True,
        repeat=False
    )

    # Instructions
    instruction_text = fig.text(
        0.5,
        0.01,
        "Press Enter to continue to trajectory animation, Esc to exit",
        ha="center",
        color="black",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    def on_key(event):
        if finished[0]:
            if event.key == "enter":
                proceed[0] = True
                plt.close(fig)
            elif event.key == "escape":
                proceed[0] = False
                plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()
    return proceed[0]


def bcd(map_grid, return_progress=False):
    """
    Implement Boustrophedon Cellular Decomposition
    Args:
        map_grid: 2D array with obstacles (1) and free space (0)
        return_progress: If True, also return a list of intermediate images for animation
    Returns:
        separate_img: Image with cell decomposition
        current_cell: Number of cells
        cell_boundaries: Dictionary of cell boundaries
        x_coordinates: Dictionary of x-coordinates
        non_neighboor_cells: List of non-neighbor cells
        [separate_img_progress]: (optional) List of intermediate images
    """
    erode_img = 1 - map_grid
    last_connectivity = 0
    last_connectivity_parts = []
    current_cell = 1
    current_cells = []
    separate_img = np.copy(erode_img)
    cell_boundaries = {}
    non_neighboor_cells = []
    separate_img_progress = [] if return_progress else None

    for col in range(erode_img.shape[1]):
        current_slice = erode_img[:, col]
        connectivity, connective_parts = calc_connectivity(current_slice)

        if last_connectivity == 0:
            current_cells = [current_cell + i for i in range(connectivity)]
            current_cell += connectivity
        elif connectivity == 0:
            current_cells = []
            if return_progress:
                separate_img_progress.append(np.copy(separate_img))
            continue
        else:
            adj_matrix = get_adjacency_matrix(
                last_connectivity_parts, connective_parts)
            new_cells = [0] * len(connective_parts)
            for i in range(adj_matrix.shape[0]):
                if np.sum(adj_matrix[i, :]) == 1:
                    idx = np.argwhere(adj_matrix[i, :])[0][0]
                    new_cells[int(idx)] = current_cells[i]
                elif np.sum(adj_matrix[i, :]) > 1:
                    for idx in np.argwhere(adj_matrix[i, :]):
                        new_cells[int(idx[0])] = current_cell
                        current_cell += 1
            for i in range(adj_matrix.shape[1]):
                if np.sum(adj_matrix[:, i]) > 1 or np.sum(adj_matrix[:, i]) == 0:
                    new_cells[i] = current_cell
                    current_cell += 1
            current_cells = new_cells

        for cell, slice_part in zip(current_cells, connective_parts):
            separate_img[slice_part[0]: slice_part[1], col] = cell
            cell_boundaries.setdefault(cell, []).append(slice_part)

        if len(current_cells) > 1:
            non_neighboor_cells.append(current_cells)

        last_connectivity = connectivity
        last_connectivity_parts = connective_parts
        if return_progress:
            separate_img_progress.append(np.copy(separate_img))

    x_coordinates = calculate_x_coordinates(
        separate_img.shape[1],
        separate_img.shape[0],
        range(1, current_cell),
        cell_boundaries,
        non_neighboor_cells,
    )

    if return_progress:
        return (
            separate_img,
            current_cell - 1,
            cell_boundaries,
            x_coordinates,
            non_neighboor_cells,
            separate_img_progress,
        )
    else:
        return (
            separate_img,
            current_cell - 1,
            cell_boundaries,
            x_coordinates,
            non_neighboor_cells,
        )


def calculate_x_coordinates(
    x_size, y_size, cells_to_visit, cell_boundaries, nonneighbors
):
    """Calculate x-coordinates for cells"""
    cells_x_coordinates = {}
    width_accum_prev = 0
    cell_idx = 1
    total_cell_number = len(cell_boundaries)

    while cell_idx <= total_cell_number:
        is_split_by_obstacle = False
        for subneighbor in nonneighbors:
            if subneighbor and subneighbor[0] == cell_idx:
                separated_cell_number = len(subneighbor)
                if cell_idx in cell_boundaries:
                    width_current_cell = len(cell_boundaries[cell_idx])
                    for j in range(separated_cell_number):
                        if cell_idx + j in cell_boundaries:
                            cells_x_coordinates[cell_idx + j] = list(
                                range(
                                    width_accum_prev,
                                    width_accum_prev + width_current_cell,
                                )
                            )
                    width_accum_prev += width_current_cell
                    cell_idx += separated_cell_number
                    is_split_by_obstacle = True
                    break
        if not is_split_by_obstacle and cell_idx in cell_boundaries:
            width_current_cell = len(cell_boundaries[cell_idx])
            cells_x_coordinates[cell_idx] = list(
                range(width_accum_prev, width_accum_prev + width_current_cell)
            )
            width_accum_prev += width_current_cell
            cell_idx += 1
        elif not is_split_by_obstacle:
            cell_idx += 1

    return cells_x_coordinates


def calculate_zigzag_path(separate_img, cell_boundaries, x_coordinates):
    """
    Calculate a traversal path using a recursive boustrophedon approach
    with vertical (up-down) movement from column to column,
    prioritizing revisiting unvisited regions as soon as possible.

    Args:
        separate_img: The image with cell decomposition
        cell_boundaries: Dictionary containing cell boundary information
        x_coordinates: Dictionary containing x coordinates for each cell

    Returns:
        path: List of (x, y) coordinates for the traversal path
    """
    height, width = separate_img.shape
    path = []  # Store the traversal path
    visited = np.zeros_like(separate_img, dtype=bool)

    # Mark obstacles as visited to avoid them
    visited[separate_img == 0] = True

    # Current position tracker
    current_pos = [None, None]

    # Calculate Manhattan distance between two points
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Find the closest unvisited cell to the current position
    def find_closest_unvisited():
        min_dist = float("inf")
        closest_point = None

        # Scan the entire map for unvisited cells
        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    if current_pos[0] is not None:
                        dist = manhattan_distance((y, x), current_pos)
                    else:
                        dist = 0  # Start at the first unvisited cell
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = (y, x)

        return closest_point

    # Create a safe path from current position to target position
    def create_path_to(target):
        if current_pos[0] is None or target is None:
            return []

        y1, x1 = current_pos
        y2, x2 = target
        safe_path = []

        # Move vertically first
        y_dir = 1 if y2 > y1 else -1
        for y in range(y1, y2 + y_dir, y_dir):
            if 0 <= y < height and not visited[y, x1]:
                safe_path.append((x1, y))

        # Then move horizontally
        x_dir = 1 if x2 > x1 else -1
        for x in range(x1, x2 + x_dir, x_dir):
            if 0 <= x < width and not visited[y2, x]:
                safe_path.append((x, y2))

        return safe_path

    # Optimized navigation function for zigzag traversal
    def navigate(y, x, direction):
        # Check if position is valid
        if not (0 <= y < height and 0 <= x < width):
            return False

        # Skip if obstacle or already visited
        if visited[y, x]:
            return False

        # Mark current position as visited and add to path
        visited[y, x] = True
        path.append((x, y))
        current_pos[0], current_pos[1] = y, x

        # Try to move vertically in current direction first
        next_y = y + direction
        if 0 <= next_y < height and not visited[next_y, x]:
            if navigate(next_y, x, direction):
                return True

        # Then try to move horizontally right and change direction
        next_x = x + 1
        if next_x < width and not visited[y, next_x]:
            if navigate(y, next_x, -direction):
                return True

        # If can't move right, try moving left and change direction
        next_x = x - 1
        if next_x >= 0 and not visited[y, next_x]:
            if navigate(y, next_x, -direction):
                return True

        return False

    # Start from top-left and continue until all accessible cells are visited
    remaining = np.sum(~visited)

    while remaining > 0:
        # If we're just starting or need to find a new point
        if current_pos[0] is None:
            # Find the first unvisited point
            start_found = False
            for y in range(height):
                for x in range(width):
                    if not visited[y, x]:
                        current_pos = [y, x]
                        start_found = True
                        break
                if start_found:
                    break

            # If no unvisited points, we're done
            if not start_found:
                break

            # Start navigation from this point
            navigate(current_pos[0], current_pos[1], 1)
        else:
            # Find the closest unvisited cell
            target = find_closest_unvisited()
            if target is None:
                break

            # Create path to the target
            target_path = create_path_to(target)

            # Add path points and mark them as visited
            for x, y in target_path:
                if not visited[y, x]:
                    visited[y, x] = True
                    path.append((x, y))
                    current_pos = [y, x]

            # Start navigation from the new point
            navigate(current_pos[0], current_pos[1], 1)

        # Recalculate remaining unvisited cells
        remaining = np.sum(~visited)

    return path


def animate_traversal(map_grid, separate_img, num_cells, path, mode='classic'):
    """
    Animate the traversal of the map using a simple robot visualization

    Args:
        map_grid: Original map with obstacles
        separate_img: Image with cell decomposition
        num_cells: Number of cells
        path: List of (x, y) coordinates to traverse
        mode: 'classic' or 'bfs' to determine the title
    """
    # Create figure with single view - cellular decomposition
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display cell decomposition with colors
    cmap = plt.colormaps["tab20"].resampled(num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)
    masked_img = np.ma.masked_where(separate_img == 0, separate_img)
    ax.imshow(masked_img, cmap=cmap, norm=norm, origin="lower")

    # Add black for obstacles
    obstacle_img = np.zeros((*separate_img.shape, 4))
    obstacle_img[separate_img == 0, 3] = 1
    ax.imshow(obstacle_img, origin="lower")

    # Set title based on mode
    if mode == 'classic':
        ax.set_title("Classic BCD Traversal")
    else:
        ax.set_title("BCD Traversal with BFS")

    # Create path segments that don't cross obstacles
    height, width = map_grid.shape
    path_segments = []
    current_segment = []

    # Helper function to check if a line segment crosses any obstacles
    def line_crosses_obstacle(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        # For very close points, just check if either endpoint is an obstacle
        if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1:
            return False

        # Number of interpolation steps (more for longer lines)
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 2

        # Check points along the line
        for step in range(steps + 1):
            t = step / steps  # Interpolation parameter [0,1]
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= x < width and 0 <= y < height:
                if map_grid[y, x] == 1:  # Obstacle
                    return True

        return False

    # Process the path into non-obstacle-crossing segments
    for i, point in enumerate(path):
        if not current_segment:
            current_segment.append(point)
        else:
            if i > 0 and line_crosses_obstacle(path[i - 1], point):
                # If this segment would cross an obstacle, end the current segment
                if len(current_segment) > 0:
                    path_segments.append(current_segment)
                current_segment = [point]
            else:
                current_segment.append(point)

    # Add the last segment if not empty
    if current_segment:
        path_segments.append(current_segment)

    # Robot visualization - position marker and path segments
    robot = ax.plot([], [], "ro", markersize=8)[0]

    # Create line objects for path segments
    path_lines = [ax.plot([], [], "r-", linewidth=2)[0] for _ in path_segments]

    # Animation controls
    animation_speed = [100]  # Initial speed (milliseconds)
    current_frame = [0]  # Current frame counter
    is_complete = [False]  # Completion state
    frame_skip = [1]  # Number of frames to skip for faster animation
    # Only use frame skip for large maps
    use_frame_skip = map_grid.shape[0] > 50

    def init():
        robot.set_data([], [])
        for line in path_lines:
            line.set_data([], [])
        return [robot] + path_lines

    def animate(i):
        current_frame[0] = i * frame_skip[0] if use_frame_skip else i
        if current_frame[0] >= len(path):
            current_frame[0] = len(path) - 1

        # Update robot position
        x, y = path[current_frame[0]]
        robot.set_data([x], [y])

        # Update path segments - only show segments we've traversed
        current_point_index = current_frame[0]
        active_points = set(path[: current_point_index + 1])

        for seg_idx, segment in enumerate(path_segments):
            visible_points = [p for p in segment if p in active_points]
            if visible_points:
                x_vals, y_vals = (
                    zip(*visible_points) if len(visible_points) > 1 else ([], [])
                )
                path_lines[seg_idx].set_data(x_vals, y_vals)
            else:
                path_lines[seg_idx].set_data([], [])

        # Check if animation is complete
        if current_frame[0] == len(path) - 1:
            is_complete[0] = True
            if mode == 'classic':
                ax.set_title("Classic BCD Traversal (Complete)")
            else:
                ax.set_title("BCD Traversal with BFS (Complete)")

        return [robot] + path_lines

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(path) // frame_skip[0] if use_frame_skip else len(path),
        interval=animation_speed[0],
        blit=True,
        repeat=False
    )

    # Function to update animation speed
    def update_speed():
        try:
            ani.event_source.stop()
            ani.event_source.interval = animation_speed[0]
            ani.event_source.start()
        except AttributeError:
            pass

    # Status text for instructions
    instruction_text = fig.text(
        0.5,
        0.01,
        "Controls: R (restart), ← (slow), → (max speed), Enter (new map), O/P (fewer/more obstacles), Esc (exit)",
        ha="center",
        color="black",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Keyboard event handler
    def on_key(event):
        if event.key == "enter":  # Generate new map
            plt.close(fig)
            return True, None  # Keep obstacle count unchanged
        elif event.key == "escape":  # Exit
            plt.close(fig)
            return False, None
        elif event.key == "left":  # Slow down
            if use_frame_skip and frame_skip[0] > 1:
                frame_skip[0] = max(1, frame_skip[0] - 1)
            else:
                animation_speed[0] = min(animation_speed[0] + 50, 1000)
            update_speed()
        elif event.key == "right":  # Speed up
            if use_frame_skip:
                if animation_speed[0] > 1:
                    animation_speed[0] = max(1, animation_speed[0] - 50)
                else:
                    # Skip up to 10 frames
                    frame_skip[0] = min(10, frame_skip[0] + 1)
            else:
                animation_speed[0] = max(1, animation_speed[0] - 50)
            update_speed()
        elif event.key == "r":  # Restart animation
            if is_complete[0]:
                # Reset completion state and title
                is_complete[0] = False
                if mode == 'classic':
                    ax.set_title("Classic BCD Traversal")
                else:
                    ax.set_title("BCD Traversal with BFS")
                # Create a new animation with the same speed
                nonlocal ani
                ani = animation.FuncAnimation(
                    fig,
                    animate,
                    init_func=init,
                    frames=len(
                        path) // frame_skip[0] if use_frame_skip else len(path),
                    interval=animation_speed[0],
                    blit=True,
                    repeat=False
                )
                # Force the animation to start with the current speed
                ani.event_source.interval = animation_speed[0]
                plt.draw()
        elif event.key == "o":  # Fewer obstacles
            plt.close(fig)
            return True, -1  # Signal to decrease obstacles
        elif event.key == "p":  # More obstacles
            plt.close(fig)
            return True, 1  # Signal to increase obstacles
        return None

    # Connect keyboard handler
    result = [None, None]  # [regenerate, obstacle_change]

    def handle_close(evt):
        if result[0] is None:
            result[0] = False

    def on_key_wrapper(event):
        key_result = on_key(event)
        if key_result is not None:
            result[0], result[1] = key_result

    fig.canvas.mpl_connect("key_press_event", on_key_wrapper)
    fig.canvas.mpl_connect("close_event", handle_close)

    plt.tight_layout()
    plt.show()

    return result


def bfs_path(start, goal, map_grid):
    """
    Find the shortest path from start to goal using BFS, avoiding obstacles.
    Args:
        start: (x, y) tuple
        goal: (x, y) tuple
        map_grid: 2D numpy array, 1=obstacle, 0=free
    Returns:
        List of (x, y) tuples representing the path (including start and goal)
        or [] if no path found
    """
    width, height = map_grid.shape[1], map_grid.shape[0]
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        current = queue.popleft()
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                if map_grid[ny, nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))
    return []


def calculate_zigzag_path_bfs(separate_img, cell_boundaries, x_coordinates, map_grid):
    """
    BCD traversal: traverse cells in BCD order, zigzag in each cell, BFS between cells for continuity.
    Args:
        separate_img: The image with cell decomposition
        cell_boundaries: Dictionary containing cell boundary information
        x_coordinates: Dictionary containing x coordinates for each cell
        map_grid: The original map with obstacles (for BFS)
    Returns:
        path: List of (x, y) coordinates for the traversal path
    """
    path = []
    prev_end = None
    map_height, map_width = map_grid.shape if map_grid is not None else (0, 0)

    for cell in sorted(cell_boundaries.keys()):
        x_cols = x_coordinates[cell]
        y_ranges = cell_boundaries[cell]
        cell_path = []
        for i, (col, (row_start, row_end)) in enumerate(zip(x_cols, y_ranges)):
            # Ensure column index is within bounds
            if col >= map_width:
                continue

            rng = range(row_start, row_end)
            if i % 2 == 0:
                for row in rng:
                    if map_grid is None or (row < map_height and map_grid[row, col] == 0):
                        cell_path.append((col, row))
            else:
                for row in reversed(rng):
                    if map_grid is None or (row < map_height and map_grid[row, col] == 0):
                        cell_path.append((col, row))
        if not cell_path:
            continue  # skip empty cells
        cell_start = cell_path[0]
        if prev_end is not None and map_grid is not None:
            bfs = bfs_path(prev_end, cell_start, map_grid)
            if bfs and len(bfs) > 1:
                if bfs[-1] == cell_start:
                    path.extend(bfs[1:])
                else:
                    path.extend(bfs[1:])
                    path.append(cell_start)
            else:
                path.append(cell_start)
        else:
            path.append(cell_start)
        path.extend(cell_path[1:])
        prev_end = cell_path[-1]
    return path


def animate_traversal_side_by_side(map_grid, separate_img, num_cells, path_classic, path_bfs):
    """
    Animate the traversal of the map using a simple robot visualization
    side-by-side comparison between classic and BCD traversal

    Args:
        map_grid: Original map with obstacles
        separate_img: Image with cell decomposition
        num_cells: Number of cells
        path_classic: List of (x, y) coordinates for classic traversal
        path_bfs: List of (x, y) coordinates for BCD traversal
    """
    # Create figure with two views - cellular decomposition
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Display cell decomposition with colors
    cmap = plt.colormaps["tab20"].resampled(num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)
    masked_img = np.ma.masked_where(separate_img == 0, separate_img)
    axs[0].imshow(masked_img, cmap=cmap, norm=norm, origin="lower")
    axs[1].imshow(masked_img, cmap=cmap, norm=norm, origin="lower")

    # Add black for obstacles
    obstacle_img = np.zeros((*separate_img.shape, 4))
    obstacle_img[separate_img == 0, 3] = 1
    axs[0].imshow(obstacle_img, origin="lower")
    axs[1].imshow(obstacle_img, origin="lower")
    axs[0].set_title("BCD Traversal")
    axs[1].set_title("BCD Traversal with BFS")

    # Create path segments that don't cross obstacles
    height, width = map_grid.shape
    path_segments_classic = []
    path_segments_bfs = []
    current_segment_classic = []
    current_segment_bfs = []

    # Helper function to check if a line segment crosses any obstacles
    def line_crosses_obstacle(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        # For very close points, just check if either endpoint is an obstacle
        if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1:
            return False

        # Number of interpolation steps (more for longer lines)
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 2

        # Check points along the line
        for step in range(steps + 1):
            t = step / steps  # Interpolation parameter [0,1]
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= x < width and 0 <= y < height:
                if map_grid[y, x] == 1:  # Obstacle
                    return True

        return False

    # Process the paths into non-obstacle-crossing segments
    for i, point in enumerate(path_classic):
        if not current_segment_classic:
            current_segment_classic.append(point)
        else:
            if i > 0 and line_crosses_obstacle(path_classic[i - 1], point):
                # If this segment would cross an obstacle, end the current segment
                if len(current_segment_classic) > 0:
                    path_segments_classic.append(current_segment_classic)
                current_segment_classic = [point]
            else:
                current_segment_classic.append(point)

    # Add the last segment if not empty
    if current_segment_classic:
        path_segments_classic.append(current_segment_classic)

    for i, point in enumerate(path_bfs):
        if not current_segment_bfs:
            current_segment_bfs.append(point)
        else:
            if i > 0 and line_crosses_obstacle(path_bfs[i - 1], point):
                # If this segment would cross an obstacle, end the current segment
                if len(current_segment_bfs) > 0:
                    path_segments_bfs.append(current_segment_bfs)
                current_segment_bfs = [point]
            else:
                current_segment_bfs.append(point)

    # Add the last segment if not empty
    if current_segment_bfs:
        path_segments_bfs.append(current_segment_bfs)

    # Robot visualization - position markers and path segments
    robot_classic = axs[0].plot([], [], "ro", markersize=8)[0]
    robot_bfs = axs[1].plot([], [], "ro", markersize=8)[0]

    # Create line objects for path segments
    path_lines_classic = [axs[0].plot(
        [], [], "r-", linewidth=2)[0] for _ in path_segments_classic]
    path_lines_bfs = [axs[1].plot([], [], "r-", linewidth=2)[0]
                      for _ in path_segments_bfs]

    # Animation controls
    animation_speed = [100]  # Initial speed (milliseconds)
    current_frame = [0]  # Current frame counter
    is_complete = [False]  # Completion state
    frame_skip = [1]  # Number of frames to skip for faster animation
    # Only use frame skip for large maps
    use_frame_skip = map_grid.shape[0] > 50

    def init():
        robot_classic.set_data([], [])
        robot_bfs.set_data([], [])
        for line in path_lines_classic:
            line.set_data([], [])
        for line in path_lines_bfs:
            line.set_data([], [])
        return [robot_classic] + path_lines_classic + [robot_bfs] + path_lines_bfs

    def animate(i):
        current_frame[0] = i * frame_skip[0] if use_frame_skip else i
        idx1 = min(current_frame[0], len(path_classic) - 1)
        idx2 = min(current_frame[0], len(path_bfs) - 1)
        x_classic, y_classic = path_classic[idx1]
        x_bfs, y_bfs = path_bfs[idx2]
        robot_classic.set_data([x_classic], [y_classic])
        robot_bfs.set_data([x_bfs], [y_bfs])

        # Update path segments - only show segments we've traversed
        current_point_index_classic = idx1
        current_point_index_bfs = idx2
        active_points_classic = set(
            path_classic[: current_point_index_classic + 1])
        active_points_bfs = set(path_bfs[: current_point_index_bfs + 1])

        for seg_idx, segment in enumerate(path_segments_classic):
            visible_points_classic = [
                p for p in segment if p in active_points_classic]
            if visible_points_classic:
                x_vals_classic, y_vals_classic = (
                    zip(*visible_points_classic) if len(visible_points_classic) > 1 else ([], [])
                )
                path_lines_classic[seg_idx].set_data(
                    x_vals_classic, y_vals_classic)
            else:
                path_lines_classic[seg_idx].set_data([], [])

        for seg_idx, segment in enumerate(path_segments_bfs):
            visible_points_bfs = [p for p in segment if p in active_points_bfs]
            if visible_points_bfs:
                x_vals_bfs, y_vals_bfs = (
                    zip(*visible_points_bfs) if len(visible_points_bfs) > 1 else ([], [])
                )
                path_lines_bfs[seg_idx].set_data(x_vals_bfs, y_vals_bfs)
            else:
                path_lines_bfs[seg_idx].set_data([], [])

        # Check if animation is complete
        if idx1 == len(path_classic) - 1 and idx2 == len(path_bfs) - 1:
            is_complete[0] = True
            axs[0].set_title("BCD Traversal (Complete)")
            axs[1].set_title("BCD Traversal with BFS (Complete)")

        return [robot_classic] + path_lines_classic + [robot_bfs] + path_lines_bfs

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=max(len(path_classic), len(
            path_bfs)) // frame_skip[0] if use_frame_skip else max(len(path_classic), len(path_bfs)),
        interval=animation_speed[0],
        blit=True,
        repeat=False
    )

    # Function to update animation speed
    def update_speed():
        try:
            ani.event_source.stop()
            ani.event_source.interval = animation_speed[0]
            ani.event_source.start()
        except AttributeError:
            pass

    # Status text for instructions
    instruction_text = fig.text(
        0.5,
        0.01,
        "Controls: R (restart), ← (slow), → (max speed), Enter (new map), O/P (fewer/more obstacles), Esc (exit)",
        ha="center",
        color="black",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7),
    )

    # Keyboard event handler
    def on_key(event):
        if event.key == "enter":  # Generate new map
            plt.close(fig)
            return True, None  # Keep obstacle count unchanged
        elif event.key == "escape":  # Exit
            plt.close(fig)
            return False, None
        elif event.key == "left":  # Slow down
            if use_frame_skip and frame_skip[0] > 1:
                frame_skip[0] = max(1, frame_skip[0] - 1)
            else:
                animation_speed[0] = min(animation_speed[0] + 50, 1000)
            update_speed()
        elif event.key == "right":  # Speed up
            if use_frame_skip:
                if animation_speed[0] > 1:
                    animation_speed[0] = max(1, animation_speed[0] - 50)
                else:
                    # Skip up to 10 frames
                    frame_skip[0] = min(10, frame_skip[0] + 1)
            else:
                animation_speed[0] = max(1, animation_speed[0] - 50)
            update_speed()
        elif event.key == "r":  # Restart animation
            if is_complete[0]:
                # Reset completion state and titles
                is_complete[0] = False
                axs[0].set_title("BCD Traversal")
                axs[1].set_title("BCD Traversal with BFS")
                # Create a new animation with the same speed
                nonlocal ani
                ani = animation.FuncAnimation(
                    fig,
                    animate,
                    init_func=init,
                    frames=max(len(path_classic), len(
                        path_bfs)) // frame_skip[0] if use_frame_skip else max(len(path_classic), len(path_bfs)),
                    interval=animation_speed[0],
                    blit=True,
                    repeat=False
                )
                # Force the animation to start with the current speed
                ani.event_source.interval = animation_speed[0]
                plt.draw()
        elif event.key == "o":  # Fewer obstacles
            plt.close(fig)
            return True, -1  # Signal to decrease obstacles
        elif event.key == "p":  # More obstacles
            plt.close(fig)
            return True, 1  # Signal to increase obstacles
        return None

    # Connect keyboard handler
    result = [None, None]  # [regenerate, obstacle_change]

    def handle_close(evt):
        if result[0] is None:
            result[0] = False

    def on_key_wrapper(event):
        key_result = on_key(event)
        if key_result is not None:
            result[0], result[1] = key_result

    fig.canvas.mpl_connect("key_press_event", on_key_wrapper)
    fig.canvas.mpl_connect("close_event", handle_close)

    plt.tight_layout()
    plt.show()

    return result


def main():
    """
    Main function to run the application

    Usage:
        python proiect.py [map_size] [--mode classic|bfs|both]

    Args:
        map_size (int, optional): Size of the square map. Defaults to 15 if not provided.
        --mode: 'classic' (default), 'bfs', or 'both' for side-by-side comparison.
            classic - Classic BCD (no BFS between cells, just zigzag and jump)
            bfs     - BCD with BFS transitions (zigzag in each cell, BFS between cells)
            both    - Show both traversals side by side for comparison
    """
    # Increase recursion limit for larger maps
    sys.setrecursionlimit(10000)

    # Get map size from command line argument or use default
    parser = argparse.ArgumentParser()
    parser.add_argument('map_size', nargs='?', type=int, default=15)
    parser.add_argument(
        '--mode', choices=['classic', 'bfs', 'both'], default='classic')
    args = parser.parse_args()
    size = args.map_size
    mode = args.mode
    if size < 5:
        print("Map size must be at least 5x5. Using default size of 15.")
        size = 15
    seed = 90  # Initial seed
    num_obstacles = 5  # Initial number of obstacles

    regenerate = True

    while regenerate:
        # Create map and calculate cellular decomposition
        map_grid = create_map(size, seed, num_obstacles)
        bcd_result = bcd(map_grid, return_progress=False)
        separate_img, num_cells, cell_boundaries, x_coordinates, non_neighboor_cells = bcd_result

        # Animate decomposition cell by cell, wait for Enter
        proceed = animate_decomposition_cells(
            map_grid, separate_img, num_cells)
        if not proceed:
            break

        # Calculate and animate traversal path
        if mode == 'classic':
            path = calculate_zigzag_path(
                separate_img, cell_boundaries, x_coordinates)
            result = animate_traversal(
                map_grid, separate_img, num_cells, path, mode='classic')
        elif mode == 'bfs':
            path = calculate_zigzag_path_bfs(
                separate_img, cell_boundaries, x_coordinates, map_grid=map_grid)
            result = animate_traversal(
                map_grid, separate_img, num_cells, path, mode='bfs')
        elif mode == 'both':
            path_classic = calculate_zigzag_path(
                separate_img, cell_boundaries, x_coordinates)
            path_bfs = calculate_zigzag_path_bfs(
                separate_img, cell_boundaries, x_coordinates, map_grid=map_grid)
            result = animate_traversal_side_by_side(
                map_grid, separate_img, num_cells, path_classic, path_bfs)
        regenerate, obstacle_change = result

        if regenerate:
            seed += 1  # Increment seed for a different map

            # Adjust number of obstacles if requested
            if obstacle_change is not None:
                num_obstacles = max(1, num_obstacles + obstacle_change)
                print(f"Number of obstacles: {num_obstacles}")


if __name__ == "__main__":
    main()

"""
=========================================
BOUSTROPHEDON CELLULAR DECOMPOSITION
=========================================

A robot path planning algorithm that decomposes the environment into cells 
and creates a traversal path to cover all accessible areas.

USAGE:
------
    python proiect.py [map_size] [--mode classic|bfs|both]

    map_size: Optional integer argument specifying the size of the square map (default: 15)
    --mode:   Optional argument to select the traversal mode:
              classic - Classic BCD (no BFS between cells, just zigzag and jump)
              bfs     - BCD with BFS transitions (zigzag in each cell, BFS between cells)
              both    - Show both traversals side by side for comparison

    Examples:
        python proiect.py 20 --mode classic
        python proiect.py 20 --mode bfs
        python proiect.py 20 --mode both

KEYBOARD CONTROLS:
------------------
    LEFT ARROW:  Slow down animation
    RIGHT ARROW: Speed up animation (to max)
    O:           Fewer obstacles in the next map
    P:           More obstacles in the next map
    ENTER:       Generate a new map with current settings
    ESC:         Exit the application

VIEWS:
------
    --mode classic: Shows classic BCD traversal
    --mode bfs:     Shows BCD traversal with BFS transitions
    --mode both:    Shows both traversals side by side

ALGORITHM BEHAVIOR:
-------------------
    - The algorithm decomposes the map into cells using BCD
    - The robot traverses each cell in a zigzag pattern
    - The robot prioritizes visiting the closest unvisited cells
    - The path visualization avoids cutting through obstacles (in BFS mode)

ABOUT THE ALGORITHM:
--------------------
Boustrophedon Cellular Decomposition is a method for complete coverage path planning.
It works by dividing the environment into cells at critical points where obstacles
change the connectivity of the free space. Each cell is then covered using a back-and-forth
motion (the term "boustrophedon" refers to this pattern, similar to an ox plowing a field).
"""
