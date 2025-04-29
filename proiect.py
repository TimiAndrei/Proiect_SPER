import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import random
import matplotlib.animation as animation


def create_map(size, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    map_grid = np.zeros((size, size))
    num_obstacles = random.randint(3, max(3, size // 3))
    max_obs_size = max(2, size // 5)
    for _ in range(num_obstacles):
        obs_size = random.randint(2, max_obs_size)
        start_row = random.randint(0, size - obs_size)
        start_col = random.randint(0, size - obs_size)
        map_grid[start_row : start_row + obs_size, start_col : start_col + obs_size] = 1
    return map_grid


def calc_connectivity(slice):
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
    adjacency_matrix = np.zeros([len(parts_left), len(parts_right)])
    for l, lparts in enumerate(parts_left):
        for r, rparts in enumerate(parts_right):
            if min(lparts[1], rparts[1]) - max(lparts[0], rparts[0]) > 0:
                adjacency_matrix[l, r] = 1
    return adjacency_matrix


def bcd(map_grid):
    erode_img = 1 - map_grid
    last_connectivity = 0
    last_connectivity_parts = []
    current_cell = 1
    current_cells = []
    separate_img = np.copy(erode_img)
    cell_boundaries = {}
    non_neighboor_cells = []

    for col in range(erode_img.shape[1]):
        current_slice = erode_img[:, col]
        connectivity, connective_parts = calc_connectivity(current_slice)

        if last_connectivity == 0:
            current_cells = [current_cell + i for i in range(connectivity)]
            current_cell += connectivity
        elif connectivity == 0:
            current_cells = []
            continue
        else:
            adj_matrix = get_adjacency_matrix(last_connectivity_parts, connective_parts)
            new_cells = [0] * len(connective_parts)
            for i in range(adj_matrix.shape[0]):
                if np.sum(adj_matrix[i, :]) == 1:
                    new_cells[int(np.argwhere(adj_matrix[i, :])[0])] = current_cells[i]
                elif np.sum(adj_matrix[i, :]) > 1:
                    for idx in np.argwhere(adj_matrix[i, :]):
                        new_cells[int(idx)] = current_cell
                        current_cell += 1
            for i in range(adj_matrix.shape[1]):
                if np.sum(adj_matrix[:, i]) > 1 or np.sum(adj_matrix[:, i]) == 0:
                    new_cells[i] = current_cell
                    current_cell += 1
            current_cells = new_cells

        for cell, slice_part in zip(current_cells, connective_parts):
            separate_img[slice_part[0] : slice_part[1], col] = cell
            cell_boundaries.setdefault(cell, []).append(slice_part)

        if len(current_cells) > 1:
            non_neighboor_cells.append(current_cells)

        last_connectivity = connectivity
        last_connectivity_parts = connective_parts

    x_coordinates = calculate_x_coordinates(
        separate_img.shape[1],
        separate_img.shape[0],
        range(1, current_cell),
        cell_boundaries,
        non_neighboor_cells,
    )

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
    prioritizing revisiting unvisited regions as soon as possible

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
        if current_pos[0] is None:
            return None

        min_dist = float("inf")
        closest_point = None

        # Scan the entire map for unvisited cells
        for y in range(height):
            for x in range(width):
                if not visited[y, x]:
                    dist = manhattan_distance((y, x), current_pos)
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

        # Try moving vertically first (only if coordinates are valid)
        if y1 is not None and y2 is not None:
            y_dir = 1 if y2 > y1 else -1
            y_start = y1 + y_dir
            y_end = y2

            # Ensure we move in the right direction
            if y_dir > 0:
                y_range = range(y_start, y_end + y_dir, y_dir)
            else:
                y_range = range(y_start, y_end + y_dir, y_dir)

            for y in y_range:
                if 0 <= y < height and not visited[y, x1]:
                    safe_path.append((x1, y))

        # Then move horizontally (only if coordinates are valid)
        if x1 is not None and x2 is not None:
            x_dir = 1 if x2 > x1 else -1
            x_start = x1 + x_dir
            x_end = x2

            # Ensure we move in the right direction
            if x_dir > 0:
                x_range = range(x_start, x_end + x_dir, x_dir)
            else:
                x_range = range(x_start, x_end + x_dir, x_dir)

            for x in x_range:
                if 0 <= x < width and not visited[y2, x]:
                    safe_path.append((x, y2))

        return safe_path

    # The recursive navigation function for up-down column movement
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


def animate_traversal(map_grid, separate_img, num_cells, path):
    """
    Animate the traversal of the map using a simple robot visualization

    Args:
        map_grid: Original map with obstacles
        separate_img: Image with cell decomposition
        num_cells: Number of cells
        path: List of (x, y) coordinates to traverse
    """
    # Create figure with two views - original map and cellular decomposition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Display original map
    ax1.imshow(map_grid, cmap="Greys", origin="lower")
    ax1.set_title("Original Map")

    # Display cell decomposition with colors
    cmap = cm.get_cmap("tab20", num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)
    masked_img = np.ma.masked_where(separate_img == 0, separate_img)
    ax2.imshow(masked_img, cmap=cmap, norm=norm, origin="lower")

    # Add black for obstacles
    obstacle_img = np.zeros((*separate_img.shape, 4))
    obstacle_img[separate_img == 0, 3] = 1
    ax2.imshow(obstacle_img, origin="lower")
    ax2.set_title("Cellular Decomposition")

    # Robot visualization - dot with trail
    (robot1,) = ax1.plot([], [], "ro-", markersize=8, linewidth=2)
    (robot2,) = ax2.plot([], [], "ro-", markersize=8, linewidth=2)

    # Animation controls
    animation_speed = [100]  # Initial speed (milliseconds)

    def init():
        robot1.set_data([], [])
        robot2.set_data([], [])
        return robot1, robot2

    def animate(i):
        if i < len(path):
            # Extract points up to current frame
            points = path[: i + 1]
            x, y = zip(*points)

            # Update robot positions
            robot1.set_data(x, y)
            robot2.set_data(x, y)
        return robot1, robot2

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(path) + 1,
        interval=animation_speed[0],
        blit=True,
    )

    # Function to update animation speed
    def update_speed():
        ani.event_source.stop()
        ani.event_source.interval = animation_speed[0]
        ani.event_source.start()

    # Keyboard event handler
    def on_key(event):
        if event.key == "enter":  # Generate new map
            plt.close(fig)
            return True
        elif event.key == "escape":  # Exit
            plt.close(fig)
            return False
        elif event.key == "left":  # Reduce speed
            animation_speed[0] = min(animation_speed[0] + 50, 1000)
            print(f"Speed: {animation_speed[0]}ms (slower)")
            update_speed()
        elif event.key == "right":  # Increase speed
            animation_speed[0] = max(animation_speed[0] - 50, 10)
            print(f"Speed: {animation_speed[0]}ms (faster)")
            update_speed()
        return None

    # Connect keyboard handler
    result = [None]  # To store the result

    def handle_close(evt):
        result[0] = False if result[0] is None else result[0]

    def on_key_wrapper(event):
        key_result = on_key(event)
        if key_result is not None:
            result[0] = key_result

    fig.canvas.mpl_connect("key_press_event", on_key_wrapper)
    fig.canvas.mpl_connect("close_event", handle_close)

    plt.tight_layout()
    plt.show()

    return result[0]


def main():
    """
    Main function to run the application
    """
    size = 15  # Default map size
    traverse_mode = True  # Always run in traverse mode

    seed = 100  # Initial seed
    regenerate = True

    while regenerate:
        # Create map and calculate cellular decomposition
        map_grid = create_map(size, seed)
        separate_img, num_cells, cell_boundaries, x_coordinates, non_neighboor_cells = (
            bcd(map_grid)
        )

        # Calculate and animate traversal path
        path = calculate_zigzag_path(separate_img, cell_boundaries, x_coordinates)
        regenerate = animate_traversal(map_grid, separate_img, num_cells, path)

        if regenerate:
            seed += 1  # Increment seed for a different map


if __name__ == "__main__":
    main()
