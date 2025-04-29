import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import sys
import random
import matplotlib.animation as animation
import time

# Creare harta cu obstacole generate aleator


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


# Funcție pentru calculul conectivității unei felii
def calc_connectivity(slice):
    """
    Calculează conectivitatea unei felii și returnează zonele conectate
    """
    connectivity = 0
    last_data = 0
    open_part = False
    connective_parts = []
    start_point = 0  # Inițializăm pentru a evita erori

    for i, data in enumerate(slice):
        if last_data == 0 and data == 1:  # Începe o zonă liberă
            open_part = True
            start_point = i
        elif last_data == 1 and data == 0 and open_part:  # Se termină o zonă liberă
            open_part = False
            connectivity += 1
            end_point = i
            connective_parts.append((start_point, end_point))

        last_data = data

    # Verifică dacă ultima zonă este deschisă
    if open_part:
        connectivity += 1
        connective_parts.append((start_point, len(slice)))

    return connectivity, connective_parts


# Funcție pentru obținerea matricei de adiacență între două felii
def get_adjacency_matrix(parts_left, parts_right):
    """
    Obține matricea de adiacență între două felii vecine
    """
    adjacency_matrix = np.zeros([len(parts_left), len(parts_right)])
    for l, lparts in enumerate(parts_left):
        for r, rparts in enumerate(parts_right):
            if min(lparts[1], rparts[1]) - max(lparts[0], rparts[0]) > 0:
                adjacency_matrix[l, r] = 1

    return adjacency_matrix


# Algoritm Boustrophedon Cellular Decomposition
def bcd(map_grid):
    """
    Implementează algoritmul de descompunere celulară Boustrophedon

    Args:
        map_grid: Harta cu 0 pentru spațiu liber și 1 pentru obstacole

    Returns:
        separate_img: Harta cu celulele marcate diferit
        cells: Numărul total de celule
        cell_boundaries: Conține informații despre limitele celulelor
    """
    # Inversăm valorile: 0 pentru obstacole, 1 pentru spațiu liber
    erode_img = 1 - map_grid

    assert len(erode_img.shape) == 2, "Harta trebuie să aibă un singur canal."
    last_connectivity = 0
    last_connectivity_parts = []
    current_cell = 1
    current_cells = []
    separate_img = np.copy(erode_img)
    cell_boundaries = {}  # Stochează limitele y pentru fiecare celulă
    non_neighboor_cells = []  # Stochează celulele separate de obstacole

    for col in range(erode_img.shape[1]):
        current_slice = erode_img[:, col]
        connectivity, connective_parts = calc_connectivity(current_slice)

        if last_connectivity == 0:
            current_cells = []
            for i in range(connectivity):
                current_cells.append(current_cell)
                current_cell += 1
        elif connectivity == 0:
            current_cells = []
            continue
        else:
            adj_matrix = get_adjacency_matrix(last_connectivity_parts, connective_parts)
            new_cells = [0] * len(connective_parts)

            for i in range(adj_matrix.shape[0]):
                if np.sum(adj_matrix[i, :]) == 1:
                    new_cells[np.argwhere(adj_matrix[i, :])[0][0]] = current_cells[i]
                # IN a avut loc: o parte anterioară conectată la mai multe părți curente
                elif np.sum(adj_matrix[i, :]) > 1:
                    for idx in np.argwhere(adj_matrix[i, :]):
                        new_cells[idx[0]] = current_cell
                        current_cell = current_cell + 1

            for i in range(adj_matrix.shape[1]):
                # OUT a avut loc: o parte curentă conectată la mai multe părți anterioare
                if np.sum(adj_matrix[:, i]) > 1:
                    new_cells[i] = current_cell
                    current_cell = current_cell + 1
                # Partea curentă nu comunică cu nicio parte anterioară
                elif np.sum(adj_matrix[:, i]) == 0:
                    new_cells[i] = current_cell
                    current_cell = current_cell + 1
            current_cells = new_cells

        # Desenăm informațiile de partiționare pe hartă
        for cell, slice_part in zip(current_cells, connective_parts):
            separate_img[slice_part[0] : slice_part[1], col] = cell

            # Salvăm limitele celulelor pentru traversare
            cell_boundaries.setdefault(cell, [])
            cell_boundaries[cell].append(slice_part)

        # Verificăm celule separate de obstacole în slice-ul curent
        if len(current_cells) > 1:
            non_neighboor_cells.append(current_cells)

        last_connectivity = connectivity
        last_connectivity_parts = connective_parts

    # Calculăm coordonatele x pentru fiecare celulă
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


# Calculează coordonatele x pentru fiecare celulă
def calculate_x_coordinates(
    x_size, y_size, cells_to_visit, cell_boundaries, nonneighbors
):
    """
    Calculează coordonatele x pentru fiecare celulă în funcție de structura hărții
    """
    cells_to_visit = list(cell_boundaries.keys())
    total_cell_number = len(cells_to_visit)

    # Calculăm coordonatele x pentru fiecare celulă
    cells_x_coordinates = {}
    width_accum_prev = 0
    cell_idx = 1

    while cell_idx <= total_cell_number:
        is_split_by_obstacle = False

        # Verificăm dacă celula este separată de obstacole
        for subneighbor in nonneighbors:
            if subneighbor and subneighbor[0] == cell_idx:
                # Celula curentă este împărțită de un obstacol
                separated_cell_number = len(subneighbor)
                # Folosim lungimea primei celule divizate ca referință
                if cell_idx in cell_boundaries:
                    width_current_cell = len(cell_boundaries[cell_idx])

                    # Toate celulele separate de obstacol au aceleași coordonate x
                    for j in range(separated_cell_number):
                        if cell_idx + j in cell_boundaries:
                            cells_x_coordinates[cell_idx + j] = list(
                                range(
                                    width_accum_prev,
                                    width_accum_prev + width_current_cell,
                                )
                            )

                    width_accum_prev += width_current_cell
                    cell_idx = cell_idx + separated_cell_number
                    is_split_by_obstacle = True
                    break

        if not is_split_by_obstacle and cell_idx in cell_boundaries:
            # Celula nu este separată de niciun obstacol
            width_current_cell = len(cell_boundaries[cell_idx])
            cells_x_coordinates[cell_idx] = list(
                range(width_accum_prev, width_accum_prev + width_current_cell)
            )
            width_accum_prev += width_current_cell
            cell_idx = cell_idx + 1
        elif not is_split_by_obstacle:
            # Celula nu există în boundaries (poate s-a omis în algoritmul BCD)
            cell_idx += 1

    return cells_x_coordinates


# Display the original map and cell decomposition side by side
def display_maps(map_grid, separate_img, num_cells):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Display original map with obstacles in black
    # Create colored image for the original map
    color_img = np.ones((*map_grid.shape, 3), dtype=np.uint8) * 255  # White background
    # Set obstacles to black
    color_img[map_grid == 1] = [0, 0, 0]  # Black obstacles

    # Display original map
    ax1.imshow(color_img, origin="lower")
    ax1.set_title("Original Map")

    # Display cell decomposition with colors
    cmap = cm.get_cmap("tab20", num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)

    # Set obstacle cells to black in the decomposition map
    display_img = separate_img.copy()

    # Create a masked array to display obstacles as black
    masked_img = np.ma.masked_where(display_img == 0, display_img)

    # Create the image with masked data and colormap
    ax2.imshow(masked_img, cmap=cmap, norm=norm, origin="lower")

    # Add black for obstacles (where mask is True)
    obstacle_mask = display_img == 0
    obstacle_img = np.zeros((*display_img.shape, 4))  # RGBA image for obstacles
    obstacle_img[obstacle_mask, 3] = 1  # Set alpha to 1 for obstacles
    ax2.imshow(obstacle_img, origin="lower")

    ax2.set_title("Boustrophedon Cellular Decomposition")

    plt.tight_layout()

    # Add keyboard handler for regenerating map
    def on_key(event):
        if event.key == "enter":  # Press Enter to generate a new map
            plt.close(fig)
            return True
        elif event.key == "escape":  # Press Escape to exit
            plt.close(fig)
            return False
        return None

    # Connect the key event handler
    result = [None]  # Using a list as a mutable container for the result

    def handle_close(evt):
        result[0] = False if result[0] is None else result[0]

    def on_key_wrapper(event):
        key_result = on_key(event)
        if key_result is not None:
            result[0] = key_result

    fig.canvas.mpl_connect("key_press_event", on_key_wrapper)
    fig.canvas.mpl_connect("close_event", handle_close)

    plt.show()
    return result[0]


# Function to calculate a zig-zag traversal path
def calculate_zigzag_path(separate_img, cell_boundaries, x_coordinates):
    """
    Calculate a recursive traversal path using a boustrophedon pattern

    Args:
        separate_img: The image with cell decomposition
        cell_boundaries: Dictionary containing cell boundary information
        x_coordinates: Dictionary containing x coordinates for each cell

    Returns:
        path: List of (x, y) coordinates for the traversal path
    """
    # Get image dimensions
    height, width = separate_img.shape

    # Initialize path
    path = []

    # Create a visited map to track where we've been
    visited = np.zeros_like(separate_img, dtype=bool)

    # Mark obstacles as visited
    visited[separate_img == 0] = True

    # Recursive function to navigate the map
    def navigate(y, x, direction):
        # Check if position is valid
        if not (0 <= y < height and 0 <= x < width):
            return

        # Check if cell is obstacle or already visited
        if visited[y, x]:
            return

        # Mark as visited and add to path
        visited[y, x] = True
        path.append((x, y))

        # Try to move horizontally in current direction
        next_x = x + direction
        if 0 <= next_x < width and not visited[y, next_x]:
            navigate(y, next_x, direction)

        # Try to move down
        next_y = y + 1
        if next_y < height and not visited[next_y, x]:
            # When moving down, reverse direction for boustrophedon pattern
            navigate(next_y, x, -direction)

        # Try to move up if we couldn't move down
        next_y = y - 1
        if next_y >= 0 and not visited[next_y, x]:
            # When moving up, keep same direction
            navigate(next_y, x, direction)

    # Find a good starting point (bottom left if possible)
    start_x, start_y = None, None

    # Try to start from bottom left
    for x in range(width):
        for y in range(height - 1, -1, -1):  # Start from bottom
            if not visited[y, x]:
                start_y, start_x = y, x
                break
        if start_y is not None:
            break

    # If no starting point found, return empty path
    if start_y is None:
        return []

    # Start navigation with a direction of 1 (right)
    navigate(start_y, start_x, 1)

    return path


# Function to animate the traversal
def animate_traversal(map_grid, separate_img, num_cells, path):
    """
    Animate the traversal of the map in a zig-zag pattern

    Args:
        map_grid: Original map with obstacles
        separate_img: Image with cell decomposition
        num_cells: Number of cells
        path: List of (x, y) coordinates to traverse
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Display original map with obstacles in black
    color_img = np.ones((*map_grid.shape, 3), dtype=np.uint8) * 255  # White background
    color_img[map_grid == 1] = [0, 0, 0]  # Black obstacles

    # Display original map
    ax1.imshow(color_img, origin="lower")
    ax1.set_title("Original Map")

    # Display cell decomposition with colors
    cmap = cm.get_cmap("tab20", num_cells + 1)
    norm = colors.Normalize(vmin=0, vmax=num_cells)

    # Set obstacle cells to black in the decomposition map
    display_img = separate_img.copy()

    # Create a masked array to display obstacles as black
    masked_img = np.ma.masked_where(display_img == 0, display_img)

    # Create the image with masked data and colormap
    ax2.imshow(masked_img, cmap=cmap, norm=norm, origin="lower")

    # Add black for obstacles (where mask is True)
    obstacle_mask = display_img == 0
    obstacle_img = np.zeros((*display_img.shape, 4))  # RGBA image for obstacles
    obstacle_img[obstacle_mask, 3] = 1  # Set alpha to 1 for obstacles
    ax2.imshow(obstacle_img, origin="lower")

    ax2.set_title("Boustrophedon Cellular Decomposition with Traversal")

    # Plot initial position (empty)
    (point1,) = ax1.plot([], [], "ro", markersize=8)
    (point2,) = ax2.plot([], [], "ro", markersize=8)

    # Add trail to show the path history
    (trail1,) = ax1.plot([], [], "r-", linewidth=2, alpha=0.5)
    (trail2,) = ax2.plot([], [], "r-", linewidth=2, alpha=0.5)

    plt.tight_layout()

    # Animation speed control (higher = slower)
    speed = [10]  # frames to skip (higher = slower)

    # Add keyboard handler for controlling speed
    def on_key(event):
        if event.key == "right":  # Speed up
            speed[0] = max(1, speed[0] - 1)
        elif event.key == "left":  # Slow down
            speed[0] = min(50, speed[0] + 5)
        elif event.key == "escape":  # Exit
            plt.close(fig)
            return False
        return None

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Keep track of path history
    path_history_x = []
    path_history_y = []

    # Animation function
    def animate(i):
        if i >= len(path):
            return point1, point2, trail1, trail2

        # Skip frames based on speed
        frame_idx = min(len(path) - 1, i // speed[0])
        x, y = path[frame_idx]

        # Set data as sequences (lists) rather than single values
        point1.set_data([x], [y])
        point2.set_data([x], [y])

        # Add current point to path history
        path_history_x.append(x)
        path_history_y.append(y)

        # Update trail
        trail1.set_data(path_history_x, path_history_y)
        trail2.set_data(path_history_x, path_history_y)

        return point1, point2, trail1, trail2

    # Create animation with faster interval
    frames = len(path) * speed[0]
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, blit=True)

    plt.show()

    return True


def main():
    size = 15  # Default value
    traverse_mode = False

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            print(
                "Dimensiunea hărții trebuie să fie un număr întreg. Se folosește valoarea implicită de 15."
            )
            size = 15

    # Check for traverse mode
    if len(sys.argv) > 2 and sys.argv[2].lower() == "traverse":
        traverse_mode = True

    seed = 100  # Initial seed
    regenerate = True

    while regenerate:
        map_grid = create_map(size, seed)
        separate_img, num_cells, cell_boundaries, x_coordinates, non_neighboor_cells = (
            bcd(map_grid)
        )

        if traverse_mode:
            # Calculate and animate zigzag path
            path = calculate_zigzag_path(separate_img, cell_boundaries, x_coordinates)
            regenerate = animate_traversal(map_grid, separate_img, num_cells, path)
        else:
            # Display maps side by side
            regenerate = display_maps(map_grid, separate_img, num_cells)

        if regenerate:
            seed += 1  # Increment seed to generate a different map


if __name__ == "__main__":
    main()

# Informații despre utilizare
# 1. Rulați scriptul fără argumente pentru a genera o hartă de dimensiune 15x15.
# 2. Rulați scriptul cu un argument pentru a specifica dimensiunea hărții (ex: python proiect.py 20).
# 3. Rulați scriptul cu al doilea argument "traverse" pentru a activa animația de traversare (ex: python proiect.py 20 traverse).
# 4. În modul normal: Apăsați Enter pentru a genera o nouă hartă.
# 5. În modul normal/traversare: Apăsați Escape pentru a ieși din aplicație.
# 6. În modul traversare: Apăsați săgeata stânga pentru a încetini animația.
# 7. În modul traversare: Apăsați săgeata dreapta pentru a accelera animația.
