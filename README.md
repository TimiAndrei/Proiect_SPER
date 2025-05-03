# Boustrophedon Cellular Decomposition (BCD) Path Planning

A robot path planning algorithm that decomposes the environment into cells and creates a traversal path to cover all accessible areas. The implementation includes both classic BCD traversal and BCD with BFS transitions between cells.

---

## Features

- Cellular decomposition of the environment
- Two traversal modes:
  - Classic BCD: Simple zigzag pattern within cells
  - BCD with BFS: Zigzag pattern within cells with BFS transitions between cells
- Interactive visualization with animation controls
- Obstacle avoidance
- Complete coverage path planning
- Side-by-side comparison of traversal methods

---

## Requirements

- Python 3.x
- NumPy
- Matplotlib

---

## Installation

1. Clone the repository.
2. Install the required packages:

   ```bash
   pip install numpy matplotlib
   ```

---

## Usage

Run the script with the following command:

```bash
python proiect.py [map_size] [--mode classic|bfs|both]
```

### Arguments

- `map_size`: Optional integer argument specifying the size of the square map (default: 15).
- `--mode`: Optional argument to select the traversal mode:
  - `classic`: Classic BCD (no BFS between cells, just zigzag and jump).
  - `bfs`: BCD with BFS transitions (zigzag in each cell, BFS between cells).
  - `both`: Show both traversals side by side for comparison.

### Examples

1. Default map size with classic traversal:

   ```bash
   python proiect.py
   ```

2. 20x20 map with BFS traversal:

   ```bash
   python proiect.py 20 --mode bfs
   ```

3. 25x25 map with side-by-side comparison:

   ```bash
   python proiect.py 25 --mode both
   ```

---

## Controls

During animation:

- `LEFT ARROW`: Slow down animation.
- `RIGHT ARROW`: Speed up animation (to max).
- `R`: Restart animation (when complete).
- `O`: Fewer obstacles in the next map.
- `P`: More obstacles in the next map.
- `ENTER`: Generate a new map with current settings.
- `ESC`: Exit the application.

---

## Algorithm Details

### 1. **Boustrophedon Cellular Decomposition**

The Boustrophedon Cellular Decomposition (BCD) algorithm divides the environment into cells at critical points where obstacles change the connectivity of the free space. Each cell is then covered using a back-and-forth motion (zigzag pattern). This ensures complete coverage of the environment.

#### Key Steps in BCD

1. **Map Representation**:

   - The environment is represented as a 2D grid (`map_grid`) where:
     - `1` represents obstacles.
     - `0` represents free space.

2. **Column-by-Column Scanning**:

   - The algorithm scans the map column by column to identify connected free spaces (cells).

3. **Connectivity Calculation**:

   - For each column, the `calc_connectivity` function determines the number of connected free spaces and their boundaries.

4. **Cell Assignment**:

   - Cells are assigned unique IDs, and their boundaries are stored in a dictionary (`cell_boundaries`).

5. **Adjacency Handling**:

   - The `get_adjacency_matrix` function determines how cells in the current column connect to cells in the previous column.

6. **Output**:
   - The algorithm outputs a decomposed map (`separate_img`) where each cell is assigned a unique ID.

#### Implementation

```python
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
```

#### Visualization

Watch the **cellular decomposition** in action:  
[![Cellular Decomposition](https://img.youtube.com/vi/g32WTfcHkbk/0.jpg)](https://youtu.be/g32WTfcHkbk)

---

### 2. **Classic BCD Traversal**

In this mode:

- The robot traverses each cell in a zigzag pattern.
- It directly jumps between cells without considering obstacles.

#### Key Steps

1. **Zigzag Traversal**:

   - The robot moves back and forth within each cell, covering all free spaces.

2. **Jump Between Cells**:
   - When a cell is fully traversed, the robot jumps directly to the next cell.

#### Implementation

```python
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

    # Start from top-left and continue until all accessible cells are visited
    remaining = np.sum(~visited)

    while remaining > 0:
        # If we're just starting or need to find a new point
        if current_pos[0] is None:
            # Find the first unvisited point
            for y in range(height):
                for x in range(width):
                    if not visited[y, x]:
                        current_pos = [y, x]
                        break
                if current_pos[0] is not None:
                    break

            # If no unvisited points, we're done
            if current_pos[0] is None:
                break

        # Mark current position as visited and add to path
        visited[current_pos[0], current_pos[1]] = True
        path.append((current_pos[1], current_pos[0]))

        # Move to the next cell in zigzag pattern
        if current_pos[1] + 1 < width and not visited[current_pos[0], current_pos[1] + 1]:
            current_pos[1] += 1
        elif current_pos[0] + 1 < height and not visited[current_pos[0] + 1, current_pos[1]]:
            current_pos[0] += 1
        else:
            break

        # Recalculate remaining unvisited cells
        remaining = np.sum(~visited)

    return path
```

#### Visualization

Watch the **classic BCD traversal** in action:  
[![Classic BCD Traversal](https://img.youtube.com/vi/xk15Q1_NA-A/0.jpg)](https://youtu.be/xk15Q1_NA-A)

---

### 3. **BCD Traversal with BFS**

In this mode:

- The robot traverses each cell in a zigzag pattern.
- It uses Breadth-First Search (BFS) to find the shortest path between cells, avoiding obstacles.

#### Key Steps

1. **Zigzag Traversal**:

   - The robot moves back and forth within each cell, covering all free spaces.

2. **BFS Between Cells**:
   - When a cell is fully traversed, the robot uses BFS to find the shortest path to the next cell, avoiding obstacles.

#### Implementation

```python
def calculate_zigzag_path_bfs(separate_img, cell_boundaries, x_coordinates, map_grid):
    path = []
    prev_end = None

    for cell in sorted(cell_boundaries.keys()):
        x_cols = x_coordinates[cell]
        y_ranges = cell_boundaries[cell]
        cell_path = []
        for i, (col, (row_start, row_end)) in enumerate(zip(x_cols, y_ranges)):
            rng = range(row_start, row_end)
            if i % 2 == 0:
                cell_path.extend((col, row) for row in rng)
            else:
                cell_path.extend((col, row) for row in reversed(rng))
        if prev_end is not None:
            bfs = bfs_path(prev_end, cell_path[0], map_grid)
            path.extend(bfs)
        path.extend(cell_path)
        prev_end = cell_path[-1]
    return path
```

#### Visualization

Watch the **BCD traversal with BFS** in action:  
[![BCD Traversal with BFS](https://img.youtube.com/vi/gjqcStOCpy8/0.jpg)](https://youtu.be/gjqcStOCpy8)

---

### 4. **Side-by-Side Comparison**

This mode shows both the classic BCD traversal and the BCD traversal with BFS side by side for comparison.

#### Visualization

Watch the **side-by-side comparison** in action:  
[![Side-by-Side Comparison](https://img.youtube.com/vi/FzXXPnvEGzI/0.jpg)](https://youtu.be/FzXXPnvEGzI)

---

## Visualization

The visualization shows:

- Cell decomposition with different colors for each cell.
- Obstacles in black.
- Robot position as a red dot.
- Traversed path in red.
- Completion status in the title.

---
