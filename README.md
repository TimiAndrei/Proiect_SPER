# Boustrophedon Cellular Decomposition (BCD) Path Planning

A robot path planning algorithm that decomposes the environment into cells and creates a traversal path to cover all accessible areas. The implementation includes both classic BCD traversal and BCD with BFS transitions between cells.

## Features

- Cellular decomposition of the environment
- Two traversal modes:
  - Classic BCD: Simple zigzag pattern within cells
  - BCD with BFS: Zigzag pattern within cells with BFS transitions between cells
- Interactive visualization with animation controls
- Obstacle avoidance
- Complete coverage path planning
- Side-by-side comparison of traversal methods

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install numpy matplotlib
```

## Usage

Run the script with the following command:

```bash
python proiect.py [map_size] [--mode classic|bfs|both]
```

### Arguments

- `map_size`: Optional integer argument specifying the size of the square map (default: 15)
- `--mode`: Optional argument to select the traversal mode:
  - `classic`: Classic BCD (no BFS between cells, just zigzag and jump)
  - `bfs`: BCD with BFS transitions (zigzag in each cell, BFS between cells)
  - `both`: Show both traversals side by side for comparison

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

## Controls

During animation:

- `LEFT ARROW`: Slow down animation
- `RIGHT ARROW`: Speed up animation (to max)
- `R`: Restart animation (when complete)
- `O`: Fewer obstacles in the next map
- `P`: More obstacles in the next map
- `ENTER`: Generate a new map with current settings
- `ESC`: Exit the application

## Algorithm Details

### Boustrophedon Cellular Decomposition

The algorithm works by:

1. Dividing the environment into cells at critical points where obstacles change the connectivity of the free space
2. Each cell is covered using a back-and-forth motion (zigzag pattern)
3. The robot prioritizes visiting the closest unvisited cells
4. The path visualization avoids cutting through obstacles

### Traversal Modes

#### Classic BCD

- Simple zigzag pattern within each cell
- Direct jumps between cells
- Efficient for simple environments

#### BCD with BFS

- Zigzag pattern within cells
- BFS (Breadth-First Search) for optimal transitions between cells
- Better coverage in complex environments
- Allows revisiting cells when necessary for complete coverage

## Visualization

The visualization shows:

- Cell decomposition with different colors for each cell
- Obstacles in black
- Robot position as a red dot
- Traversed path in red
- Completion status in the title

## License

This project is for educational purposes and is not licensed for
commercial use.
