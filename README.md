# Boustrophedon Cellular Decomposition with Animation

This project implements a **Boustrophedon Cellular Decomposition** algorithm for path planning and visualizes the robot's path using an animated map. The map contains randomly generated obstacles, and the animation speed can be adjusted interactively.

---

## Features

- **Random Map Generation**: Generates a grid map with random obstacles.
- **Path Planning**: Implements the Boustrophedon Cellular Decomposition algorithm to compute the robot's path.
- **Interactive Animation**: Visualizes the robot's path with adjustable animation speed.
- **Regeneration**: Allows regenerating the map with a new seed for different obstacle configurations.

---

## How to Use

1. **Run the script without arguments** to generate a map of size `15x15`:

   ```bash
   python proiect.py
   ```

2. **Run the script with an argument** to specify the map size (e.g., 20x20):

   ```bash
   python proiect.py 20
   ```

3. **Interactive Controls**:
   - Press **Enter**: Generate a new random map.
   - Press **Escape**: Exit the application.
   - Press **Left Arrow**: Decrease animation speed.
   - Press **Right Arrow**: Increase animation speed.

---

## Code Overview

### 1. **Map Generation**

The `create_map(size, seed)` function generates a grid map of the specified size with random obstacles. The seed ensures reproducibility.

### 2. **Path Planning**

The `bcd(map_grid)` function implements the **Boustrophedon Cellular Decomposition** algorithm to compute the robot's path through the map.

### 3. **Animation**

The `create_animation(path, map_grid, on_close_callback)` function visualizes the robot's path on the map. The animation speed can be adjusted interactively using the arrow keys.

### 4. **Main Function**

The `main()` function handles user input, map generation, and animation. It also supports regenerating the map with a new seed.

---

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib`

Install the required libraries using:

```bash
pip install numpy matplotlib
```

---

## Example Output

### Initial Map

A randomly generated map with obstacles (black cells) and free space (white cells).

### Animated Path

The robot's path is visualized as a red line moving through the map.

---

## Notes

- The default map size is `15x15`. You can specify a custom size by passing an integer argument when running the script.
- The animation speed can be adjusted between `10ms` (fastest) and `1000ms` (slowest).

---

## License

This project is for educational purposes and is not licensed for commercial use.
