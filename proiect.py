import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import random

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
        map_grid[start_row:start_row + obs_size,
                 start_col:start_col + obs_size] = 1
    return map_grid


# Algoritm Boustrophedon Cellular Decomposition
def bcd(map_grid):
    rows, cols = map_grid.shape
    path = []
    visited = np.zeros_like(map_grid, dtype=bool)

    def navigate(row, col, direction):
        if not (0 <= row < rows and 0 <= col < cols):
            return
        if map_grid[row, col] == 1 or visited[row, col]:
            return
        visited[row, col] = True
        path.append((row, col))
        next_col = col + direction
        navigate(row, next_col, direction)
        next_row = row + 1
        if next_row < rows and not visited[next_row, col]:
            navigate(next_row, col, -direction)

    navigate(0, 0, 1)
    return path


# Crearea animatiei pentru traseul robotului
def create_animation(path, map_grid, on_close_callback):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(map_grid, cmap='Greys', origin='lower')
    line, = ax.plot([], [], 'ro-', linewidth=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        if i < len(path):
            x, y = zip(*path[:i + 1])
            line.set_data(y, x)
        return line,

    # Initialize animation speed
    # Use a mutable object to allow modification inside the event handler
    animation_speed = [100]

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path) + 1, interval=animation_speed[0], blit=True)

    def update_speed():
        ani.event_source.stop()  # Stop the animation
        ani.event_source.interval = animation_speed[0]  # Update the interval
        ani.event_source.start()  # Restart the animation with the new interval

    def on_key(event):
        if event.key == 'enter':  # Apasă Enter pentru a genera o hartă nouă
            plt.close(fig)
            on_close_callback(True)
        elif event.key == 'escape':  # Apasă Escape pentru a ieși
            plt.close(fig)
            on_close_callback(False)
        elif event.key == 'left':  # Reduce speed
            animation_speed[0] = min(
                animation_speed[0] + 50, 1000)  # Cap at 1000ms
            update_speed()
        elif event.key == 'right':  # Increase speed
            animation_speed[0] = max(
                animation_speed[0] - 50, 10)  # Cap at 10ms
            update_speed()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def main():
    size = 15  # Valoare implicită
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])
        except ValueError:
            print(
                "Dimensiunea hărții trebuie să fie un număr întreg. Se folosește valoarea implicită de 15.")
            size = 15

    seed = 100  # Seed inițial
    regenerate = True
    while regenerate:
        map_grid = create_map(size, seed)
        path = bcd(map_grid)

        def on_close(regen):
            nonlocal regenerate, seed
            regenerate = regen
            if regen:
                seed += 1  # Incrementăm seed-ul pentru a genera o hartă diferită

        create_animation(path, map_grid, on_close)


if __name__ == "__main__":
    main()

# Informații despre utilizare
# 1. Rulați scriptul fără argumente pentru a genera o hartă de dimensiune 15x15.
# 2. Rulați scriptul cu un argument pentru a specifica dimensiunea hărții (ex: python proiect.py 20).
# 3. Apăsați Enter pentru a genera o nouă hartă.
# 4. Apăsați Escape pentru a ieși din aplicație.
# 5. Apăsați săgeata stânga pentru a opri viteza animației.
# 6. Apăsați săgeata dreapta pentru a accelera viteza animației.
