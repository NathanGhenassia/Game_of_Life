"""
Game of Life Implementation (with Numba Acceleration)
-------------------------------------------------------
A Python implementation of Conway's Game of Life using NumPy and matplotlib for visualization.
Performance-optimized using Numba's JIT compiler with parallel execution.

Usage:
- Run as a standalone script
- Press SPACE to pause/resume animation
- Left-click to activate a cell, Right-click to deactivate

Authors: Nathan Ghenassia, Maria Fernanda Camacho
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit, prange

@njit(parallel=True)
def compute_next_step(grid):
    """
    Compute the next generation of the Game of Life grid using parallel loops.

    Parameters:
    - grid (np.ndarray): Current state of the grid (2D array of 0s and 1s).

    Returns:
    - np.ndarray: Updated grid after applying Game of Life rules.
    """
    rows, cols = grid.shape
    new_grid = np.copy(grid)

    for x in prange(rows):
        for y in range(cols):
            total = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % rows
                    ny = (y + dy) % cols
                    total += grid[nx, ny]

            if grid[x, y] == 1:
                if total < 2 or total > 3:
                    new_grid[x, y] = 0
            elif total == 3:
                new_grid[x, y] = 1

    return new_grid

class GameOfLife:
    """
    Class representing the Game of Life simulation.

    Parameters:
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.
    - initial_state (2D list or np.ndarray, optional): User-defined initial state.
    - random_init (bool): If True, initialize with random grid.
    - prob_alive (float): Probability a cell is initially alive in random mode.
    """
    def __init__(self, rows, cols, initial_state=None, random_init=False, prob_alive=0.2):
        self.rows = rows
        self.cols = cols
        self.paused = True

        if initial_state is not None:
            self.grid = np.array(initial_state, dtype=np.uint8)
        else:
            self.grid = (
                np.random.choice([0, 1], size=(rows, cols), p=[1 - prob_alive, prob_alive])
                if random_init else
                np.zeros((rows, cols), dtype=np.uint8)
            )

        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.grid, cmap='gray_r', vmin=0, vmax=1)
        self.ax.set_title("Space: Pause/Resume | Left click: Alive | Right click: Dead")

        self.ani = FuncAnimation(self.fig, self.update, interval=100, blit=False)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def step(self):
        """
        Advance the game state by one iteration using compute_next_state().
        """
        self.grid = compute_next_step(self.grid)

    def run(self, steps=None):
        """
        Run the simulation.

        Parameters:
        - steps (int or None): Number of iterations to run. If None, runs interactively with GUI.
        """
        if steps is None:
            plt.show()
        else:
            for _ in range(steps):
                self.step()

    def update(self, frame):
        """
        Called by FuncAnimation for each animation frame.
        Updates grid state and refreshes plot.
        """
        if not self.paused:
            self.step()
        self.img.set_data(self.grid)
        return [self.img]

    def on_click(self, event):
        """
        Mouse click event handler.
        Left-click toggles cell alive, right-click sets it dead.
        """
        if event.inaxes != self.ax:
            return
        try:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
        except (TypeError, ValueError):
            return

        if 0 <= x < self.cols and 0 <= y < self.rows:
            if event.button == 1:
                self.grid[y, x] = 1
            elif event.button == 3:
                self.grid[y, x] = 0
            self.img.set_data(self.grid)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        """
        Keyboard event handler. Pressing space toggles pause/resume.
        """
        if event.key == ' ':
            self.paused = not self.paused
            print("Running simulation..." if not self.paused else "Paused.")

    @staticmethod
    def ask_if_random(question):
        """
        Prompt the user to choose random initialization.

        Parameters:
        - question (str): Prompt text

        Returns:
        - bool: True if user selects random init.
        """
        while True:
            response = input(question + " (y/n): ").strip().lower()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please respond with 'y' or 'n'.")

    @staticmethod
    def ask_grid_size():
        """
        Prompt the user to select a grid size from a preset list.

        Returns:
        - int: Chosen grid dimension (NxN)
        """
        options = [32, 64, 128, 256, 512, 1024]
        while True:
            print(f"Available sizes: {', '.join(map(str, options))}")
            try:
                value = int(input("Choose grid size (N for NxN): ").strip())
                if value in options:
                    return value
                print("Please select a valid size.")
            except ValueError:
                print("Invalid input. Must be a number.")

if __name__ == "__main__":
    use_random_grid = GameOfLife.ask_if_random("Start with a random grid?")
    size = GameOfLife.ask_grid_size()

    game = GameOfLife(
        rows=size,
        cols=size,
        random_init=use_random_grid,
        prob_alive=0.15
    )
    game.run()
