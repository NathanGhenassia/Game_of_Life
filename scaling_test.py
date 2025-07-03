"""
Parallel Scaling Benchmark for Game of Life
--------------------------------------------
Benchmarking script for strong and weak scaling of the Numba-accelerated
Game of Life simulation. Uses matplotlib to visualize performance metrics.

This script imports the compute_next_step function from `main_numba.py`
and evaluates runtime across multiple threads.

Authors: Nathan Ghenassia, Maria Fernanda Camacho
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import set_num_threads
from main_numba import compute_next_step

def measure_runtime(grid: np.ndarray, steps: int) -> float:
    """
    Measures total execution time for a number of Game of Life steps.

    Parameters:
    - grid (np.ndarray): Initial state of the Game of Life grid.
    - steps (int): Number of iterations to simulate.

    Returns:
    - float: Total execution time in seconds.
    """
    start = time.time()
    for _ in range(steps):
        grid = compute_next_step(grid)
    return time.time() - start


class ScalingTest:
    """
    Class encapsulating strong and weak scaling tests.
    """
    def __init__(self):
        self.results = []

    def strong_scaling_test(self):
        """
        Run strong scaling test: fixed problem size, increasing threads.
        """
        size = self._prompt_choice("Enter fixed grid size 32, 64, 128, 256, 512, 1024:", [32, 64, 128, 256, 512, 1024])
        steps = self._prompt_int("Number of iterations to run:")
        min_threads = self._prompt_int("Minimum number of threads:")
        max_threads = self._prompt_int("Maximum number of threads:")

        base_grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15]).astype(np.uint8)
        self.results.clear()

        for threads in range(min_threads, max_threads + 1):
            set_num_threads(threads)
            runtime = measure_runtime(base_grid.copy(), steps)
            self.results.append((threads, runtime))

        self._plot_scaling_results(kind="strong")

    def weak_scaling_test(self):
        """
        Run weak scaling test: problem size increases with threads.
        """
        workload_per_thread = self._prompt_int("Enter workload per thread (e.g., 10000):")
        steps = self._prompt_int("Number of iterations to run:")
        min_threads = self._prompt_int("Minimum number of threads:")
        max_threads = self._prompt_int("Maximum number of threads:")

        self.results.clear()

        for threads in range(min_threads, max_threads + 1):
            set_num_threads(threads)
            total_cells = threads * workload_per_thread
            size = int(np.sqrt(total_cells))
            grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15]).astype(np.uint8)
            runtime = measure_runtime(grid, steps)
            self.results.append((threads, runtime))

        self._plot_scaling_results(kind="weak")

    def _plot_scaling_results(self, kind: str):
        """
        Plot speedup and efficiency for strong or weak scaling.

        Parameters:
        - kind (str): Either 'strong' or 'weak'.
        """
        threads_list, runtimes = zip(*self.results)
        base_time = runtimes[0]

        if kind == "strong":
            speedups = [base_time / t for t in runtimes]
            efficiencies = [s / p for s, p in zip(speedups, threads_list)]

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(threads_list, speedups, marker='o')
            plt.title("Strong Scaling: Speedup")
            plt.xlabel("Threads")
            plt.ylabel("Speedup")

            plt.subplot(1, 2, 2)
            plt.plot(threads_list, efficiencies, marker='o')
            plt.title("Strong Scaling: Efficiency")
            plt.xlabel("Threads")
            plt.ylabel("Efficiency")

            self._save_figure("strong_scaling.png")

        elif kind == "weak":
            efficiencies = [base_time / t for t in runtimes]

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(threads_list, runtimes, marker='o')
            plt.title("Weak Scaling: Runtime")
            plt.xlabel("Threads")
            plt.ylabel("Execution Time (s)")

            plt.subplot(1, 2, 2)
            plt.plot(threads_list, efficiencies, marker='o')
            plt.title("Weak Scaling: Efficiency")
            plt.xlabel("Threads")
            plt.ylabel("Efficiency")

            self._save_figure("weak_scaling.png")

        plt.tight_layout()
        plt.show()

    def _save_figure(self, filename: str):
        """
        Save the current matplotlib figure to a 'results' directory.

        Parameters:
        - filename (str): Name of the file to save.
        """
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        plt.savefig(filepath)
        print(f"Saved figure to {filepath}")

    @staticmethod
    def _prompt_int(prompt: str) -> int:
        while True:
            try:
                return int(input(prompt).strip())
            except ValueError:
                print("Please enter a valid integer.")

    @staticmethod
    def _prompt_choice(prompt: str, options: list) -> int:
        while True:
            try:
                value = int(input(prompt).strip())
                if value in options:
                    return value
                print(f"Please select one of: {options}")
            except ValueError:
                print("Invalid input. Must be a number.")


if __name__ == "__main__":
    test = ScalingTest()
    mode = input("Choose mode: (1) Strong Scaling, (2) Weak Scaling: ").strip()
    if mode == '1':
        test.strong_scaling_test()
    else:
        test.weak_scaling_test()