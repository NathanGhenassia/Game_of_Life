"""
Performance Tester for Game of Life
-------------------------------------
This script benchmarks the performance of the GameOfLife class by measuring the
average time taken per iteration across various grid sizes.

It also visualizes the empirical results alongside theoretical time complexity curves.

Author: Nathan Ghenassia, Maria Fernanda Camacho
"""

import time
import math
import matplotlib.pyplot as plt
import importlib
import os

class PerformanceTester:
    """
    Runs performance tests on selected GameOfLife implementation across various grid sizes.
    Measures average step time and plots results against theoretical complexity curves.
    """
    def __init__(self):
        self.sizes = [32, 64, 128, 256, 512, 1024]
        self.steps = self._ask_steps()
        self.module = self._ask_implementation()
        os.makedirs("results", exist_ok=True)

    def _ask_steps(self):
        """
        Prompt user for number of steps.
        """
        while True:
            try:
                steps = int(input("Enter number of steps per test case: "))
                if steps >= 1:
                    return steps
                else:
                    print("Please enter a number >= 1.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

    def _ask_implementation(self):
        """
        Prompt user to select implementation version.
        """
        print("Choose implementation: [1] main.py (slow) | [2] main_numba.py (fast)")
        while True:
            version = input("Enter 1 or 2: ").strip()
            if version == "1":
                return importlib.import_module("main")
            elif version == "2":
                return importlib.import_module("main_numba")
            else:
                print("Invalid input. Please enter 1 or 2.")

    def _measure_performance(self, grid_size):
        """
        Measure average step time for a given grid size.
        """
        game = self.module.GameOfLife(rows=grid_size, cols=grid_size, random_init=True, prob_alive=0.2)
        game.step()  # Warm-up
        start = time.time()
        for _ in range(self.steps):
            game.step()
        end = time.time()
        return (end - start) / self.steps

    def _theoretical_curves(self, sizes):
        """
        Return theoretical complexity curves for reference.
        """
        return {
            "O(n log n)": [n * math.log2(n) for n in sizes],
            "O(n²)": [n ** 2 for n in sizes],
            "O(2^log n)": [2 ** (math.log2(n)) for n in sizes],
            "O(n!)": [math.factorial(int(math.log2(n))) for n in sizes]
        }

    def _plot_results(self, sizes, times):
        """
        Plot performance results and theoretical complexity curves.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, marker='o', linestyle='-', label='Empirical')
        for x, y in zip(sizes, times):
            label = f"{x}x{x}\n{y:.4f}s"
            plt.text(x, y, label, fontsize=8, ha='right', va='bottom')

        theoretical = self._theoretical_curves(sizes)
        max_y = max(times)
        for label, values in theoretical.items():
            scaled = [v / max(values) * max_y for v in values]
            plt.plot(sizes, scaled, linestyle='--', label=label)

        impl = "numba" if "numba" in self.module.__name__ else "sequential"
        filename = f"results/performance_{impl}.png"

        plt.title(f"Game of Life Performance ({self.steps} steps)")
        plt.xlabel("Grid Size (NxN)")
        plt.ylabel("Avg Time per Iteration (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Plot saved to {filename}")

    def run(self):
        """
        Run performance test across all predefined grid sizes.
        """
        results = []
        print(f"{'Grid Size':>10} | {'Avg Time per Iteration (s)':>30}")
        print("-" * 45)

        for size in self.sizes:
            avg_time = self._measure_performance(size)
            results.append((size, avg_time))
            print(f"{size:>10} | {avg_time:>30.6f}")

        sizes, times = zip(*results)
        self._plot_results(sizes, times)


if __name__ == "__main__":
    tester = PerformanceTester()
    tester.run()