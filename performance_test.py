"""
Performance Tester for Game of Life
-------------------------------------
This script benchmarks the performance of the GameOfLife class by measuring the
average time taken per iteration across various grid sizes.

It also visualizes the empirical results alongside theoretical time complexity curves.

Author: Nathan Ghenassia, Maria Feranda Camacho
"""

import time
import math
import matplotlib.pyplot as plt
from main_numba import GameOfLife # Change to the version you want to test (secuencial) or (parallel)

class PerformanceTester:
    """
    Runs performance tests on GameOfLife for varying grid sizes.
    Measures average step time and plots results against complexity curves.
    
    Initializes the tester with predefined grid sizes.
    Prompts the user to enter the number of steps to simulate.
    """
    def __init__(self):
        self.sizes = [32, 64, 128, 256, 512, 1024]
        self.steps = self._ask_steps()

    def _ask_steps(self):
        """
        Prompt user for number of steps per test case.

        Returns:
        - int: Steps per grid size.
        """
        while True:
            try:
                steps = int(input("Ingrese el número de steps por tamaño de grilla (mínimo 1): "))
                if steps >= 1:
                    return steps
                else:
                    print("Por favor ingrese un número mayor o igual a 1.")
            except ValueError:
                print("Entrada inválida. Debe ser un número entero.")

    def _measure_performance(self, grid_size):
        """
        Measures average time per step for a given grid size.

        Parameters:
        - grid_size (int): Grid dimension (NxN).

        Returns:
        - float: Average time per iteration (seconds).
        """
        game = GameOfLife(rows=grid_size, cols=grid_size, random_init=True, prob_alive=0.2)
        game.step()
        start = time.time()
        for _ in range(self.steps):
            game.step()
        end = time.time()
        return (end - start) / self.steps

    def _theoretical_curves(self, sizes):
        """
        Generate theoretical complexity curves for comparison.

        Parameters:
        - sizes (List[int]): Grid sizes.

        Returns:
        - Dict[str, List[float]]: Complexity curve values.
        """
        return {
            "O(n log n)": [n * math.log2(n) for n in sizes],
            "O(n²)": [n ** 2 for n in sizes],
            "O(2ⁿ)": [2 ** (math.log2(n)) for n in sizes],
            "O(n!)": [math.factorial(int(math.log2(n))) for n in sizes]
        }

    def _plot_results(self, sizes, times):
        """
        Plot measured performance results and theoretical curves.

        Parameters:
        - sizes (List[int]): Grid sizes.
        - times (List[float]): Measured times.
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

        plt.title(f"Game of Life Performance ({self.steps} steps)")
        plt.xlabel("Grid Size (NxN)")
        plt.ylabel("Avg Time per Iteration (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("performance_parallel.png", dpi=300) # Change title depending on version
        plt.show()

    def run(self):
        """
        Run performance tests across predefined grid sizes.
        Displays timing results and invokes plot generation.
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