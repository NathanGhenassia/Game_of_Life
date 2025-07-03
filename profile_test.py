"""
Profiler Utility for Game of Life
----------------------------------
This script allows performance profiling of the GameOfLife implementations:
1. cProfile: function-level timing
2. line_profiler: line-by-line analysis of the `step()` method

Profiles can be saved in the `results/` directory.

Author: Nathan Ghenassia, Maria Fernanda Camacho
"""

import cProfile
import pstats
import importlib
import os
from line_profiler import LineProfiler

class GameProfiler:
    """
    Provides both function-level and line-level profiling for Game of Life.
    """
    def __init__(self):
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_module_and_output(self, mode):
        """
        Prompt user to select which version to profile and return the module and output file path.

        Parameters:
        - mode (str): Either 'profile' or 'line_profile' for output filename context.

        Returns:
        - module: The imported GameOfLife module.
        - str: Output file path for saving profile results.
        """
        version = input(f"Choose version to {mode}: [1] main.py (slow) | [2] main_numba.py (fast): ").strip()
        if version == "1":
            return importlib.import_module("main"), f"{self.results_dir}/{mode}_main.txt"
        elif version == "2":
            return importlib.import_module("main_numba"), f"{self.results_dir}/{mode}_main_numba.txt"
        else:
            print("Invalid option.")
            return None, None

    def _get_simulation_params(self):
        """
        Prompt the user to input grid size and number of steps for simulation.

        Returns:
        - int: Grid size
        - int: Number of steps
        """
        try:
            grid_size = int(input("Enter grid size (32, 64, 128, 512 or 1024): "))
            steps = int(input("Enter number of steps: "))
            return grid_size, steps
        except ValueError:
            print("Invalid input.")
            return None, None

    def run_cprofile(self):
        """
        Run function-level profiling using cProfile and export stats to a .txt file.
        """
        module, out_file = self._get_module_and_output("profile")
        if not module:
            return

        grid_size, steps = self._get_simulation_params()
        if not grid_size:
            return

        game = module.GameOfLife(rows=grid_size, cols=grid_size, random_init=True, prob_alive=0.2)

        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(steps):
            game.step()
        profiler.disable()

        with open(out_file, "w") as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats("cumulative")
            stats.print_stats()

        print(f"cProfile output saved to {out_file}")

    def run_line_profiler(self):
        """
        Run line-by-line profiling of the `step()` function using line_profiler.
        Exports detailed line timings to a .txt file.
        """
        module, out_file = self._get_module_and_output("line_profile")
        if not module:
            return

        grid_size, steps = self._get_simulation_params()
        if not grid_size:
            return

        game = module.GameOfLife(rows=grid_size, cols=grid_size, random_init=True, prob_alive=0.2)

        lp = LineProfiler()
        lp.add_function(game.step)
        for _ in range(steps):
            lp_wrapper = lp(game.step)
            lp_wrapper()

        with open(out_file, "w") as f:
            lp.print_stats(stream=f)

        print(f"Line profiler output saved to {out_file}")

if __name__ == "__main__":
    print("[1] Run cProfile\n[2] Run line_profiler")
    choice = input("Choose mode: ").strip()

    profiler = GameProfiler()

    if choice == "1":
        profiler.run_cprofile()
    elif choice == "2":
        profiler.run_line_profiler()
    else:
        print("Invalid choice.")
