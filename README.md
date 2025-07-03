# Conway's Game of Life (Python Implementation)

This project implements **Conway's Game of Life** in Python, with support for interactive GUI visualization and optimized performance via Numba for parallel execution.

It also includes a performance analisis to compare execution time across different grid sizes, profiling and scaling.

---

## Project Structure

```
.
├── main.py               # Base implementation (no parallelism)
├── main_numba.py         # Optimized version using Numba parallel loops
├── performance_test.py # Benchmarks different grid sizes and plots results
├── profile_test.py       # Performance profiling with cProfile and line_profiler
├── scaling_test.py       # Strong and weak scaling analysis
├── performance.md        # Report summarizing all performance analysis
├── results/              # Folder containing images and .txt result files
├── README.md             # Project documentation (this file)

```

---

## Requirements

Install dependencies using pip:

```bash
pip install matplotlib numpy numba line_profiler
```

---

## Running the Game

### 1. **Standard (non-parallel) version**

```bash
python main.py
```

- User is prompted for:
  - Whether to start with a random grid.
  - Grid size (choose from preset list).
- Interact via:
  - `Space` key — Pause/Resume.
  - `Left Click` — Activate cell.
  - `Right Click` — Deactivate cell.

---

### 2. **Numba-parallel optimized version**

```bash
python main_numba.py
```

Same controls and prompts as the standard version, but optimized for faster computation using Numba:

- Uses `@njit(parallel=True)` to compute next state efficiently.

---

## Benchmarking Performance

```bash
python performance_test.py
```

- You will be prompted to enter the number of steps to simulate.
- Tests are run on grid sizes: `32, 64, 128, 256, 512, 1024`
- For each size, it measures average time per step.
- A plot `performance_parallel.png` will be saved and shown, comparing empirical results vs theoretical curves:
  - O(n log n)
  - O(n²)
  - O(2ⁿ)
  - O(n!)

---

## Profiling Performance

```bash
python profile_test.py
```

- Uses cProfile and line_profiler to identify bottlenecks in key functions.
- Results stored in results/ for further inspection.

---

## Scaling Performance

```bash
python scaling_test.py
```

- Evaluates both strong and weak scaling.
- Results saved in text and graphical format under results/.

---


## Features

- Interactive GUI with matplotlib animation
- Mouse + keyboard controls
- Configurable grid size and random initialization
- Numba-accelerated update step with parallelization
- Benchmarking, profiling, and scalability analysis
- Organized result output in /results folder
