# Conway's Game of Life (Python Implementation with Visualization & Benchmarking)

This project implements **Conway's Game of Life** in Python, with support for interactive GUI visualization and optimized performance via Numba for parallel execution.

It also includes a **performance tester** to compare execution time across different grid sizes.

---

## 📁 Project Structure

```
.
├── main.py            # Base implementation (no parallelism)
├── main_numba.py      # Optimized version using Numba parallel loops
├── performance_tester.py # Benchmarks different grid sizes and plots results
├── README.md          # Project documentation (this file)
```

---

## 🧪 Requirements

Install dependencies using pip:

```bash
pip install matplotlib numpy numba
```

---

## 🚀 Running the Game

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

## 📊 Benchmarking Performance

```bash
python performance_tester.py
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

## ✅ Features

- Interactive GUI with matplotlib animation
- Mouse + keyboard controls
- Configurable grid size and random initialization
- Numba-accelerated update step with parallelization
- Performance benchmarking + visual plots

---

## 🔧 Troubleshooting

- If animation or click interaction fails, ensure you are using a Python environment with GUI support (e.g., not headless terminal).
- To avoid Numba JIT compile time in benchmarks, consider pre-warming with a dummy step before timing.