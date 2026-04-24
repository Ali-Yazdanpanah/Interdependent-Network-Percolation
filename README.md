# Interdependent Network Percolation: A High-Performance C++/Python Simulation Engine

## Abstract / Overview

This project models cascading failures in **coupled complex networks** and studies how interdependence changes failure dynamics from a **second-order phase transition** (gradual degradation) to a **first-order phase transition** (abrupt systemic collapse).  

The implementation is intentionally hybrid: Python orchestrates experiments and visualization, while the cascade core is executed in C++ and exported to Python via `pybind11`. This removes major runtime overhead associated with Python-level object traversal in repeated Monte Carlo experiments by using cache-friendly graph memory structures (adjacency lists; extensible to CSR layouts) and low-level BFS loops.

## Theoretical Background

For an Erdős–Rényi random graph $G(n, p_e)$, each edge appears independently with probability:

$$
p_e = \frac{k}{n-1}
$$

where $k$ is the target mean degree.

In **site percolation**, a fraction $p$ of nodes is removed, and resilience is measured by the normalized size of the Largest Connected Component (LCC):

$$
S(p) = \frac{|LCC(p)|}{n}
$$

For a single network, $S(p)$ typically decays smoothly. In coupled networks with one-to-one dependencies, failures propagate across layers and trigger iterative dependency + connectivity pruning, which can induce discontinuous (first-order-like) collapse behavior.

## System Architecture

- **Python layer**
  - Generates ER graphs (`networkx`)
  - Runs Monte Carlo experiment grids
  - Produces publication-quality figures (`matplotlib`)
- **C++ layer (`pybind11`)**
  - Executes interdependent cascade kernel
  - Performs adjacency-list BFS to compute LCC under active-node masks
  - Minimizes Python overhead in critical loops
- **Binding layer**
  - `PYBIND11_MODULE(cascade_sim, m)` exposes C++ functions as importable Python module APIs

## Visualizations

### Phase Transition: Single vs Interdependent Networks

![Phase transition comparison](figures/phase_transition.png)

*Figure 1. Normalized LCC size $S(p)$ versus node failure probability $p$. The single-network baseline exhibits smooth degradation, while the interdependent system collapses more sharply due to iterative cross-layer dependency failures.*

### Runtime Benchmark: Python vs C++ Engine

![Python vs C++ benchmark](figures/benchmark.png)

*Figure 2. Single-run runtime at $n = 10^4$, $p = 0.4$. The pybind11-backed C++ cascade kernel reduces execution time relative to the pure Python implementation, demonstrating the benefit of moving BFS and cascade loops into compiled code.*

## Installation & Build Instructions

From the project root:

```bash
cd /Users/aliyazdanpanah/Work/site-percolation
python3 -m venv .venv
source .venv/bin/activate
pip install numpy networkx matplotlib pybind11
```

Build the C++ extension:

```bash
cmake -S . -B build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
```

Run experiments and generate figures:

```bash
PYTHONPATH=build python plot_cascade.py
```

Expected outputs:
- `figures/phase_transition.png`
- `figures/benchmark.png`

## Future work

- **GPU-accelerated Monte Carlo:** parallelize independent cascade realizations and parameter sweeps on CUDA (e.g. batched runs over seeds and $p$) to scale experiments to larger graphs and higher replication counts with minimal host–device transfer overhead.

## Author

Portfolio: [ali-yazdanpanah.github.io](https://ali-yazdanpanah.github.io/)
