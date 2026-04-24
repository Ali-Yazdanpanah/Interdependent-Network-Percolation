from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from site_percolation import interdependent_percolation as py_interdependent_percolation
from site_percolation import site_percolation

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import cascade_sim  # noqa: E402


def generate_network(n: int, k: float, seed: int | None = None) -> nx.Graph:
    if n <= 0:
        return nx.empty_graph(0)
    p_edge = k / (n - 1) if n > 1 else 0.0
    return nx.erdos_renyi_graph(n, p_edge, seed=seed)


def to_adjacency_list(G: nx.Graph, n: int) -> list[list[int]]:
    return [list(G.neighbors(i)) for i in range(n)]


def benchmark_python_vs_cpp(p: float = 0.4, n: int = 10_000, k: float = 4.0) -> tuple[float, float]:
    graph_seed = 314159
    run_seed = 424242

    G_A = generate_network(n, k, seed=graph_seed)
    G_B = generate_network(n, k, seed=graph_seed + 1)
    adj_A = to_adjacency_list(G_A, n)
    adj_B = to_adjacency_list(G_B, n)

    # Warm-up calls to reduce one-time overhead noise.
    py_interdependent_percolation(G_A, G_B, p, np.random.default_rng(run_seed))
    cascade_sim.interdependent_percolation(adj_A, adj_B, p, run_seed)

    start_py = time.perf_counter()
    py_interdependent_percolation(G_A, G_B, p, np.random.default_rng(run_seed))
    py_time = time.perf_counter() - start_py

    start_cpp = time.perf_counter()
    cascade_sim.interdependent_percolation(adj_A, adj_B, p, run_seed)
    cpp_time = time.perf_counter() - start_cpp
    return py_time, cpp_time


def main() -> None:
    figures_dir = PROJECT_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    n = 1000
    k = 4.0
    mc_runs = 10
    p_values = np.arange(0.0, 1.0001, 0.02)

    graph_seed_rng = np.random.default_rng(12345)
    G_A = generate_network(n, k, seed=int(graph_seed_rng.integers(0, 2**31 - 1)))
    G_B = generate_network(n, k, seed=int(graph_seed_rng.integers(0, 2**31 - 1)))
    adj_A = to_adjacency_list(G_A, n)
    adj_B = to_adjacency_list(G_B, n)

    avg_lcc_single = np.zeros_like(p_values, dtype=float)
    avg_lcc_interdep = np.zeros_like(p_values, dtype=float)
    run_seed_rng = np.random.default_rng(2026)

    for i, p in enumerate(p_values):
        total_single = 0.0
        total_interdep = 0.0
        for _ in range(mc_runs):
            seed = int(run_seed_rng.integers(0, 2**63 - 1, dtype=np.int64))
            total_single += site_percolation(G_A, float(p), rng=np.random.default_rng(seed))
            total_interdep += cascade_sim.interdependent_percolation(adj_A, adj_B, float(p), seed)
        avg_lcc_single[i] = total_single / mc_runs
        avg_lcc_interdep[i] = total_interdep / mc_runs

    plt.figure(figsize=(9, 5.5))
    plt.plot(p_values, avg_lcc_single, linewidth=2.0, color="C0", label="Single Network")
    plt.plot(
        p_values,
        avg_lcc_interdep,
        linewidth=2.0,
        color="C1",
        label="Interdependent Networks",
    )
    plt.xlabel("Failure probability p")
    plt.ylabel("Average LCC size / n")
    plt.title("Percolation Phase Transition in Coupled Erdős–Rényi Networks")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "phase_transition.png", dpi=300, bbox_inches="tight")
    plt.close()

    py_time, cpp_time = benchmark_python_vs_cpp(p=0.4, n=10_000, k=4.0)
    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(
        ["Pure Python", "Pybind11 C++"],
        [py_time, cpp_time],
        color=["#4c78a8", "#f58518"],
        width=0.6,
    )
    plt.ylabel("Execution time (seconds)")
    plt.title("Single-run Benchmark at n=10,000, p=0.4")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, value in zip(bars, [py_time, cpp_time]):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}s",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(figures_dir / "benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
