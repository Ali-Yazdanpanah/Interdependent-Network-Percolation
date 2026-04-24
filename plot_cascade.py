from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cascade_sim
except ImportError as exc:
    raise SystemExit(
        "Cannot import cascade_sim. From the project root, build the extension: "
        "  pip install -e '.[dev]'\n"
    ) from exc
from site_percolation import (
    generate_network,
    graph_to_adjacency_list,
    interdependent_percolation as py_interdependent_percolation,
    site_percolation,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def benchmark_python_vs_cpp(
    p: float = 0.4,
    n: int = 10_000,
    k: float = 4.0,
    n_trials: int = 20,
    graph_seed_a: int = 314_159,
    graph_seed_b: int = 314_160,
) -> tuple[float, float, float, float]:
    """
    Return mean and sample std (ddof=1) of wall time per trial for Python vs C++.
    Each trial uses a distinct RNG seed; warm-up is done once before timing.
    """
    G_A = generate_network(n, k, seed=graph_seed_a)
    G_B = generate_network(n, k, seed=graph_seed_b)
    adj_A = graph_to_adjacency_list(G_A, n)
    adj_B = graph_to_adjacency_list(G_B, n)

    py_interdependent_percolation(G_A, G_B, p, np.random.default_rng(0))
    cascade_sim.interdependent_percolation(adj_A, adj_B, p, 0)

    py_times: list[float] = []
    cpp_times: list[float] = []
    for t in range(n_trials):
        run_seed = 10_000 + t
        t0 = time.perf_counter()
        py_interdependent_percolation(G_A, G_B, p, np.random.default_rng(run_seed))
        py_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cascade_sim.interdependent_percolation(adj_A, adj_B, p, run_seed)
        cpp_times.append(time.perf_counter() - t0)

    py_arr = np.array(py_times, dtype=float)
    cpp_arr = np.array(cpp_times, dtype=float)
    mean_py, std_py = float(py_arr.mean()), float(py_arr.std(ddof=1) if n_trials > 1 else 0.0)
    mean_cpp, std_cpp = float(cpp_arr.mean()), float(
        cpp_arr.std(ddof=1) if n_trials > 1 else 0.0
    )
    return mean_py, std_py, mean_cpp, std_cpp


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase transition & benchmark figures")
    ap.add_argument("--seed", type=int, default=12_345, help="Base RNG seed for graph & MC")
    ap.add_argument(
        "--figure-dir", type=Path, default=PROJECT_ROOT / "figures", help="Output directory"
    )
    ap.add_argument("--n", type=int, default=1000, help="ER graph size for phase plot")
    ap.add_argument("--k", type=float, default=4.0, help="Target mean degree")
    ap.add_argument("--mc-runs", type=int, default=10, help="MC runs per p (phase plot)")
    ap.add_argument(
        "--p-step", type=float, default=0.02, help="p grid from 0 to 1 inclusive"
    )
    ap.add_argument(
        "--benchmark-trials",
        type=int,
        default=20,
        help="Number of timed trials for the benchmark bar chart",
    )
    ap.add_argument(
        "--big-n", type=int, default=10_000, help="n for the runtime benchmark"
    )
    args = ap.parse_args(argv)
    n_steps = int(round(1.0 / args.p_step)) + 1
    p_values = np.linspace(0.0, 1.0, n_steps, dtype=float)

    figures_dir = args.figure_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    graph_seed_rng = np.random.default_rng(args.seed)
    g_seed_a = int(graph_seed_rng.integers(0, 2**31 - 1))
    g_seed_b = int(graph_seed_rng.integers(0, 2**31 - 1))
    G_A = generate_network(args.n, args.k, seed=g_seed_a)
    G_B = generate_network(args.n, args.k, seed=g_seed_b)
    adj_A = graph_to_adjacency_list(G_A, args.n)
    adj_B = graph_to_adjacency_list(G_B, args.n)

    avg_lcc_single = np.zeros_like(p_values, dtype=float)
    avg_lcc_interdep = np.zeros_like(p_values, dtype=float)
    run_seed_rng = np.random.default_rng(int(args.seed) + 1)

    for i, p in enumerate(p_values):
        total_single = 0.0
        total_interdep = 0.0
        for _ in range(args.mc_runs):
            s = int(run_seed_rng.integers(0, 2**63 - 1, dtype=np.int64))
            total_single += site_percolation(G_A, float(p), rng=np.random.default_rng(s))
            total_interdep += cascade_sim.interdependent_percolation(adj_A, adj_B, float(p), s)
        avg_lcc_single[i] = total_single / args.mc_runs
        avg_lcc_interdep[i] = total_interdep / args.mc_runs

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

    m_py, s_py, m_cpp, s_cpp = benchmark_python_vs_cpp(
        p=0.4, n=args.big_n, k=args.k, n_trials=args.benchmark_trials
    )
    means = [m_py, m_cpp]
    stds = [s_py, s_cpp]
    _, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(
        ["Pure Python", "pybind11 C++"],
        means,
        yerr=stds,
        color=["#4c78a8", "#f58518"],
        width=0.6,
        capsize=4.0,
        ecolor="black",
    )
    ax.set_ylabel("Execution time (seconds)")
    ax.set_title(
        f"Runtime benchmark: n={args.big_n}, p=0.4 ({args.benchmark_trials} trials, mean ± 1 s.d.)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    tops = [m + s for m, s in zip(means, stds, strict=True)]
    ymax = max(tops) if tops else 0.0
    pad = max(ymax * 0.08, 1e-9)
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        label = f"{m:.4f} ± {s:.4f}s" if s > 0 else f"{m:.4f}s"
        ax.text(i, m + s + pad, label, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, (ymax + 2 * pad) if ymax > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(figures_dir / "benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
