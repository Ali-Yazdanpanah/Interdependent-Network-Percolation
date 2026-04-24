"""
Site percolation on an Erdős–Rényi random graph:
Monte Carlo average of the largest connected component (LCC) size vs. node failure probability.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generate_network(n: int, k: float, seed: int | None = None) -> nx.Graph:
    """
    Erdős–Rényi G(n, p) with edge probability p = k / (n - 1) so expected mean degree is k.
    """
    if n <= 0:
        return nx.empty_graph(0, create_using=nx.Graph)
    p_edge = k / (n - 1) if n > 1 else 0.0
    return nx.erdos_renyi_graph(n, p_edge, seed=seed)


def site_percolation(
    G: nx.Graph,
    p: float,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Randomly remove a fraction p of the nodes, then return LCC size / original n.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    k_remove = int(round(p * n))
    k_remove = min(max(k_remove, 0), n)
    nodes = list(G.nodes())
    to_remove = set(rng.choice(nodes, size=k_remove, replace=False))
    remaining = [u for u in nodes if u not in to_remove]
    if not remaining:
        return 0.0
    sub = G.subgraph(remaining)
    ccs = nx.connected_components(sub)
    lcc = max((len(c) for c in ccs), default=0)
    return lcc / n


def graph_to_adjacency_list(G: nx.Graph, n: int) -> list[list[int]]:
    """Adjacency list aligned with node labels 0..n-1 (used by the C++ engine)."""
    return [list(G.neighbors(i)) for i in range(n)]


def _largest_component_nodes(G: nx.Graph, active_nodes: set[int]) -> set[int]:
    """
    Return nodes in the LCC of G induced by active_nodes.
    """
    if not active_nodes:
        return set()
    sub = G.subgraph(active_nodes)
    largest = max(nx.connected_components(sub), key=len, default=set())
    return set(largest)


def interdependent_percolation(
    G_A: nx.Graph,
    G_B: nx.Graph,
    p: float,
    rng: np.random.Generator,
) -> float:
    """
    Simulate cascading failures on two interdependent networks with 1-to-1 node dependency.
    """
    n = G_A.number_of_nodes()
    if n == 0 or G_B.number_of_nodes() != n:
        return 0.0

    active_A = set(G_A.nodes())
    active_B = set(G_B.nodes())

    # Initial trigger: random attack on network A.
    k_remove = int(round(p * n))
    k_remove = min(max(k_remove, 0), n)
    to_remove = set(rng.choice(list(active_A), size=k_remove, replace=False))
    active_A -= to_remove

    # Cascade until no further changes in surviving sets.
    while True:
        prev_sizes = (len(active_A), len(active_B))

        # a) Dependency failure in B.
        active_B &= active_A
        # b) Connectivity failure in B.
        active_B = _largest_component_nodes(G_B, active_B)
        # c) Dependency failure in A.
        active_A &= active_B
        # d) Connectivity failure in A.
        active_A = _largest_component_nodes(G_A, active_A)

        if (len(active_A), len(active_B)) == prev_sizes:
            break

    return len(active_A) / n


def main() -> None:
    n = 1000
    k = 4.0
    mc_runs = 10
    p_values = np.linspace(0.0, 1.0, 51)  # 0.0, 0.02, …, 1.0

    G_A = generate_network(n, k)
    G_B = generate_network(n, k)
    avg_lcc_single = np.zeros_like(p_values, dtype=float)
    avg_lcc_interdep = np.zeros_like(p_values, dtype=float)

    base_rng = np.random.default_rng(42)
    for i, p in enumerate(p_values):
        total_single = 0.0
        total_interdep = 0.0
        for _ in range(mc_runs):
            total_single += site_percolation(G_A, float(p), rng=base_rng)
            total_interdep += interdependent_percolation(G_A, G_B, float(p), rng=base_rng)
        avg_lcc_single[i] = total_single / mc_runs
        avg_lcc_interdep[i] = total_interdep / mc_runs

    plt.figure(figsize=(8, 5))
    plt.plot(p_values, avg_lcc_single, color="C0", linewidth=1.8, label="Single Network")
    plt.plot(
        p_values,
        avg_lcc_interdep,
        color="C1",
        linewidth=1.8,
        label="Interdependent Networks",
    )
    plt.xlabel("Failure probability p")
    plt.ylabel("Average LCC size / n")
    plt.title("Site Percolation on Erdős–Rényi Graph")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
