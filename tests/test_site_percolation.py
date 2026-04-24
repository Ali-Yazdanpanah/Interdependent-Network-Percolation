from __future__ import annotations

import networkx as nx
import numpy as np

from site_percolation import (
    generate_network,
    graph_to_adjacency_list,
    interdependent_percolation,
    site_percolation,
)


def test_graph_to_adjacency_list_path() -> None:
    G = nx.path_graph(4)
    adj = graph_to_adjacency_list(G, 4)
    assert adj == [[1], [0, 2], [1, 3], [2]]


def test_site_percolation_p1_empty() -> None:
    G = generate_network(20, 4.0, seed=0)
    r = site_percolation(G, 1.0, rng=np.random.default_rng(1))
    assert r == 0.0


def test_interdependent_mismatched_n() -> None:
    g_a = nx.path_graph(3)
    g_b = nx.path_graph(2)  # 0–1 only; three nodes in A, two in B
    out = interdependent_percolation(g_a, g_b, 0.0, np.random.default_rng(0))
    assert out == 0.0


def test_interdependent_p0_on_complete_clique() -> None:
    n = 12
    g_a = nx.complete_graph(n)
    g_b = nx.complete_graph(n)
    f = interdependent_percolation(g_a, g_b, 0.0, np.random.default_rng(42))
    assert abs(f - 1.0) < 1e-12
