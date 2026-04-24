from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from site_percolation import (
    generate_network,
    graph_to_adjacency_list,
    interdependent_percolation,
)


def test_interdependent_p0_py_matches_cpp() -> None:
    cascade_sim = pytest.importorskip("cascade_sim", reason="build with: pip install -e '.[dev]'")
    n = 12
    g_a = nx.complete_graph(n)
    g_b = nx.complete_graph(n)
    adj_a = graph_to_adjacency_list(g_a, n)
    adj_b = graph_to_adjacency_list(g_b, n)
    py_f = interdependent_percolation(g_a, g_b, 0.0, np.random.default_rng(123))
    cpp_f = cascade_sim.interdependent_percolation(adj_a, adj_b, 0.0, 123)
    assert abs(py_f - cpp_f) < 1e-12
    assert abs(cpp_f - 1.0) < 1e-12


def test_interdependent_p1_zero_cpp() -> None:
    cascade_sim = pytest.importorskip("cascade_sim", reason="build with: pip install -e '.[dev]'")
    n = 40
    g_a = generate_network(n, 4.0, seed=1)
    g_b = generate_network(n, 4.0, seed=2)
    a = graph_to_adjacency_list(g_a, n)
    b = graph_to_adjacency_list(g_b, n)
    out = cascade_sim.interdependent_percolation(a, b, 1.0, 9_999_999_999_999_999_999)
    assert out == 0.0


def test_mismatched_adjacency_size_returns_zero() -> None:
    cascade_sim = pytest.importorskip("cascade_sim", reason="build with: pip install -e '.[dev]'")
    a = graph_to_adjacency_list(nx.path_graph(3), 3)
    b = [[1], [0]]  # length 2 vs n=3
    out = cascade_sim.interdependent_percolation(a, b, 0.0, 1)
    assert out == 0.0
