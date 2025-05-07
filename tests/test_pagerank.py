import os
import sys
import random
import pytest
import networkx as nx

try:
    from graph.pagerank import pagerank
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'src')))
    from graph.pagerank import pagerank


def _unpack_pr(result):
    """Unpack pagerank result into (rank, diffs)."""
    if isinstance(result, dict):
        return result, []
    return result


def test_empty_graph() -> None:
    """Pagerank returns empty dict for empty graph."""
    G = nx.DiGraph()
    result = pagerank(G, damping_factor=0.85,
                      max_iterations=100, tol=1e-6)

    assert isinstance(result, dict)
    assert result == {}


def test_single_node_graph() -> None:
    """Pagerank on single node graph returns rank 1."""
    G = nx.DiGraph()
    G.add_node('A')
    rank, diffs = _unpack_pr(pagerank(
        G, damping_factor=0.85, max_iterations=100, tol=1e-8))

    assert rank == {'A': pytest.approx(1.0)}
    assert len(diffs) == 1
    assert diffs[0] == pytest.approx(0.0)


def test_two_node_one_iteration() -> None:
    """Pagerank with max_iterations=1 yields expected ranks."""
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    damping = 0.85
    rank, diffs = _unpack_pr(pagerank(
        G, damping_factor=damping, max_iterations=1, tol=0.0))
    exp_A = (1 - damping) / 2 + damping * (0 + 0.5 / 2)
    exp_B = (1 - damping) / 2 + damping * (0.5 + 0.5 / 2)

    assert rank['A'] == pytest.approx(exp_A)
    assert rank['B'] == pytest.approx(exp_B)
    assert len(diffs) == 1


def test_two_node_cycle_uniform() -> None:
    """Pagerank on 2-node cycle yields uniform distribution."""
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'A')
    rank, _ = _unpack_pr(pagerank(
        G, damping_factor=0.85, max_iterations=100, tol=1e-8))

    assert rank['A'] == pytest.approx(0.5)
    assert rank['B'] == pytest.approx(0.5)
    assert sum(rank.values()) == pytest.approx(1.0)


def test_damping_zero_uniform() -> None:
    """Pagerank with zero damping yields uniform distribution."""
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 1)
    G.add_node(3)
    rank, _ = _unpack_pr(pagerank(
        G, damping_factor=0.0, max_iterations=10, tol=0.0))
    n = len(G)

    for val in rank.values():
        assert val == pytest.approx(1.0 / n)


def test_complete_graph_uniform() -> None:
    """Pagerank on complete graph yields uniform distribution."""
    n = 5
    G = nx.DiGraph()
    for u in range(n):
        for v in range(n):
            if u != v:
                G.add_edge(u, v)
    rank, _ = _unpack_pr(pagerank(
        G, damping_factor=1.0, max_iterations=100, tol=1e-8))

    for val in rank.values():
        assert val == pytest.approx(1.0 / n)
    assert sum(rank.values()) == pytest.approx(1.0)


def test_all_dangling_nodes() -> None:
    """Pagerank on graph with dangling nodes yields uniform distribution."""
    n = 5
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    rank, diffs = _unpack_pr(pagerank(
        G, damping_factor=0.85, max_iterations=10, tol=1e-9))

    for val in rank.values():
        assert val == pytest.approx(1.0 / n)
    assert len(diffs) == 1
    assert diffs[0] == pytest.approx(0.0)


def test_random_graph_convergence() -> None:
    """Pagerank on random graph converges to valid distribution."""
    random.seed(42)
    G = nx.gnp_random_graph(10, 0.3, directed=True, seed=42)
    rank, diffs = _unpack_pr(pagerank(
        G, damping_factor=0.85, max_iterations=100, tol=1e-6))

    assert sum(rank.values()) == pytest.approx(1.0, rel=1e-6)
    assert all(val >= 0 for val in rank.values())
    assert all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1))
