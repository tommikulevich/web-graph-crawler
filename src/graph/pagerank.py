import networkx as nx
from typing import Dict

# [MS] 2.4

def pagerank(G: nx.DiGraph, damping_factor: float, max_iterations: int,
             tol: float) -> Dict[str, float]:
    """Compute PageRank for each node."""
    
    N = G.number_of_nodes()
    if N == 0:
        return {}
    nodes = list(G.nodes())
    
    # Initialize rank to uniform distribution
    rank = {n: 1.0 / N for n in nodes}
    
    # Precompute incoming links
    inbound = {n: list(G.predecessors(n)) for n in nodes}
    
    # Dangling nodes
    dangling = [n for n in nodes if G.out_degree(n) == 0]
    for _ in range(max_iterations):
        new_rank = {}
        
        # Sum of ranks of dangling nodes
        dangling_sum = sum(rank[n] for n in dangling)
        for n in nodes:
            rank_sum = 0.0
            for m in inbound.get(n, []):
                out_deg = G.out_degree(m)
                if out_deg > 0:
                    rank_sum += rank[m] / out_deg
                    
            # Teleportation and damping
            new_rank[n] = ((1.0 - damping_factor) / N
                           + damping_factor * (rank_sum + dangling_sum / N))
            
        # Check convergence
        diff = sum(abs(new_rank[n] - rank[n]) for n in nodes)
        rank = new_rank
        if diff < tol:
            break
        
    return rank
