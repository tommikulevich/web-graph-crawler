import powerlaw
import networkx as nx
from collections import Counter
from itertools import combinations
from typing import Any, Optional, Set, List, Tuple, Dict

# Helpers

def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
    """Construct a directed graph from a list of (source, target) edges."""
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    return G

def _prepare_undirected_subgraph(G: nx.DiGraph) -> nx.Graph:
    """Return undirected subgraph of the largest connected component."""
    
    U = G.to_undirected()
    if not nx.is_connected(U):
        largest_cc = max(nx.connected_components(U), key=len)
        U = U.subgraph(largest_cc).copy()
        
    return U

# [MS] 2.3.1
   
def get_number_of_nodes(G: nx.DiGraph) -> int:
    """Return the number of nodes in the graph."""
    
    return G.number_of_nodes()

def get_number_of_edges(G: nx.DiGraph) -> int:
    """Return the number of edges in the graph."""
    
    return G.number_of_edges()

# [MS] 2.3.2

def get_weakly_connected_components(G: nx.DiGraph) -> List[Set[Any]]:
    """Return list of weakly connected components."""
    
    return list(nx.weakly_connected_components(G))

def get_strongly_connected_components(G: nx.DiGraph) -> List[Set[Any]]:
    """Return list of strongly connected components."""
    
    return list(nx.strongly_connected_components(G))

def get_largest_scc(G: nx.DiGraph) -> Set[Any]:
    """Return the largest strongly connected component."""
    
    sccs = get_strongly_connected_components(G)
    return max(sccs, key=len) if sccs else set()

def get_in_component(G: nx.DiGraph) -> Set[Any]:
    """Return nodes that can reach the core (excluding core)."""
    
    core = get_largest_scc(G)
    in_set = set()
    for node in core:
        in_set |= nx.ancestors(G, node)
    return in_set - core

def get_out_component(G: nx.DiGraph) -> Set[Any]:
    """Return nodes reachable from the core (excluding core)."""
    
    core = get_largest_scc(G)
    out_set = set()
    for node in core:
        out_set |= nx.descendants(G, node)
    return out_set - core

# [MS] 2.3.3

def get_in_degrees(G: nx.DiGraph) -> List[int]:
    """Return list of in-degrees for all nodes."""
    
    return [d for _, d in G.in_degree()]

def get_out_degrees(G: nx.DiGraph) -> List[int]:
    """Return list of out-degrees for all nodes."""
    
    return [d for _, d in G.out_degree()]

def fit_in_degree_powerlaw(G: nx.DiGraph) -> Optional[Dict[str, float]]:
    """Fit in-degree distribution to power-law, return parameters."""
    
    indeg = get_in_degrees(G)
    try:
        fit = powerlaw.Fit(indeg, discrete=True)
        return {'alpha': fit.power_law.alpha, 'xmin': fit.power_law.xmin}
    except Exception:
        return None

def fit_out_degree_powerlaw(G: nx.DiGraph) -> Optional[Dict[str, float]]:
    """Fit out-degree distribution to power-law, return parameters."""
    
    outdeg = get_out_degrees(G)
    try:
        fit = powerlaw.Fit(outdeg, discrete=True)
        return {'alpha': fit.power_law.alpha, 'xmin': fit.power_law.xmin}
    except Exception:
        return None

# [MS] 2.3.4

def get_average_distance(G: nx.DiGraph) -> Optional[float]:
    """Return average shortest path length (largest CC)."""
    
    U = _prepare_undirected_subgraph(G)
    try:
        return nx.average_shortest_path_length(U)
    except Exception:
        return None

def get_diameter(G: nx.DiGraph) -> Optional[int]:
    """Return diameter (largest CC)."""
    
    U = _prepare_undirected_subgraph(G)
    try:
        return nx.diameter(U)
    except Exception:
        return None

def get_eccentricity(G: nx.DiGraph) -> Dict[Any, int]:
    """Return eccentricity per node (largest CC)."""
    
    U = _prepare_undirected_subgraph(G)
    try:
        return nx.eccentricity(U)
    except Exception:
        return {}

def get_distance_histogram(G: nx.DiGraph) -> Dict[int, int]:
    """Return histogram of shortest path lengths (largest CC)."""
    
    U = _prepare_undirected_subgraph(G)
    lengths = []
    for node in U.nodes():
        lengths.extend(dict(nx.shortest_path_length(U, node)).values())
    return dict(Counter(lengths))
    
def get_average_distance_per_node(G: nx.DiGraph) -> Dict[Any, float]:
    """Return average shortest path length per node (largest CC)."""
    
    U = _prepare_undirected_subgraph(G)
    avg_dist = {}
    for node in U.nodes():
        lengths = list(dict(nx.shortest_path_length(U, node)).values())
        if lengths:
            avg_dist[node] = sum(lengths) / len(lengths)
            
    return avg_dist

# [MS] 2.3.5
    
def get_local_clustering(G: nx.DiGraph) -> Dict[Any, float]:
    """Return local clustering coefficient per node."""
    
    UG = G.to_undirected()
    return nx.clustering(UG)

def get_global_clustering(G: nx.DiGraph) -> float:
    """Return global clustering coefficient (transitivity)."""
    
    UG = G.to_undirected()
    return nx.transitivity(UG)

def get_local_clustering_histogram(G: nx.DiGraph) -> Dict[float, int]:
    """Return histogram of local clustering values."""
    
    local = get_local_clustering(G)
    return dict(Counter(local.values()))

# [MS] 2.3.7

def get_articulation_points(G: nx.DiGraph) -> List[Any]:
    """Return list of articulation points in undirected graph."""
    
    UG = G.to_undirected()
    return list(nx.articulation_points(UG))

def get_node_connectivity(G: nx.DiGraph) -> Optional[int]:
    """Return node connectivity (min nodes to disconnect graph)."""
    
    UG = G.to_undirected()
    try:
        return nx.node_connectivity(UG)
    except Exception:
        return None

def get_articulation_pairs(G: nx.DiGraph, max_checks: int = 1000) -> List[Tuple[Any, Any]]:
    """Return list of node pairs whose removal disconnects graph."""
    
    UG = G.to_undirected()
    node_con = get_node_connectivity(G)
    sep_pairs = []
    if node_con and node_con >= 2:
        count = 0
        nodes = list(UG.nodes())
        for u, v in combinations(nodes, 2):
            if count >= max_checks:
                break
            H = UG.copy()
            H.remove_nodes_from([u, v])
            if not nx.is_connected(H):
                sep_pairs.append((u, v))
            count += 1
            
    return sep_pairs
