import os
import pandas as pd
from pyvis.network import Network

from common.config import Config
from common.logger import get_logger
import graph.analysis as ga
import experiments.plotting_helper as ph


def run_analysis_experiment(cfg: Config, G=None, report_subdir: str = None) -> dict:
    """Perform full graph analysis on given graph or edges, save CSVs and generate plots, return summary."""

    base_report_dir = cfg.analysis.reports_path
    report_dir = base_report_dir if report_subdir is None else os.path.join(base_report_dir, report_subdir)
    logger = get_logger(__name__)

    img_dir = os.path.join(report_dir, 'img')
    data_dir = os.path.join(report_dir, 'data')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if G is None:
        edges_file = os.path.join(cfg.crawler.storage_path, 'edges.csv')
        edges_df = pd.read_csv(edges_file)
        os.makedirs(data_dir, exist_ok=True)
        edges_df.to_csv(os.path.join(data_dir, 'edges.csv'), index=False)
        edges = list(edges_df.itertuples(index=False, name=None))
        G = ga.build_directed_graph(edges)
    else:
        os.makedirs(data_dir, exist_ok=True)
        edges = list(G.edges())
        pd.DataFrame(edges, columns=['source', 'target']).to_csv(
            os.path.join(data_dir, 'edges.csv'), index=False
        )
        
    # [MS] 2.3.1-2.3.5, 2.3.7

    num_nodes, num_edges = ga.get_number_of_nodes(G), ga.get_number_of_edges(G)
    logger.info(f"Graph: {num_nodes} nodes, {num_edges} edges.")

    try:
        net = Network(height='800px', width='100%', directed=True)
        net.toggle_physics(False)
        net.from_nx(G)
        html_path = os.path.join(data_dir, 'graph.html')
        net.write_html(html_path)
        logger.info(f"Saved interactive graph to {html_path}")
    except Exception as e:
        logger.warning(f"Failed to generate interactive graph: {e}")

    wccs = ga.get_weakly_connected_components(G)
    logger.info(f"Number of weakly connected components: {len(wccs)}")
    logger.info(f"Number of nodes in largest weakly connected component: {len(max(wccs, key=len))}")
    logger.info(f"Number of edges in largest weakly connected component: {len(G.subgraph(max(wccs, key=len)).edges())}")
    
    sccs = ga.get_strongly_connected_components(G)
    sorted_sccs = sorted(sccs, key=len, reverse=True)
    largest_scc = sorted_sccs[0] if sorted_sccs else set()
    second_largest_scc = sorted_sccs[1] if len(sorted_sccs) > 1 else set()
    third_largest_scc = sorted_sccs[2] if len(sorted_sccs) > 2 else set()
    logger.info(f"Number of strongly connected components: {len(sccs)}")
    logger.info(f"Number of nodes in largest SCC: {len(largest_scc)}")
    logger.info(f"Number of edges in largest SCC: {len(G.subgraph(largest_scc).edges())}")
    logger.info(f"Number of nodes in 2nd largest SCC: {len(second_largest_scc)}")
    logger.info(f"Number of edges in 2nd largest SCC: {len(G.subgraph(second_largest_scc).edges())}")
    logger.info(f"Number of nodes in 3rd largest SCC: {len(third_largest_scc)}")
    logger.info(f"Number of edges in 3rd largest SCC: {len(G.subgraph(third_largest_scc).edges())}")
    core = ga.get_largest_scc(G)
    
    in_comp = ga.get_in_component(G)
    logger.info(f"Number of nodes in in-component: {len(in_comp)}")
    logger.info(f"Number of edges in in-component: {len(G.subgraph(in_comp).edges())}")

    out_comp = ga.get_out_component(G)
    logger.info(f"Number of nodes in out-component: {len(out_comp)}")
    logger.info(f"Number of edges in out-component: {len(G.subgraph(out_comp).edges())}")
    
    comp_records = []
    comp_sizes = {
        'wcc': [len(c) for c in wccs],
        'scc': [len(c) for c in sccs],
        'in': [len(in_comp)],
        'out': [len(out_comp)],
        'core': [len(core)],
    }
    for comp_type, sizes in comp_sizes.items():
        for sz in sizes:
            comp_records.append({'type': comp_type, 'size': sz})
    pd.DataFrame(comp_records).to_csv(
        os.path.join(data_dir, 'components.csv'), index=False)

    in_degs = ga.get_in_degrees(G)
    logger.info(f"Average in-degree: {sum(in_degs) / len(in_degs)}")
    logger.info(f"Max in-degree: {max(in_degs)}")

    out_degs = ga.get_out_degrees(G)
    logger.info(f"Average out-degree: {sum(out_degs) / len(out_degs)}")
    logger.info(f"Max out-degree: {max(out_degs)}")

    in_fit = ga.fit_in_degree_powerlaw(G)
    logger.info(f"In-degree power-law fit: {in_fit}")
    
    out_fit = ga.fit_out_degree_powerlaw(G)
    logger.info(f"Out-degree power-law fit: {out_fit}")
    
    pd.DataFrame({'in_degree': in_degs}).to_csv(
        os.path.join(data_dir, 'in_degrees.csv'), index=False)
    pd.DataFrame({'out_degree': out_degs}).to_csv(
        os.path.join(data_dir, 'out_degrees.csv'), index=False)
    ph.plot_degree_distribution(in_degs, out_degs, img_dir)

    avg_dist = ga.get_average_distance(G)
    logger.info(f"Average distance: {avg_dist}")
    
    diam = ga.get_diameter(G)
    logger.info(f"Diameter: {diam}")
    
    radius = ga.get_radius(G)
    logger.info(f"Radius: {radius}")
    
    ecc = ga.get_eccentricity(G)
    dist_hist = ga.get_distance_histogram(G)
    
    pd.DataFrame(list(dist_hist.items()), columns=['distance', 'count']).to_csv(
        os.path.join(data_dir, 'distance_histogram.csv'), index=False)
    ph.plot_distance_histogram(dist_hist, img_dir)

    glob_clust = ga.get_global_clustering(G)
    local_hist = ga.get_local_clustering_histogram(G)
    logger.info(f"Global clustering coefficient: {glob_clust}")
    
    pd.DataFrame({'clustering': list(local_hist.keys()),
                  'count': list(local_hist.values())}).to_csv(
                            os.path.join(data_dir, 'clustering_histogram.csv'), 
                                        index=False)
    ph.plot_clustering_histogram(local_hist, img_dir)

    art_points = ga.get_articulation_points(G)
    logger.info(f"Articulation points: {art_points}")
    node_con = ga.get_node_connectivity(G)
    logger.info(f"Node connectivity: {node_con}")
    art_pairs = ga.get_articulation_pairs(G)
    logger.info(f"Articulation pairs: {art_pairs}")

    pd.DataFrame(list(ecc.items()), columns=['node', 'eccentricity']).to_csv(
        os.path.join(data_dir, 'eccentricity.csv'), index=False)
    pd.DataFrame(art_pairs, columns=['node1', 'node2']).to_csv(
        os.path.join(data_dir, 'articulation_pairs.csv'), index=False)
    pd.DataFrame({'articulation_point': art_points}).to_csv(
        os.path.join(data_dir, 'articulation_points.csv'), index=False)
    
    summary = {
        'nodes': num_nodes,
        'edges': num_edges,
        'wcc_count': len(wccs),
        'scc_count': len(sccs),
        'in_size': len(in_comp),
        'out_size': len(out_comp),
        'core_size': len(core),
        'articulation_points': len(art_points),
        'articulation_pairs': len(art_pairs),
        'average_distance': avg_dist,
        'diameter': diam,
        'global_clustering': glob_clust,
        'node_connectivity': node_con,
        'alpha_in': in_fit['alpha'] if in_fit else None,
        'xmin_in': in_fit['xmin'] if in_fit else None,
        'alpha_out': out_fit['alpha'] if out_fit else None,
        'xmin_out': out_fit['xmin'] if out_fit else None,
    }
    
    pd.DataFrame(list(summary.items()), columns=['metric', 'value']).to_csv(
        os.path.join(data_dir, 'metrics_summary.csv'), index=False)
    logger.info(f"Metrics summary: {summary}")
    
    return summary
