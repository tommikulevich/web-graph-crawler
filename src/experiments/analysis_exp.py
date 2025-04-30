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
        G = ga.build_graph(edges)
    else:
        os.makedirs(data_dir, exist_ok=True)
        edges = list(G.edges())
        pd.DataFrame(edges, columns=['source', 'target']).to_csv(
            os.path.join(data_dir, 'edges.csv'), index=False
        )
        
    # [MS] 2.3.1-2.3.5, 2.3.7

    num_nodes = ga.get_number_of_nodes(G)
    num_edges = ga.get_number_of_edges(G)
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

    logger.info("Computing component structure...")
    wccs = ga.get_weakly_connected_components(G)
    sccs = ga.get_strongly_connected_components(G)
    core = ga.get_largest_scc(G)
    in_comp = ga.get_in_component(G)
    out_comp = ga.get_out_component(G)

    logger.info("Computing degree distributions...")
    in_degs = ga.get_in_degrees(G)
    out_degs = ga.get_out_degrees(G)
    in_fit = ga.fit_in_degree_powerlaw(G)
    out_fit = ga.fit_out_degree_powerlaw(G)

    logger.info("Computing distance metrics...")
    avg_dist = ga.get_average_distance(G)
    diam = ga.get_diameter(G)
    ecc = ga.get_eccentricity(G)
    dist_hist = ga.get_distance_histogram(G)

    logger.info("Computing clustering metrics...")
    glob_clust = ga.get_global_clustering(G)
    local_hist = ga.get_local_clustering_histogram(G)

    logger.info("Computing connectivity metrics...")
    art_points = ga.get_articulation_points(G)
    node_con = ga.get_node_connectivity(G)
    art_pairs = ga.get_articulation_pairs(G)

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
    pd.DataFrame({'in_degree': in_degs}).to_csv(
        os.path.join(data_dir, 'in_degrees.csv'), index=False)
    pd.DataFrame({'out_degree': out_degs}).to_csv(
        os.path.join(data_dir, 'out_degrees.csv'), index=False)
    pd.DataFrame(list(dist_hist.items()), columns=['distance', 'count']).to_csv(
        os.path.join(data_dir, 'distance_histogram.csv'), index=False)
    pd.DataFrame({'clustering': list(local_hist.keys()),
                  'count': list(local_hist.values())}).to_csv(
                            os.path.join(data_dir, 'clustering_histogram.csv'), 
                                        index=False)
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
    pd.DataFrame(list(ecc.items()), columns=['node', 'eccentricity']).to_csv(
        os.path.join(data_dir, 'eccentricity.csv'), index=False)
    pd.DataFrame(art_pairs, columns=['node1', 'node2']).to_csv(
        os.path.join(data_dir, 'articulation_pairs.csv'), index=False)
    pd.DataFrame({'articulation_point': art_points}).to_csv(
        os.path.join(data_dir, 'articulation_points.csv'), index=False)

    logger.info(f"Metrics summary: {summary}")

    logger.info("Generating plots for graph analysis...")
    ph.plot_component_sizes(comp_records, img_dir)
    ph.plot_degree_distribution(in_degs, out_degs, img_dir)
    ph.plot_distance_histogram(dist_hist, img_dir)
    ph.plot_clustering_histogram(local_hist, img_dir)
    ph.plot_average_distance_histogram(ga.get_average_distance_per_node(G), img_dir)
    ph.plot_eccentricity_histogram(ecc, img_dir)
    
    return summary
