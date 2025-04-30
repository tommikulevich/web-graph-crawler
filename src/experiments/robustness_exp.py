import os
import csv
import random
import argparse
import pandas as pd
from typing import List, Optional

from common.config import load_config, Config
from common.logger import get_logger, init_file_logger
import graph.analysis as ga
import experiments.plotting_helper as ph
from experiments.analysis_exp import run_analysis_experiment


def run_robustness_experiment(cfg: Config, 
                              removal_fractions: Optional[List[float]] = None) -> pd.DataFrame:
    """Execute robustness experiments by running full analysis on perturbed graphs."""
    
    log_file = os.path.join(cfg.analysis.reports_path, 'logger.log')
    init_file_logger(log_file)
    logger = get_logger(__name__)
    
    edges_file = os.path.join(cfg.crawler.storage_path, 'edges.csv')
    edges = []
    with open(edges_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row['source'], row['target']))
            
    G = ga.build_graph(edges)
    
    # [MS] 2.3.6
    
    if removal_fractions is None:
        removal_fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    records = []
    for frac in removal_fractions:
        for targeted in (False, True):
            logger.info(f"Running robustness: fraction={frac}, targeted={targeted}")
            
            total = ga.get_number_of_nodes(G)
            n_remove = int(frac * total)
            if n_remove > 0:
                nodes = list(G.nodes())
                if targeted:
                    nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
                    to_remove = nodes_sorted[:n_remove]
                else:
                    to_remove = random.sample(nodes, min(n_remove, len(nodes)))
            else:
                to_remove = []
                
            H = G.copy()
            H.remove_nodes_from(to_remove)
            
            subdir = f"robust_frac_{frac}_{'targeted' if targeted else 'random'}"
            summary = run_analysis_experiment(cfg, G=H, report_subdir=subdir)
            summary['fraction'] = frac
            summary['targeted'] = targeted
            summary['remaining_nodes'] = summary.get('nodes')
            summary['remaining_edges'] = summary.get('edges')
            records.append(summary)
            
            logger.info(f"Result: {summary}")
            
    df = pd.DataFrame(records)
    
    img_dir = os.path.join(cfg.analysis.reports_path, 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    return df

if __name__ == '__main__':
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/config.yaml')
    parser.add_argument('-o', '--output', default='robustness.csv')
    parser.add_argument('-r', '--fractions', nargs='*', type=float, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    out_csv = args.output
    fractions = args.fractions
    
    df = run_robustness_experiment(cfg, fractions)
    df.to_csv(out_csv, index=False)
    
    logger.info(f"Saved robustness results to {out_csv}")
