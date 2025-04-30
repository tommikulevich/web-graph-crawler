import os
import csv
import time
import argparse
import pandas as pd
from typing import List, Optional

from common.config import load_config, Config
from common.logger import get_logger, init_file_logger
from graph.analysis import build_graph
from graph.pagerank import pagerank
import experiments.plotting_helper as ph


def run_pagerank_experiment(cfg: Config, damping_factors: Optional[List[float]] = None) -> pd.DataFrame:
    """Execute PageRank experiments for given damping factors."""
    
    log_file = os.path.join(cfg.analysis.reports_path, 'logger.log')
    init_file_logger(log_file)
    logger = get_logger(__name__)

    edges_file = os.path.join(cfg.crawler.storage_path, 'edges.csv')
    edges = []
    with open(edges_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row['source'], row['target']))
            
    img_dir = os.path.join(cfg.analysis.reports_path, 'img')
    os.makedirs(img_dir, exist_ok=True)
            
    G = build_graph(edges)

    # [MS] 2.3.4
    
    if damping_factors is None:
        damping_factors = [0.3, 0.5, 0.7, 0.85, 0.95, 0.99]

    records = []
    for d in damping_factors:
        start = time.perf_counter()
        ranks = pagerank(
            G,
            damping_factor=d,
            max_iterations=cfg.pagerank.max_iterations,
            tol=cfg.pagerank.tol,
        )
        elapsed = time.perf_counter() - start

        top5 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"PageRank d={d}: time={elapsed:.4f}s, top5={[u for u,_ in top5]}")

        ph.plot_pagerank_distribution(ranks, d, img_dir)
        logger.info(f"Generated PageRank distribution plot in {img_dir}")

        records.append({
            'damping': d,
            'time': elapsed,
            'top5': top5,
        })
        
    df = pd.DataFrame(records)

    ph.plot_pagerank_time(df, img_dir)
    logger.info(f"Generated PageRank time plot in {img_dir}")
    
    return df

if __name__ == '__main__':
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/config.yaml')
    parser.add_argument('-o', '--output', default='pagerank.csv')
    parser.add_argument('-d', '--damping', nargs='*', type=float, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    out_csv = args.output
    damping_factors = args.damping
    
    df = run_pagerank_experiment(cfg, damping_factors)
    df.to_csv(out_csv, index=False)
    
    logger.info(f"Saved PageRank results to {out_csv}")
