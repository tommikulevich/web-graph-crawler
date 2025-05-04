import os
import csv
import time
import argparse
import pandas as pd
from typing import List, Optional

from common.config import load_config, Config
from common.logger import get_logger, init_file_logger
from graph.analysis import build_directed_graph
from graph.pagerank import pagerank
import experiments.plotting_helper as ph


def run_pagerank_experiment(cfg: Config, damping_factors: Optional[List[float]] = None,
                            max_iterations: int = 300) -> pd.DataFrame:
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
            
    G = build_directed_graph(edges)

    # [MS] 2.3.4
    
    if damping_factors is None:
        damping_factors = [0.65, 0.75, 0.85, 0.95, 1]

    records = []
    convergences = {}
    dist_ranks_085 = {}
    for d in damping_factors:
        start = time.perf_counter()
        ranks, diffs = pagerank(
            G,
            damping_factor=d,
            max_iterations=max_iterations,
            tol=cfg.pagerank.tol,
        )
        elapsed = time.perf_counter() - start

        top10 = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"PageRank beta={d}: time={elapsed:.3f}s, top10={top10}")

        convergences[d] = diffs

        if abs(d - 0.85) < 1e-6:
            dist_ranks_085 = ranks

        records.append({
            "damping": d,
            "time": elapsed,
            "iterations": len(diffs),
            "final_diff": diffs[-1] if diffs else None,
            "top10": top10,
        })
        
    ph.plot_all_pagerank_convergence(
        convergences,
        os.path.join(img_dir, "pagerank_all_convergence.png")
    )
    logger.info("Saved combined convergence plot.")

    if dist_ranks_085:
        ph.plot_pagerank_histogram_and_powerlaw(
            dist_ranks_085,
            0.85,
            os.path.join(img_dir, "pagerank_histogram_powerlaw_0_85.png")
        )
        logger.info("Saved histogram + power-law fit for beta=0.85.")
        
    df = pd.DataFrame(records)
    
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
