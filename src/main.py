
import os
import argparse
from typing import Dict

from common.config import load_config
from common.logger import get_logger, init_file_logger
from crawler.crawler import Crawler
from experiments.analysis_exp import run_analysis_experiment
from experiments.performance_exp import run_performance_experiment
from experiments.robustness_exp import run_robustness_experiment
from experiments.pagerank_exp import run_pagerank_experiment


def _create_dirs(base_dir: str) -> Dict[str, str]:
    """Create base experiment directory and subfolders for images and data."""
    
    os.makedirs(base_dir, exist_ok=True)
    
    img_dir = os.path.join(base_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    return {'base': base_dir, 'img': img_dir, 'data': data_dir}

def cmd_crawl(args):
    """Run the crawling process based on configuration."""
    
    cfg = load_config(args.config)
    
    log_file = os.path.join(cfg.analysis.reports_path, 'logger.log')
    init_file_logger(log_file)
    logger = get_logger(__name__)
    
    logger.info(f"Run crawler")
    crawler = Crawler(
        base_url=cfg.crawler.base_url,
        max_pages=cfg.crawler.max_pages,
        threads_num=cfg.crawler.threads_num,
        user_agent=cfg.crawler.user_agent,
        storage_path=cfg.crawler.storage_path,
        timeout_s=cfg.crawler.timeout_s,
    )
    crawler.start()

def cmd_analyze(args):
    """Run full analysis."""
    
    cfg = load_config(args.config)
    report_dir = cfg.analysis.reports_path

    base_dir = report_dir
    dirs = _create_dirs(base_dir)
    data_dir = dirs['data']
    
    log_file = os.path.join(cfg.analysis.reports_path, 'logger.log')
    init_file_logger(log_file)
    logger = get_logger(__name__)

    # --- Experiments ---
    
    run_analysis_experiment(cfg)
    logger.info(f"End of analysis experiment")
    
    perf_df = run_performance_experiment(cfg)
    perf_csv = os.path.join(data_dir, 'performance.csv')
    perf_df.to_csv(perf_csv, index=False)
    logger.info(f"Saved performance results to {perf_csv}")

    rob_df = run_robustness_experiment(cfg)
    rob_csv = os.path.join(data_dir, 'robustness.csv')
    rob_df.to_csv(rob_csv, index=False)
    logger.info(f"Saved robustness results to {rob_csv}")

    pr_df = run_pagerank_experiment(cfg)
    pr_csv = os.path.join(data_dir, 'pagerank.csv')
    pr_df.to_csv(pr_csv, index=False)
    logger.info(f"Saved pagerank results to {pr_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Web crawler and graph analysis CLI'
    )
    parser.add_argument('-c', '--config', default='configs/config.yaml',
                        help='Path to configuration file')

    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('crawl', help='Run crawler').set_defaults(func=cmd_crawl)
    subparsers.add_parser('analyze', help='Run full analysis and generate report').set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)
