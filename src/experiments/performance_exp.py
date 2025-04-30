import os
import time
import argparse
import pandas as pd
from typing import List, Optional

from common.logger import get_logger, init_file_logger
from common.config import load_config, Config
from crawler.scheduler import Scheduler
import experiments.plotting_helper as ph


def run_performance_experiment(cfg: Config, thread_list: Optional[List[int]] = None,
                               max_pages: Optional[int] = None) -> pd.DataFrame:
    """Execute performance experiment varying number of threads."""
    
    log_file = os.path.join(cfg.analysis.reports_path, 'logger.log')
    init_file_logger(log_file)
    logger = get_logger(__name__)
    
    orig_storage = cfg.crawler.storage_path

    # [MS] 2.2
    
    if thread_list is None:
        thread_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
        
    results = []
    for t in thread_list:
        # Override thread count and optional max_pages
        cfg.crawler.threads_num = t
        if max_pages is not None:
            cfg.crawler.max_pages = max_pages
        
        # Set storage path for this thread count to avoid overwriting
        cfg.crawler.storage_path = os.path.join(
            os.path.dirname(orig_storage),
            'temp_performance',
            str(t),
            os.path.basename(orig_storage)
        )
        
        logger.info(f"Performance test: threads={t}, storage={cfg.crawler.storage_path}")
        
        start = time.perf_counter()
        sched = Scheduler(
            base_url=cfg.crawler.base_url,
            max_pages=cfg.crawler.max_pages,
            threads_num=cfg.crawler.threads_num,
            user_agent=cfg.crawler.user_agent,
            storage_path=cfg.crawler.storage_path,
            timeout_s=cfg.crawler.timeout_s,
        )
        sched.start()
        
        elapsed = time.perf_counter() - start
        results.append({'threads': t, 'elapsed': elapsed})
        
    df = pd.DataFrame(results)

    img_dir = os.path.join(cfg.analysis.reports_path, 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    ph.plot_performance(df, img_dir)
    logger.info(f"Generated performance plot in {img_dir}")
    
    return df

if __name__ == '__main__':
    logger = get_logger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/config.yaml')
    parser.add_argument('-o', '--output', default='performance.csv')
    parser.add_argument('-m', '--max-pages', type=int, default=None)
    parser.add_argument('-t', '--threads', nargs='*', type=int, default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    out_csv = args.output
    max_pages = args.max_pages
    threads_list = args.threads
    
    df = run_performance_experiment(cfg, threads_list, max_pages)
    df.to_csv(out_csv, index=False)
    
    logger.info(f"Saved performance results to {out_csv}")
