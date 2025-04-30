import yaml
from dataclasses import dataclass


@dataclass
class CrawlerConfig:
    base_url: str
    max_pages: int
    threads_num: int
    timeout_s: float
    user_agent: str
    storage_path: str

@dataclass
class AnalysisConfig:
    reports_path: str

@dataclass
class PageRankConfig:
    damping_factor: float
    max_iterations: int
    tol: float

@dataclass
class Config:
    crawler: CrawlerConfig
    analysis: AnalysisConfig
    pagerank: PageRankConfig


def load_config(path: str) -> Config:
    """Load YAML configuration and return Config object."""
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    crawler_data = data.get('crawler', {})
    crawler_cfg = CrawlerConfig(
        base_url=crawler_data['base_url'],
        max_pages=int(crawler_data.get('max_pages', 0)),
        threads_num=int(crawler_data.get('threads_num', 1)),
        user_agent=crawler_data.get('user_agent', ''),
        storage_path=crawler_data.get('storage_path', ''),
        timeout_s=float(crawler_data.get('timeout_s', 10.0)),
    )
    
    analysis_data = data.get('analysis', {})
    analysis_cfg = AnalysisConfig(
        reports_path=analysis_data.get('reports_path', ''),
    )
    
    pagerank_data = data.get('pagerank', {})
    pagerank_cfg = PageRankConfig(
        damping_factor=float(pagerank_data.get('damping_factor', 0.85)),
        max_iterations=int(pagerank_data.get('max_iterations', 100)),
        tol=float(pagerank_data.get('tol', 1e-6)),
    )
    
    config = Config(
        crawler=crawler_cfg,
        analysis=analysis_cfg,
        pagerank=pagerank_cfg,
    )

    return config
