import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Dict

# [MS] 2.2

def plot_performance(perf_df: pd.DataFrame, img_dir: str) -> None:
    plt.figure()
    sns.lineplot(data=perf_df, x='threads', y='elapsed', marker='o')
    plt.title('Crawl Time vs Threads')
    plt.savefig(os.path.join(img_dir, 'performance.png'))
    plt.close()

# [MS] 2.3.2

def plot_component_sizes(comp_records: list, img_dir: str) -> None:
    df = pd.DataFrame(comp_records)
    plt.figure()
    sns.histplot(data=df, x='size', hue='type', log_scale=(True, True))
    plt.title('Component Size Distribution')
    plt.savefig(os.path.join(img_dir, 'component_sizes.png'))
    plt.close()

# [MS] 2.3.3

def plot_degree_distribution(in_degrees: list, out_degrees: list, img_dir: str) -> None:
    plt.figure()
    sns.histplot(in_degrees, log_scale=(True, True), color='blue', label='in')
    sns.histplot(out_degrees, log_scale=(True, True), color='orange', label='out')
    plt.legend(); plt.title('Degree Distributions (log-log)')
    plt.savefig(os.path.join(img_dir, 'degree_distribution.png'))
    plt.close()
    
# [MS] 2.3.4    

def plot_distance_histogram(distance_hist: Dict[int, int], img_dir: str) -> None:
    xs, ys = zip(*sorted(distance_hist.items()))
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Distance Histogram')
    plt.savefig(os.path.join(img_dir, 'distance_histogram.png'))
    plt.close()
    
def plot_average_distance_histogram(avg_distances: Dict[Any, float], img_dir: str) -> None:
    plt.figure()
    sns.histplot(x=list(avg_distances.values()))
    plt.title('Average Distance per Node Histogram')
    plt.savefig(os.path.join(img_dir, 'average_distance_histogram.png'))
    plt.close()

# [MS] 2.3.5

def plot_clustering_histogram(local_hist: Dict[float, int], img_dir: str) -> None:
    xs, ys = zip(*sorted(local_hist.items()))
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Count')
    plt.title('Clustering Coefficient Histogram')
    plt.savefig(os.path.join(img_dir, 'clustering_histogram.png'))
    plt.close()

# [MS] 2.3.7

def plot_eccentricity_histogram(ecc: Dict[Any, int], img_dir: str) -> None:
    plt.figure()
    sns.histplot(x=list(ecc.values()))
    plt.title('Eccentricity Histogram')
    plt.savefig(os.path.join(img_dir, 'eccentricity_histogram.png'))
    plt.close()

# [MS] 2.4

def plot_pagerank_time(pr_df: pd.DataFrame, img_dir: str) -> None:
    plt.figure()
    sns.lineplot(data=pr_df, x='damping', y='time', marker='o')
    plt.title('PageRank Convergence Time vs Damping')
    plt.savefig(os.path.join(img_dir, 'pagerank_time.png'))
    plt.close()

# TODO: zbieżność PageRanku
def plot_pagerank_distribution(ranks: Dict[Any, float], damping: float, img_dir: str) -> None:
    values = list(ranks.values())
    plt.figure()
    sns.histplot(values, log_scale=(True, True), bins=50)
    plt.title(f'PageRank Value Distribution (d={damping})')
    plt.savefig(os.path.join(img_dir, f'pagerank_dist_{str(damping).replace(".", "_")}.png'))
    plt.close()
    