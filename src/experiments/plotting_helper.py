import os
import powerlaw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Dict, List

# [MS] 2.2

def plot_performance(perf_df: pd.DataFrame, img_dir: str) -> None:
    plt.figure()
    sns.lineplot(data=perf_df, x='threads', y='elapsed', marker='o')
    plt.xlabel('Threads')
    plt.ylabel('Time [s]')
    plt.savefig(os.path.join(img_dir, 'performance.png'))
    plt.close()

# [MS] 2.3.3

def plot_degree_distribution(in_degrees: List[int], out_degrees: List[int], img_dir: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for degrees, ax, title in (
        (in_degrees,  axes[0], 'In-Degree'),
        (out_degrees, axes[1], 'Out-Degree'),
    ):
        arr = np.array(degrees, dtype=int)
        arr = arr[arr > 0]
        if arr.size == 0:
            ax.set_title(f'{title} (no positive degrees)')
            continue

        fit = powerlaw.Fit(arr, discrete=True)

        max_k = arr.max()
        counts = np.bincount(arr, minlength=max_k+1)[1:]
        ks = np.arange(1, max_k+1)
        pmf = counts / counts.sum()

        mask = ks >= fit.xmin
        pmf_masked = counts[mask] / counts[mask].sum()

        ax.scatter(ks[~mask], pmf[~mask],
                   color='gray', alpha=0.6, label='k < xmin')
        ax.scatter(ks[mask], pmf_masked,
                   color='C0', marker='o', label='k ≥ xmin')

        x = np.arange(int(fit.xmin), max_k+1)
        y = fit.power_law.pdf(x)
        ax.plot(x, y, color='C3', linestyle='--',
                label=f'fit (alpha={fit.power_law.alpha:.2f})')

        ax.axvline(fit.xmin, color='black', linestyle=':', label=f'xmin = {int(fit.xmin)}')
        ax.fill_between(x, y, alpha=0.2, color='C2')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree k')
        ax.set_ylabel('P(k)')
        ax.set_title(f'{title} Distribution')
        ax.legend()

    plt.tight_layout()
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, 'degree_distribution.png'))
    plt.close()

# [MS] 2.3.4    

def plot_distance_histogram(distance_hist: Dict[int, int], img_dir: str) -> None:
    xs, ys = zip(*sorted(distance_hist.items()))
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel('Distance')
    plt.ylabel('Number of Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'distance_histogram.png'))
    plt.close()

def plot_distance_histogram_log(dist_hist: Dict[int,int], img_dir: str) -> None:
    ks = np.array(sorted([d for d in dist_hist.keys() if d > 0]), dtype=int)
    counts = np.array([dist_hist[d] for d in ks], dtype=float)
    pmf = counts / counts.sum()  # P(D=d)
    
    sample = np.repeat(ks, counts.astype(int))
    
    fit = powerlaw.Fit(sample, discrete=True)
    xmin = int(fit.xmin)
    alpha = fit.power_law.alpha
    
    plt.figure()
    
    mask_low = ks < xmin
    plt.scatter(ks[mask_low], pmf[mask_low],
                color='gray', alpha=0.6, label=f'd < xmin={xmin}')
    
    mask_high = ks >= xmin
    plt.scatter(ks[mask_high], pmf[mask_high],
                color='C0', alpha=0.8, label=f'd ≥ xmin={xmin}')
    
    x_fit = np.arange(xmin, ks.max()+1)
    y_fit = fit.power_law.pdf(x_fit)
    plt.plot(x_fit, y_fit, 'C3--',
             label=f'fit ($d^{{-{alpha:.2f}}}$)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance d')
    plt.ylabel('P(d)')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, 'distance_distribution.png'))
    plt.close()

def plot_eccentricity_histogram(ecc: Dict[Any, int], img_dir: str) -> None:
    plt.figure()
    sns.histplot(x=list(ecc.values()))
    plt.savefig(os.path.join(img_dir, 'eccentricity_histogram.png'))
    plt.close()

# [MS] 2.3.5

def plot_clustering_histogram(local_hist: Dict[float, int],
                              img_dir: str,
                              fname: str = 'clustering_histogram.png') -> None:
    all_coeffs = []
    for coeff, cnt in local_hist.items():
        all_coeffs.extend([coeff] * cnt)

    try:
        plt.style.use('seaborn-whitegrid')
    except (ValueError, OSError):
        plt.style.use('default')

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor']   = 'white'

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    bins = np.linspace(0.0, 1.0, 31)  # 30 bins from 0 to 1
    ax.hist(all_coeffs,
            bins=bins,
            color='blue',
            edgecolor='black',
            alpha=0.75)

    ax.set_xlabel('Clustering Coefficient', fontsize=12)
    ax.set_ylabel('Number of Nodes', fontsize=12)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    outpath = os.path.join(img_dir, fname)
    fig.savefig(outpath, dpi=150, facecolor='white')
    plt.close(fig)

# [MS] 2.4

def plot_pagerank_distribution(ranks: Dict[Any, float], damping: float, img_dir: str) -> None:
    vals = np.array(list(ranks.values()), dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return

    fit = powerlaw.Fit(vals, discrete=False)
    xmin = fit.xmin
    alpha = fit.power_law.alpha

    uniq, counts = np.unique(vals, return_counts=True)
    pdf_emp = counts / counts.sum()

    mask = uniq >= xmin
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(uniq[~mask], pdf_emp[~mask],
               color='gray', alpha=0.6, s=20, label='x < xmin')
    ax.scatter(uniq[mask], pdf_emp[mask],
               color='C0', alpha=0.8, s=20, label='x ≥ xmin')

    x = np.logspace(np.log10(xmin), np.log10(vals.max()), 200)
    y = (alpha - 1) / xmin * (x / xmin) ** (-alpha)
    ax.plot(x, y, color='C3', linestyle='--',
            label=f'fit (alpha={alpha:.2f})')

    ax.axvline(xmin, color='black', linestyle=':',
               label=f'xmin = {xmin:.2f}')
    ax.fill_between(x, y, alpha=0.2, color='C3')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('PageRank')
    ax.set_ylabel('PDF')
    ax.legend()

    os.makedirs(img_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(
        img_dir,
        f'pagerank_dist_{str(damping).replace(".", "_")}.png'
    ))
    plt.close(fig)
    
def plot_pagerank_convergence(diffs: List[float], damping: float, img_dir: str) -> None:
    plt.figure()
    plt.plot(range(1, len(diffs) + 1), diffs, marker='o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('L1 difference')
    plt.savefig(os.path.join(img_dir, f'pagerank_conv_{str(damping).replace(".", "_")}.png'))
    plt.close()

def plot_all_pagerank_convergence(convs: Dict[float, List[float]], output_path: str) -> None:
    """Plot ||Δr(t)|| vs iteration for multiple damping factors on one log-scale plot."""
    
    plt.figure()
    for d, diffs in sorted(convs.items()):
        plt.plot(
            range(1, len(diffs) + 1),
            diffs,
            marker="o",
            linestyle="-",
            label=f"beta={d:.2f}"
        )
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("L1 difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_pagerank_histogram_and_powerlaw(ranks: Dict[Any, float], damping: float, 
                                         output_path: str) -> None:
    """Plot a log-log histogram of PageRank values and overlay the MLE power-law fit (using powerlaw.Fit)."""
    
    vals = np.array(list(ranks.values()), dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        return

    fit = powerlaw.Fit(vals, discrete=False)
    xmin = fit.xmin
    alpha = fit.power_law.alpha

    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 1000)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])

    plt.figure(figsize=(6, 4))
    plt.loglog(centers, hist, marker="o", linestyle="none", label="empirical")

    x_fit = np.logspace(np.log10(xmin), np.log10(vals.max()), 200)
    
    frac = np.sum(vals >= xmin) / vals.size
    C = (alpha - 1) / xmin
    y_fit = C * (x_fit / xmin) ** (-alpha)
    y_fit *= frac
    plt.loglog(x_fit, y_fit, "r--", label=f"fit (alpha={alpha:.2f})")

    plt.axvline(xmin, color="gray", linestyle=":", label=f"xmin = {xmin}")
    plt.xlabel("PageRank")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()   
