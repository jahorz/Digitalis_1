#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import numpy as np


# CONFIG

ks_file = "PATH/TO/all_Ks.txt"
output_dir = "PATH/TO/OUTPUT_DIR"
os.makedirs(output_dir, exist_ok=True)
bins = 200
gmm_components = 5


# Read Ks values

ks_df = pd.read_csv(ks_file, header=None)
ks_series = ks_df.iloc[:,0].dropna()  # drop NaNs
print(f"Total Ks values read: {len(ks_series)}")

# Filtered Ks for general plotting
ks_filtered = ks_series[(ks_series >= 0.02) & (ks_series <= 5)]

# Filter Ks > 0.5 for ancient events
ks_ancient = ks_series[ks_series > 0.5]


#Function to plot histogram + KDE + peaks

def plot_ks(series, title, filename, bins=200, xlim=None, log_scale=False):
    # Compute KDE for peak detection
    kde = sns.kdeplot(series, bw_adjust=0.3)
    xs, ys = kde.get_lines()[0].get_data()
    plt.close()  # close dummy plot

    # Plot histogram + KDE
    plt.figure(figsize=(13,8))
    sns.histplot(series, bins=bins, kde=True, color='skyblue', stat='count')

    plt.title(title, pad=20)
    plt.xlabel("Ks")
    plt.ylabel("Frequency")
    if xlim:
        plt.xlim(xlim)
    if log_scale:
        plt.yscale("log")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# Functions to fit and plot GMM

def fit_gmm(series, n_components=5):
    X = series.values.reshape(-1,1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    sorted_idx = np.argsort(means)
    return means[sorted_idx], stds[sorted_idx], weights[sorted_idx]

def plot_gmm(series, title, filename, n_components=5, bins=200, log_scale=False):
    means, stds, weights = fit_gmm(series, n_components)
    
    plt.figure(figsize=(13,8))
    
    # Counts histogram
    sns.histplot(series, bins=bins, kde=False, color='skyblue', stat='count')
    x = np.linspace(series.min(), series.max(), 1000)
    for mu, sigma, w in zip(means, stds, weights):
        # Standard Gaussian PDF
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        # Scale by component weight
        pdf_weighted = w * pdf
        # Convert density to counts for overlaying histogram
        y = pdf_weighted * len(series) * (x[1] - x[0])
        # plot curves into histogram and add labels at mean 
        plt.plot(x, y, color='red', linestyle='--', alpha=0.7)
        plt.text(mu, plt.ylim()[1]*0.9, f"{mu:.2f}", color='red', rotation=90, va='top')
    plt.ylabel("Frequency")
    
    plt.title(title, pad=20)
    plt.xlabel("Ks")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename} with GMM overlay")

# Generate plots

# All Ks with GMM
plot_gmm(
    ks_series,
    title="Ks Distribution with GMM (All Pairs)",
    filename=os.path.join(output_dir, "Ks_all_GMM.png"),
    n_components=gmm_components
)

# Ks > 0.5 (ancient events)
plot_ks(
    ks_ancient,
    title="Ks Distribution (Ks > 0.5, Ancient Events)",
    filename=os.path.join(output_dir, "Ks_ancient.png"),
    bins=bins,
    xlim=(0.5, ks_series.max())
)

# All Ks in y log scale
plot_ks(
    ks_series,
    title="Ks Distribution)",
    filename=os.path.join(output_dir, "Ks_log_all.png"),
    bins=bins,
    xlim=(0, ks_series.max()),
    log_scale=True
    )

print(f"Total pairs: {len(ks_series)}, Filtered pairs (0.02–5): {len(ks_filtered)}, Ancient pairs (Ks>0.5): {len(ks_ancient)}")
