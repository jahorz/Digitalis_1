#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde
from scipy.signal import find_peaks
import numpy as np

# CONFIG
single_file = "single_busco_coverage.txt"
dup_file = "duplicated_busco_coverage.txt"
output = "BUSCO_coverage_violin_with_peak_lines.png"

# LOAD DATA
cols = ["contig", "start", "end", "gene", "coverage"]

single = pd.read_csv(single_file, sep="\t", names=cols)
dup = pd.read_csv(dup_file, sep="\t", names=cols)

single["category"] = "Single-copy BUSCOs"
dup["category"] = "Duplicated BUSCOs"

df = pd.concat([single, dup], ignore_index=True)

# STATISTICAL TEST
u_stat, pval = mannwhitneyu(
    single["coverage"],
    dup["coverage"],
    alternative="two-sided"
)
print(f"Mann–Whitney U p-value: {pval:.3e}")

# PLOTTING VIOLIN
plt.figure(figsize=(6,5))

sns.violinplot(
    data=df,
    x="category",
    y="coverage",
    inner=None,
    cut=0,
    linewidth=1
)

sns.boxplot(
    data=df,
    x="category",
    y="coverage",
    width=0.05,
    showcaps=True,
    boxprops={"facecolor": "white"},
    showfliers=False
)

plt.ylabel("Mean read depth")
plt.xlabel("")
plt.title("Coverage of single-copy vs duplicated BUSCO genes")

# PEAK DETECTION
def detect_peaks(data, bw=0.22, min_distance=40, min_height=0.01):
    kde = gaussian_kde(data, bw_method=bw)
    x = np.linspace(min(data), max(data), 1000)
    y = kde(x)
    peaks, _ = find_peaks(y, distance=min_distance, height=min_height)
    return x[peaks], y[peaks]



# SIGNIFICANCE ANNOTATION

def p_to_stars(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

y_max = df["coverage"].max()
y_sig = y_max * 1.10
y_lim = y_max * 1.25

plt.ylim(0, y_lim)

# Draw horizontal lines at peaks
categories = df["category"].unique()
for i, cat in enumerate(categories):
    subset = df[df["category"] == cat]["coverage"].values
    peak_xs, peak_ys = detect_peaks(subset)
    print(f"Peaks for {cat}: {np.round(peak_xs, 2)}")

    px_snapped = [min(subset, key=lambda v: abs(v - p)) for p in peak_xs]

    # Horizontal lines across the violin
    for px in px_snapped:
        plt.hlines(
            y=px,       # y-coordinate = peak coverage
            xmin=i-0.35, 
            xmax=i+0.35,
            colors="red",
            linestyles="--",
            linewidth=1
        )

plt.plot([0, 1], [y_sig, y_sig], color="black", linewidth=1)
stars = p_to_stars(pval)
plt.text(
    0.5,
    y_sig * 1.03,
    stars,
    ha="center",
    va="bottom",
    fontsize=14
)

plt.tight_layout()
plt.savefig(output, dpi=300)
plt.close()
print(f"Saved plot to {output}")
