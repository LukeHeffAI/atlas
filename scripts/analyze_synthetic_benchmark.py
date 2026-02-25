"""Analyze synthetic vs real task vector benchmark results.

Reads the benchmark JSON produced by src/benchmark_synthetic_vs_real.py and
generates visualizations and a Markdown report.

Usage:
    python scripts/analyze_synthetic_benchmark.py \
        --results-file checkpoints/ViT-B-32/benchmark_synthetic_vs_real.json \
        --output-dir results
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


CONDITION_LABELS = {
    "real_pool_learned_coef": "Real + Learned",
    "synthetic_pool_learned_coef": "Synthetic + Learned",
    "mixed_pool_learned_coef": "Mixed + Learned",
    "real_pool_hypernetwork": "Real + Hypernetwork",
    "synthetic_pool_hypernetwork": "Synthetic + Hypernetwork",
}

CONDITION_COLORS = {
    "real_pool_learned_coef": "#2196F3",
    "synthetic_pool_learned_coef": "#FF9800",
    "mixed_pool_learned_coef": "#4CAF50",
    "real_pool_hypernetwork": "#9C27B0",
    "synthetic_pool_hypernetwork": "#F44336",
}

CONDITIONS = list(CONDITION_LABELS.keys())


def load_results(results_file: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_accuracies(results: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract top-1 accuracies from results, converting to percentages.

    Returns:
        Dict mapping dataset name -> condition name -> accuracy (as percentage) or None.
    """
    accuracies = {}
    for dataset, dataset_results in results["results"].items():
        accuracies[dataset] = {}
        for cond in CONDITIONS:
            val = dataset_results.get(cond, {})
            if "error" in val or "top1" not in val:
                accuracies[dataset][cond] = None
            else:
                accuracies[dataset][cond] = val["top1"] * 100
    return accuracies


def plot_bar_chart(accuracies: Dict, output_path: str):
    """Bar chart: top-1 accuracy per dataset across all 5 conditions."""
    datasets = sorted(accuracies.keys())
    n_datasets = len(datasets)
    n_conditions = len(CONDITIONS)

    fig, ax = plt.subplots(figsize=(max(14, n_datasets * 1.2), 6))

    bar_width = 0.15
    x = np.arange(n_datasets)

    for j, cond in enumerate(CONDITIONS):
        values = []
        for ds in datasets:
            val = accuracies[ds].get(cond)
            values.append(val if val is not None else 0)
        offset = (j - n_conditions / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=CONDITION_LABELS[cond],
            color=CONDITION_COLORS[cond],
            alpha=0.85,
        )
        # Mark missing values
        for i, val in enumerate(values):
            if accuracies[datasets[i]].get(cond) is None:
                ax.text(
                    x[i] + offset, 1, "N/A",
                    ha='center', va='bottom', fontsize=6, rotation=90,
                )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Synthetic vs Real Task Vectors: Per-Dataset Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart: {output_path}")


def plot_scatter(accuracies: Dict, output_path: str):
    """Scatter plot: synthetic pool accuracy vs real pool accuracy per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Learned coefficients
    ax = axes[0]
    real_vals, synth_vals, labels = [], [], []
    for ds in sorted(accuracies.keys()):
        r = accuracies[ds].get("real_pool_learned_coef")
        s = accuracies[ds].get("synthetic_pool_learned_coef")
        if r is not None and s is not None:
            real_vals.append(r)
            synth_vals.append(s)
            labels.append(ds)

    if real_vals:
        ax.scatter(real_vals, synth_vals, c=CONDITION_COLORS["real_pool_learned_coef"], s=60, alpha=0.8)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (real_vals[i], synth_vals[i]), fontsize=7, ha='left', va='bottom')
        lims = [min(min(real_vals), min(synth_vals)) - 5, max(max(real_vals), max(synth_vals)) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_xlabel("Real Pool Accuracy (%)")
    ax.set_ylabel("Synthetic Pool Accuracy (%)")
    ax.set_title("Learned Coefficients")
    ax.grid(alpha=0.3)

    # Hypernetwork coefficients
    ax = axes[1]
    real_vals, synth_vals, labels = [], [], []
    for ds in sorted(accuracies.keys()):
        r = accuracies[ds].get("real_pool_hypernetwork")
        s = accuracies[ds].get("synthetic_pool_hypernetwork")
        if r is not None and s is not None:
            real_vals.append(r)
            synth_vals.append(s)
            labels.append(ds)

    if real_vals:
        ax.scatter(real_vals, synth_vals, c=CONDITION_COLORS["real_pool_hypernetwork"], s=60, alpha=0.8)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (real_vals[i], synth_vals[i]), fontsize=7, ha='left', va='bottom')
        lims = [min(min(real_vals), min(synth_vals)) - 5, max(max(real_vals), max(synth_vals)) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_xlabel("Real Pool Accuracy (%)")
    ax.set_ylabel("Synthetic Pool Accuracy (%)")
    ax.set_title("Hypernetwork Coefficients")
    ax.grid(alpha=0.3)

    plt.suptitle("Real vs Synthetic Pool: Per-Dataset Accuracy", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot: {output_path}")


def plot_delta_chart(accuracies: Dict, output_path: str):
    """Delta chart: accuracy gain/loss from using synthetic TVs."""
    datasets = sorted(accuracies.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 0.8), 5))

    deltas_learned = []
    deltas_hyper = []
    valid_datasets = []

    for ds in datasets:
        r_learned = accuracies[ds].get("real_pool_learned_coef")
        s_learned = accuracies[ds].get("synthetic_pool_learned_coef")
        r_hyper = accuracies[ds].get("real_pool_hypernetwork")
        s_hyper = accuracies[ds].get("synthetic_pool_hypernetwork")

        d_learned = (s_learned - r_learned) if (r_learned is not None and s_learned is not None) else None
        d_hyper = (s_hyper - r_hyper) if (r_hyper is not None and s_hyper is not None) else None

        if d_learned is not None or d_hyper is not None:
            valid_datasets.append(ds)
            deltas_learned.append(d_learned if d_learned is not None else 0)
            deltas_hyper.append(d_hyper if d_hyper is not None else 0)

    if not valid_datasets:
        plt.close()
        return

    x = np.arange(len(valid_datasets))
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width / 2, deltas_learned, bar_width,
                   label="Learned Coefs", color=CONDITION_COLORS["synthetic_pool_learned_coef"], alpha=0.85)
    bars2 = ax.bar(x + bar_width / 2, deltas_hyper, bar_width,
                   label="Hypernetwork Coefs", color=CONDITION_COLORS["synthetic_pool_hypernetwork"], alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy Delta (Synthetic - Real) (%)")
    ax.set_title("Accuracy Change: Synthetic vs Real Task Vector Pool")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved delta chart: {output_path}")


def compute_summary(accuracies: Dict) -> Dict[str, Dict[str, float]]:
    """Compute average accuracy and gap per condition."""
    summary = {}
    for cond in CONDITIONS:
        values = [accuracies[ds][cond] for ds in accuracies if accuracies[ds].get(cond) is not None]
        if values:
            summary[cond] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }
        else:
            summary[cond] = {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}

    # Compute gaps
    real_learned = summary["real_pool_learned_coef"]["mean"]
    synth_learned = summary["synthetic_pool_learned_coef"]["mean"]
    real_hyper = summary["real_pool_hypernetwork"]["mean"]
    synth_hyper = summary["synthetic_pool_hypernetwork"]["mean"]

    summary["gap_learned"] = real_learned - synth_learned
    summary["gap_hypernetwork"] = real_hyper - synth_hyper

    return summary


def generate_markdown_report(
    data: Dict,
    accuracies: Dict,
    summary: Dict,
    output_path: str,
    plot_dir: str,
):
    """Generate Markdown report with tables, metrics, and figure references."""
    model = data.get("model", "unknown")
    backend = data.get("synthetic_backend", "unknown")
    subsample = data.get("subsample", "unknown")
    elapsed = data.get("elapsed_seconds", 0)

    datasets = sorted(accuracies.keys())

    lines = []
    lines.append(f"# Synthetic vs Real Task Vector Benchmark Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {model}")
    lines.append(f"**T2I Backend:** {backend}")
    lines.append(f"**Subsample:** {subsample}")
    lines.append(f"**Runtime:** {elapsed:.0f}s")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Condition | Mean Acc (%) | Std | N Datasets |")
    lines.append("|-----------|-------------|-----|------------|")
    for cond in CONDITIONS:
        s = summary[cond]
        label = CONDITION_LABELS[cond]
        lines.append(f"| {label} | {s['mean']:.2f} | {s['std']:.2f} | {s['count']} |")
    lines.append("")
    lines.append(f"**Accuracy gap (real - synthetic), learned coefs:** {summary['gap_learned']:.2f}%")
    lines.append(f"**Accuracy gap (real - synthetic), hypernetwork:** {summary['gap_hypernetwork']:.2f}%")
    lines.append("")

    # Per-dataset table
    lines.append("## Per-Dataset Results")
    lines.append("")
    header = "| Dataset | " + " | ".join(CONDITION_LABELS[c] for c in CONDITIONS) + " |"
    separator = "|---------|" + "|".join(["-------" for _ in CONDITIONS]) + "|"
    lines.append(header)
    lines.append(separator)

    for ds in datasets:
        row = f"| {ds} |"
        for cond in CONDITIONS:
            val = accuracies[ds].get(cond)
            if val is not None:
                row += f" {val:.2f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    # Biggest winners/losers
    lines.append("## Per-Dataset Analysis")
    lines.append("")
    lines.append("### Datasets that benefit most from synthetic TVs (learned coefs)")
    lines.append("")

    deltas = {}
    for ds in datasets:
        r = accuracies[ds].get("real_pool_learned_coef")
        s = accuracies[ds].get("synthetic_pool_learned_coef")
        if r is not None and s is not None:
            deltas[ds] = s - r

    if deltas:
        sorted_deltas = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
        for ds, delta in sorted_deltas:
            sign = "+" if delta >= 0 else ""
            lines.append(f"- **{ds}**: {sign}{delta:.2f}%")
    else:
        lines.append("- No comparable results available")
    lines.append("")

    # Figures
    rel_plot_dir = os.path.relpath(plot_dir, os.path.dirname(output_path))
    lines.append("## Figures")
    lines.append("")
    lines.append(f"![Per-dataset comparison]({rel_plot_dir}/bar_chart.png)")
    lines.append("")
    lines.append(f"![Real vs synthetic scatter]({rel_plot_dir}/scatter_plot.png)")
    lines.append("")
    lines.append(f"![Accuracy delta]({rel_plot_dir}/delta_chart.png)")
    lines.append("")

    # Incomplete pools
    if "incomplete_pools" in data:
        lines.append("## Incomplete Pools")
        lines.append("")
        for ds, info in data["incomplete_pools"].items():
            if info.get("missing_real"):
                lines.append(f"- **{ds}** missing real TVs: {', '.join(info['missing_real'])}")
            if info.get("missing_synthetic"):
                lines.append(f"- **{ds}** missing synthetic TVs: {', '.join(info['missing_synthetic'])}")
        lines.append("")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic vs real benchmark results")
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to benchmark_synthetic_vs_real.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output plots and report",
    )
    args = parser.parse_args()

    # Load results
    data = load_results(args.results_file)
    accuracies = extract_accuracies(data)

    if not accuracies:
        print("No results found in the benchmark file.")
        return

    # Create output directories
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Generate plots
    plot_bar_chart(accuracies, os.path.join(plot_dir, "bar_chart.png"))
    plot_scatter(accuracies, os.path.join(plot_dir, "scatter_plot.png"))
    plot_delta_chart(accuracies, os.path.join(plot_dir, "delta_chart.png"))

    # Also save as PDF
    plot_bar_chart(accuracies, os.path.join(plot_dir, "bar_chart.pdf"))
    plot_scatter(accuracies, os.path.join(plot_dir, "scatter_plot.pdf"))
    plot_delta_chart(accuracies, os.path.join(plot_dir, "delta_chart.pdf"))

    # Compute summary stats
    summary = compute_summary(accuracies)

    # Generate Markdown report
    report_path = os.path.join(args.output_dir, "synthetic_benchmark_report.md")
    generate_markdown_report(data, accuracies, summary, report_path, plot_dir)

    print("\nDone! Generated:")
    print(f"  Report: {report_path}")
    print(f"  Plots:  {plot_dir}/")


if __name__ == "__main__":
    main()
