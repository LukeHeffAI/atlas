"""Compare multi-modal hypernetwork against text-only baseline.

This script loads evaluation results from both approaches and generates
publication-quality comparison plots:

1. **Shot-count sweep plot**: Top-1 accuracy vs. number of support images
   (0, 1, 2, 4, 8, 16), with the text-only zero-shot baseline as a
   horizontal reference line.
2. **Per-dataset bar chart**: Side-by-side comparison at a fixed shot count
   across all evaluated datasets.
3. **Improvement heatmap** (optional): Absolute accuracy improvement of
   multi-modal over text-only across datasets and shot counts.

Results are saved as PDF (vector) and PNG (raster) for flexibility.

Usage:
    python scripts/compare_multimodal_vs_text.py \\
        --multimodal-results checkpoints/ViT-B-32/multimodal_adapted/ \\
        --textonly-results checkpoints/ViT-B-32/text_adapted/ \\
        --output figures/multimodal_vs_text.pdf

    # Compare specific datasets only
    python scripts/compare_multimodal_vs_text.py \\
        --multimodal-results checkpoints/ViT-B-32/multimodal_adapted/ \\
        --textonly-results checkpoints/ViT-B-32/text_adapted/ \\
        --datasets Flowers102,Cars,DTD \\
        --output figures/multimodal_vs_text_selected.pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_results(results_dir: str) -> Dict:
    """Load all JSON result files from a directory tree.

    Expects structure:
        results_dir/<DatasetName>/<mode>_results.json

    Returns:
        Dict mapping dataset names to their result dicts.
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: results directory not found: {results_dir}")
        return results

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        results[dataset_name] = {}

        for json_file in dataset_dir.glob("*_results.json"):
            with open(json_file) as f:
                data = json.load(f)
            mode = json_file.stem.replace("_results", "")
            results[dataset_name][mode] = data

    return results


def plot_shot_sweep(
    multimodal_results: Dict,
    textonly_results: Dict,
    dataset_name: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot accuracy vs. shot count for one dataset.

    Args:
        multimodal_results: Multi-modal results dict for this dataset.
        textonly_results: Text-only results dict for this dataset.
        dataset_name: Dataset name (for title).
        ax: Matplotlib axes (created if None).

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    # Extract sweep results
    sweep_data = multimodal_results.get("sweep", {})
    if not sweep_data:
        # Try to reconstruct from individual results
        for key, val in multimodal_results.items():
            if isinstance(val, dict) and "top1" in val:
                sweep_data[key] = val

    shot_counts = []
    accuracies = []
    for key in sorted(sweep_data.keys(), key=lambda x: int(x.replace("-shot", ""))):
        k = int(key.replace("-shot", ""))
        shot_counts.append(k)
        accuracies.append(sweep_data[key]["top1"])

    # Text-only baseline
    textonly_acc = None
    if "hypernetwork" in textonly_results:
        textonly_acc = textonly_results["hypernetwork"].get("top1")
    elif "text_only" in textonly_results:
        textonly_acc = textonly_results["text_only"].get("top1")

    # Plot multi-modal sweep
    ax.plot(
        shot_counts, accuracies,
        "o-", color="#2196F3", linewidth=2, markersize=6,
        label="Multi-modal", zorder=3,
    )

    # Plot text-only baseline
    if textonly_acc is not None:
        ax.axhline(
            y=textonly_acc, color="#FF5722", linestyle="--", linewidth=1.5,
            label=f"Text-only ({textonly_acc:.1f}%)", zorder=2,
        )

    # Also mark the 0-shot point from multi-modal (should match text-only)
    if 0 in shot_counts:
        idx = shot_counts.index(0)
        ax.plot(
            0, accuracies[idx], "s", color="#FF5722", markersize=8, zorder=4,
        )

    ax.set_xlabel("Number of support images (K)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(dataset_name)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Use log-ish x-axis ticks
    if shot_counts:
        ax.set_xticks(shot_counts)
        ax.set_xticklabels([str(k) for k in shot_counts])

    return ax


def plot_dataset_comparison(
    multimodal_results: Dict,
    textonly_results: Dict,
    datasets: List[str],
    shot_count: int = 4,
) -> plt.Figure:
    """Bar chart comparing text-only vs. multi-modal at a fixed shot count.

    Args:
        multimodal_results: Full multi-modal results dict.
        textonly_results: Full text-only results dict.
        datasets: List of dataset names to include.
        shot_count: Fixed shot count for comparison.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.2), 5))

    text_accs = []
    mm_accs = []
    valid_datasets = []

    for ds in datasets:
        # Text-only accuracy
        t_acc = None
        if ds in textonly_results:
            t_data = textonly_results[ds]
            if "hypernetwork" in t_data and "top1" in t_data["hypernetwork"]:
                t_acc = t_data["hypernetwork"]["top1"]

        # Multi-modal accuracy at fixed shot count
        m_acc = None
        if ds in multimodal_results:
            m_data = multimodal_results[ds]
            sweep = m_data.get("sweep", {})
            key = f"{shot_count}-shot"
            if key in sweep and "top1" in sweep[key]:
                m_acc = sweep[key]["top1"]

        if t_acc is not None and m_acc is not None:
            text_accs.append(t_acc)
            mm_accs.append(m_acc)
            valid_datasets.append(ds)

    if not valid_datasets:
        ax.text(0.5, 0.5, "No overlapping results found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    x = np.arange(len(valid_datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, text_accs, width, label="Text-only",
                   color="#FF5722", alpha=0.8)
    bars2 = ax.bar(x + width/2, mm_accs, width, label=f"Multi-modal ({shot_count}-shot)",
                   color="#2196F3", alpha=0.8)

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Text-Only vs. Multi-Modal ({shot_count}-shot)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_datasets, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare multi-modal vs. text-only hypernetwork adaptation"
    )
    parser.add_argument(
        "--multimodal-results", required=True,
        help="Directory containing multi-modal evaluation results",
    )
    parser.add_argument(
        "--textonly-results", required=True,
        help="Directory containing text-only evaluation results",
    )
    parser.add_argument(
        "--datasets", type=lambda x: x.split(","), default=None,
        help="Comma-separated list of datasets to include (default: all)",
    )
    parser.add_argument(
        "--shot-count", type=int, default=4,
        help="Fixed shot count for bar chart comparison (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default="figures/multimodal_vs_text.pdf",
        help="Output file path (supports .pdf and .png)",
    )
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    mm_results = load_results(args.multimodal_results)
    text_results = load_results(args.textonly_results)

    if not mm_results and not text_results:
        print("Error: No results found in either directory")
        sys.exit(1)

    # Determine datasets
    datasets = args.datasets or sorted(
        set(mm_results.keys()) | set(text_results.keys())
    )
    print(f"Datasets: {datasets}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Shot sweep per dataset ---
    sweep_datasets = [ds for ds in datasets if ds in mm_results
                      and "sweep" in mm_results[ds]]

    if sweep_datasets:
        ncols = min(3, len(sweep_datasets))
        nrows = (len(sweep_datasets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        if nrows * ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, ds in enumerate(sweep_datasets):
            t_data = text_results.get(ds, {})
            plot_shot_sweep(
                mm_results[ds].get("sweep", mm_results[ds]),
                t_data,
                ds,
                ax=axes[i],
            )

        # Hide unused axes
        for j in range(len(sweep_datasets), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Multi-Modal Adaptation: Accuracy vs. Support Images",
                     fontsize=14, y=1.02)
        fig.tight_layout()

        sweep_path = str(output_path).replace(
            output_path.suffix, f"_sweep{output_path.suffix}"
        )
        fig.savefig(sweep_path)
        print(f"Saved sweep plot: {sweep_path}")
        plt.close(fig)

    # --- Figure 2: Dataset comparison bar chart ---
    fig2 = plot_dataset_comparison(
        mm_results, text_results, datasets, args.shot_count
    )
    fig2.tight_layout()
    fig2.savefig(str(output_path))
    print(f"Saved comparison plot: {output_path}")
    plt.close(fig2)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Text-only':>10} {'MM-{}-shot':>12} {'Δ':>8}".format(
        args.shot_count))
    print("-" * 50)

    for ds in datasets:
        t_acc = "--"
        m_acc = "--"
        delta = "--"

        if ds in text_results:
            t_data = text_results[ds]
            if "hypernetwork" in t_data:
                t_acc = f"{t_data['hypernetwork'].get('top1', 0):.1f}%"

        if ds in mm_results:
            sweep = mm_results[ds].get("sweep", {})
            key = f"{args.shot_count}-shot"
            if key in sweep:
                m_val = sweep[key].get("top1", 0)
                m_acc = f"{m_val:.1f}%"

                if isinstance(t_acc, str) and t_acc != "--":
                    t_val = float(t_acc.rstrip("%"))
                    delta = f"+{m_val - t_val:.1f}%"

        print(f"{ds:<15} {t_acc:>10} {m_acc:>12} {delta:>8}")

    print("=" * 60)


if __name__ == "__main__":
    main()
