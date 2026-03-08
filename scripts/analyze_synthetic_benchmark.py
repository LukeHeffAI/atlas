"""Analyze aTLAS few-shot results: zero-shot vs aTLAS (per shot count) vs fine-tuned.

Auto-discovers models from results/ and checkpoints/ directories and generates
per-model visualizations plus a cross-model comparison when multiple models exist.

Usage:
    python scripts/analyze_synthetic_benchmark.py
    python scripts/analyze_synthetic_benchmark.py --model ViT-B-32 --output-dir results/analysis
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_models(results_dir: str, checkpoints_dir: str) -> List[str]:
    """Find model names that have learned_composition.json in results/."""
    models = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return models
    for entry in sorted(results_path.iterdir()):
        if entry.is_dir() and (entry / "learned_composition.json").exists():
            models.append(entry.name)
    return models


def load_json(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_model_data(model: str, results_dir: str, checkpoints_dir: str) -> Dict:
    """Load all data for a single model.

    Returns dict with keys: composition, ft, zeroshot, shots, datasets.
    """
    composition = load_json(os.path.join(results_dir, model, "learned_composition.json"))
    ft = load_json(os.path.join(checkpoints_dir, model, "ft_accuracies.json"))
    zeroshot = load_json(os.path.join(checkpoints_dir, model, "zeroshot_accuracies.json"))

    if composition is None:
        return {"composition": None, "ft": ft, "zeroshot": zeroshot, "shots": [], "datasets": []}

    # Discover shot counts (sorted numerically)
    shots = sorted(
        [int(k.replace("_shot", "")) for k in composition.keys() if k.endswith("_shot")]
    )

    # Discover datasets from the shot block with most entries
    datasets = set()
    for shot_key in composition:
        block = composition[shot_key]
        for k in block:
            if k.endswith("_trained") and not k.endswith("Val_trained"):
                ds = k.replace("_trained", "")
                datasets.add(ds)
    datasets = sorted(datasets)

    return {
        "composition": composition,
        "ft": ft,
        "zeroshot": zeroshot,
        "shots": shots,
        "datasets": datasets,
    }


def get_atlas_acc(composition: Dict, shot: int, dataset: str) -> Optional[float]:
    """Get aTLAS test-set accuracy for a given shot count and dataset."""
    block = composition.get(f"{shot}_shot")
    if block is None:
        return None
    val = block.get(f"{dataset}_trained")
    if val is not None:
        return val * 100
    return None


def get_baseline_acc(acc_dict: Optional[Dict], dataset: str) -> Optional[float]:
    """Get test-set accuracy from ft_accuracies or zeroshot_accuracies."""
    if acc_dict is None:
        return None
    val = acc_dict.get(dataset)
    if val is not None:
        return val * 100
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SHOT_COLORS = {
    1: "#66c2a5",
    2: "#fc8d62",
    4: "#8da0cb",
    8: "#e78ac3",
    16: "#a6d854",
}
ZS_COLOR = "#377eb8"
FT_COLOR = "#e41a1c"


def _color_for_shot(shot: int) -> str:
    return SHOT_COLORS.get(shot, "#999999")


def plot_grouped_bar(
    datasets: List[str],
    shots: List[int],
    composition: Dict,
    ft: Optional[Dict],
    zeroshot: Optional[Dict],
    model: str,
    output_path: str,
):
    """Grouped bar chart: zero-shot, aTLAS per shot, fine-tuned per dataset."""
    # Build condition list: zero-shot, each shot, fine-tuned
    conditions = ["Zero-shot"] + [f"{s}-shot aTLAS" for s in shots] + ["Fine-tuned"]
    n_cond = len(conditions)
    n_ds = len(datasets)

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 1.5), 6))
    bar_width = 0.8 / n_cond
    x = np.arange(n_ds)

    for j, cond in enumerate(conditions):
        values = []
        for ds in datasets:
            if cond == "Zero-shot":
                val = get_baseline_acc(zeroshot, ds)
            elif cond == "Fine-tuned":
                val = get_baseline_acc(ft, ds)
            else:
                shot = int(cond.split("-")[0])
                val = get_atlas_acc(composition, shot, ds)
            values.append(val if val is not None else 0)

        offset = (j - n_cond / 2 + 0.5) * bar_width
        color = (
            ZS_COLOR if cond == "Zero-shot"
            else FT_COLOR if cond == "Fine-tuned"
            else _color_for_shot(int(cond.split("-")[0]))
        )
        ax.bar(x + offset, values, bar_width, label=cond, color=color, alpha=0.85)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"{model}: Zero-shot vs aTLAS Few-shot vs Fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved grouped bar chart: {output_path}")


def plot_delta_chart(
    datasets: List[str],
    shots: List[int],
    composition: Dict,
    ft: Optional[Dict],
    model: str,
    output_path: str,
):
    """Delta chart: gap between aTLAS (each shot) and fine-tuned upper bound."""
    n_ds = len(datasets)
    n_shots = len(shots)

    fig, ax = plt.subplots(figsize=(max(12, n_ds * 1.2), 5))
    bar_width = 0.8 / n_shots
    x = np.arange(n_ds)

    for j, shot in enumerate(shots):
        deltas = []
        for ds in datasets:
            atlas_val = get_atlas_acc(composition, shot, ds)
            ft_val = get_baseline_acc(ft, ds)
            if atlas_val is not None and ft_val is not None:
                deltas.append(atlas_val - ft_val)
            else:
                deltas.append(0)
        offset = (j - n_shots / 2 + 0.5) * bar_width
        ax.bar(x + offset, deltas, bar_width, label=f"{shot}-shot",
               color=_color_for_shot(shot), alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy Gap vs Fine-tuned (%)")
    ax.set_title(f"{model}: aTLAS Accuracy Gap to Fine-tuned Upper Bound")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved delta chart: {output_path}")


def plot_shot_scaling(
    datasets: List[str],
    shots: List[int],
    composition: Dict,
    ft: Optional[Dict],
    zeroshot: Optional[Dict],
    model: str,
    output_path: str,
):
    """Line plot: accuracy vs shot count per dataset, with ZS/FT reference lines."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(datasets))

    for i, ds in enumerate(datasets):
        accs = []
        valid_shots = []
        for shot in shots:
            val = get_atlas_acc(composition, shot, ds)
            if val is not None:
                accs.append(val)
                valid_shots.append(shot)
        if accs:
            ax.plot(valid_shots, accs, "o-", label=ds, color=cmap(i), linewidth=1.5,
                    markersize=5)

        # Reference lines
        zs_val = get_baseline_acc(zeroshot, ds)
        ft_val = get_baseline_acc(ft, ds)
        if zs_val is not None:
            ax.axhline(y=zs_val, color=cmap(i), linestyle=":", alpha=0.3, linewidth=0.8)
        if ft_val is not None:
            ax.axhline(y=ft_val, color=cmap(i), linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Number of Shots")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"{model}: aTLAS Accuracy vs Shot Count")
    ax.set_xticks(shots)
    ax.set_xticklabels([str(s) for s in shots])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.grid(alpha=0.3)

    # Add a legend note for reference lines
    ax.plot([], [], "k:", alpha=0.5, label="Zero-shot (dotted)")
    ax.plot([], [], "k--", alpha=0.5, label="Fine-tuned (dashed)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved shot scaling plot: {output_path}")


def plot_cross_model(
    model_summaries: Dict[str, Dict],
    output_path: str,
):
    """Bar chart comparing mean aTLAS accuracy at best shot count across models."""
    models = sorted(model_summaries.keys())
    if len(models) < 2:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))

    conditions = ["Zero-shot", "Best aTLAS", "Fine-tuned"]
    colors = [ZS_COLOR, "#4CAF50", FT_COLOR]
    bar_width = 0.25
    x = np.arange(len(models))

    for j, (cond, color) in enumerate(zip(conditions, colors)):
        values = [model_summaries[m].get(cond, 0) for m in models]
        offset = (j - 1) * bar_width
        ax.bar(x + offset, values, bar_width, label=cond, color=color, alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Top-1 Accuracy (%)")
    ax.set_title("Cross-Model Comparison: Mean Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross-model comparison: {output_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    model: str,
    data: Dict,
    output_path: str,
    plot_dir: str,
) -> Dict[str, float]:
    """Generate Markdown report for a single model. Returns summary dict for cross-model."""
    composition = data["composition"]
    ft = data["ft"]
    zeroshot = data["zeroshot"]
    shots = data["shots"]
    datasets = data["datasets"]

    rel_plot_dir = os.path.relpath(plot_dir, os.path.dirname(output_path))

    lines = []
    lines.append(f"# aTLAS Few-Shot Analysis: {model}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** {model}")
    lines.append(f"**Shot counts:** {', '.join(str(s) for s in shots)}")
    lines.append(f"**Datasets:** {len(datasets)}")
    lines.append("")

    # Summary table: mean accuracy per condition
    lines.append("## Summary (Mean Accuracy Across Datasets)")
    lines.append("")
    header = "| Condition | Mean Acc (%) | Datasets with Data |"
    sep = "|-----------|-------------|-------------------|"
    lines.append(header)
    lines.append(sep)

    summary = {}

    # Zero-shot
    zs_vals = [get_baseline_acc(zeroshot, ds) for ds in datasets]
    zs_vals = [v for v in zs_vals if v is not None]
    zs_mean = np.mean(zs_vals) if zs_vals else 0
    summary["Zero-shot"] = zs_mean
    lines.append(f"| Zero-shot | {zs_mean:.2f} | {len(zs_vals)} |")

    # Per shot
    best_shot_mean = 0
    for shot in shots:
        vals = [get_atlas_acc(composition, shot, ds) for ds in datasets]
        vals = [v for v in vals if v is not None]
        mean_val = np.mean(vals) if vals else 0
        summary[f"{shot}-shot aTLAS"] = mean_val
        lines.append(f"| {shot}-shot aTLAS | {mean_val:.2f} | {len(vals)} |")
        if mean_val > best_shot_mean:
            best_shot_mean = mean_val

    summary["Best aTLAS"] = best_shot_mean

    # Fine-tuned
    ft_vals = [get_baseline_acc(ft, ds) for ds in datasets]
    ft_vals = [v for v in ft_vals if v is not None]
    ft_mean = np.mean(ft_vals) if ft_vals else 0
    summary["Fine-tuned"] = ft_mean
    lines.append(f"| Fine-tuned | {ft_mean:.2f} | {len(ft_vals)} |")

    lines.append("")

    # Per-dataset table
    lines.append("## Per-Dataset Results")
    lines.append("")
    cols = ["Dataset", "Zero-shot"] + [f"{s}-shot" for s in shots] + ["Fine-tuned"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["-------" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)

    for ds in datasets:
        row_vals = [ds]
        zs_v = get_baseline_acc(zeroshot, ds)
        row_vals.append(f"{zs_v:.2f}" if zs_v is not None else "N/A")
        for shot in shots:
            v = get_atlas_acc(composition, shot, ds)
            row_vals.append(f"{v:.2f}" if v is not None else "N/A")
        ft_v = get_baseline_acc(ft, ds)
        row_vals.append(f"{ft_v:.2f}" if ft_v is not None else "N/A")
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")

    # Best/worst datasets (relative to fine-tuned, at best available shot)
    lines.append("## Best/Worst Datasets (aTLAS vs Fine-tuned)")
    lines.append("")

    gaps = {}
    for ds in datasets:
        ft_v = get_baseline_acc(ft, ds)
        if ft_v is None:
            continue
        # Use best shot for this dataset
        best_atlas = None
        for shot in shots:
            v = get_atlas_acc(composition, shot, ds)
            if v is not None and (best_atlas is None or v > best_atlas):
                best_atlas = v
        if best_atlas is not None:
            gaps[ds] = best_atlas - ft_v

    if gaps:
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        lines.append("**Closest to fine-tuned (smallest gap):**")
        for ds, gap in sorted_gaps[:5]:
            sign = "+" if gap >= 0 else ""
            lines.append(f"- {ds}: {sign}{gap:.2f}%")
        lines.append("")
        lines.append("**Largest gap from fine-tuned:**")
        for ds, gap in sorted_gaps[-5:]:
            sign = "+" if gap >= 0 else ""
            lines.append(f"- {ds}: {sign}{gap:.2f}%")
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append(f"![Grouped bar chart]({rel_plot_dir}/grouped_bar.png)")
    lines.append("")
    lines.append(f"![Delta chart]({rel_plot_dir}/delta_chart.png)")
    lines.append("")
    lines.append(f"![Shot scaling]({rel_plot_dir}/shot_scaling.png)")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report: {output_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze aTLAS few-shot results: zero-shot vs aTLAS vs fine-tuned"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Root results directory (default: results)",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="Root checkpoints directory (default: checkpoints)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/analysis",
        help="Where to write plots and reports (default: results/analysis)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Analyze a single model (default: all discovered models)",
    )
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = discover_models(args.results_dir, args.checkpoints_dir)

    if not models:
        print("No models found. Ensure results/<model>/learned_composition.json exists.")
        return

    print(f"Discovered models: {', '.join(models)}")

    model_summaries = {}

    for model in models:
        print(f"\nProcessing {model}...")
        data = load_model_data(model, args.results_dir, args.checkpoints_dir)

        if data["composition"] is None:
            print(f"  Skipping {model}: no learned_composition.json found")
            continue

        if not data["datasets"]:
            print(f"  Skipping {model}: no datasets found in results")
            continue

        # Create output dirs
        model_plot_dir = os.path.join(args.output_dir, model, "plots")
        os.makedirs(model_plot_dir, exist_ok=True)

        composition = data["composition"]
        ft = data["ft"]
        zeroshot = data["zeroshot"]
        shots = data["shots"]
        datasets = data["datasets"]

        # Generate plots (PNG + PDF)
        for ext in ["png", "pdf"]:
            plot_grouped_bar(
                datasets, shots, composition, ft, zeroshot, model,
                os.path.join(model_plot_dir, f"grouped_bar.{ext}"),
            )
            plot_delta_chart(
                datasets, shots, composition, ft, model,
                os.path.join(model_plot_dir, f"delta_chart.{ext}"),
            )
            plot_shot_scaling(
                datasets, shots, composition, ft, zeroshot, model,
                os.path.join(model_plot_dir, f"shot_scaling.{ext}"),
            )

        # Generate report
        report_path = os.path.join(args.output_dir, model, "report.md")
        summary = generate_report(model, data, report_path, model_plot_dir)
        model_summaries[model] = summary

    # Cross-model comparison
    if len(model_summaries) > 1:
        print("\nGenerating cross-model comparison...")
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
        for ext in ["png", "pdf"]:
            plot_cross_model(
                model_summaries,
                os.path.join(args.output_dir, "plots", f"cross_model.{ext}"),
            )

    print(f"\nDone! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
