"""Analyze text source ablation experiments.

This script analyzes and visualizes the results from comparing different text sources:
- Manual descriptions
- Template-based descriptions (CLIP templates)
- LLM-generated descriptions (GPT-4o, Claude)

Usage:
    python scripts/analyze_text_ablation.py \
        --results-dir checkpoints/ViT-B-32/text_adapted \
        --output ablation_text_source.pdf
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all experiment results from a directory.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dictionary mapping experiment names to results
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return results

    # Find all result JSON files
    for json_file in results_path.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract experiment identifier from path
            rel_path = json_file.relative_to(results_path)
            exp_name = str(rel_path.with_suffix(''))
            results[exp_name] = data

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """Parse experiment name to extract metadata.

    Args:
        exp_name: Experiment name (e.g., "CIFAR10/manual_results")

    Returns:
        Dictionary with parsed metadata
    """
    parts = exp_name.split('/')
    metadata = {
        'dataset': parts[0] if len(parts) > 0 else 'unknown',
        'text_source': 'unknown',
        'variant': None
    }

    if len(parts) > 1:
        result_name = parts[1]
        if 'manual' in result_name:
            metadata['text_source'] = 'manual'
        elif 'template' in result_name:
            metadata['text_source'] = 'templates'
        elif 'gpt4o' in result_name:
            metadata['text_source'] = 'generated'
            metadata['variant'] = 'gpt4o'
        elif 'claude' in result_name:
            metadata['text_source'] = 'generated'
            metadata['variant'] = 'claude'

    return metadata


def aggregate_by_text_source(results: Dict) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate results by text source.

    Args:
        results: Raw results dictionary

    Returns:
        Dictionary mapping text sources to dataset-accuracy mappings
    """
    aggregated = {
        'manual': {},
        'templates': {},
        'gpt4o': {},
        'claude': {}
    }

    for exp_name, data in results.items():
        metadata = parse_experiment_name(exp_name)
        dataset = metadata['dataset']

        # Extract accuracy
        if isinstance(data, dict):
            if 'top1' in data:
                acc = data['top1']
            elif 'hypernetwork' in data and 'top1' in data['hypernetwork']:
                acc = data['hypernetwork']['top1']
            else:
                continue
        else:
            continue

        # Determine source key
        if metadata['text_source'] == 'generated':
            source_key = metadata['variant']
        else:
            source_key = metadata['text_source']

        if source_key in aggregated:
            if dataset not in aggregated[source_key]:
                aggregated[source_key][dataset] = []
            aggregated[source_key][dataset].append(acc)

    return aggregated


def compute_statistics(aggregated: Dict) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean and std for each text source and dataset.

    Args:
        aggregated: Aggregated results

    Returns:
        Dictionary with (mean, std) tuples
    """
    stats = {}

    for source, datasets in aggregated.items():
        stats[source] = {}
        for dataset, accs in datasets.items():
            if accs:
                mean = np.mean(accs)
                std = np.std(accs) if len(accs) > 1 else 0.0
                stats[source][dataset] = (mean, std)

    return stats


def print_summary_table(stats: Dict):
    """Print a summary table of results.

    Args:
        stats: Statistics dictionary
    """
    # Get all datasets
    all_datasets = set()
    for source_stats in stats.values():
        all_datasets.update(source_stats.keys())
    all_datasets = sorted(all_datasets)

    # Get all sources with data
    sources_with_data = [s for s in stats.keys() if stats[s]]

    if not sources_with_data or not all_datasets:
        print("No results to display")
        return

    # Print header
    header = f"{'Dataset':<20}"
    for source in sources_with_data:
        header += f"{source:>15}"
    print("\n" + "=" * (20 + 15 * len(sources_with_data)))
    print("TEXT SOURCE ABLATION RESULTS")
    print("=" * (20 + 15 * len(sources_with_data)))
    print(header)
    print("-" * (20 + 15 * len(sources_with_data)))

    # Print rows
    for dataset in all_datasets:
        row = f"{dataset:<20}"
        for source in sources_with_data:
            if dataset in stats[source]:
                mean, std = stats[source][dataset]
                if std > 0:
                    row += f"{mean:>12.2f}+-{std:.1f}"
                else:
                    row += f"{mean:>15.2f}"
            else:
                row += f"{'--':>15}"
        print(row)

    # Print averages
    print("-" * (20 + 15 * len(sources_with_data)))
    row = f"{'Average':<20}"
    for source in sources_with_data:
        if stats[source]:
            avg = np.mean([v[0] for v in stats[source].values()])
            row += f"{avg:>15.2f}"
        else:
            row += f"{'--':>15}"
    print(row)
    print("=" * (20 + 15 * len(sources_with_data)))


def plot_comparison(stats: Dict, output_path: str):
    """Create comparison bar chart.

    Args:
        stats: Statistics dictionary
        output_path: Path to save plot
    """

    # Get datasets and sources
    all_datasets = set()
    for source_stats in stats.values():
        all_datasets.update(source_stats.keys())
    datasets = sorted(all_datasets)

    sources_with_data = [s for s in ['manual', 'templates', 'gpt4o', 'claude'] if stats.get(s)]

    if not datasets or not sources_with_data:
        print("No data to plot")
        return

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(sources_with_data)

    colors = {
        'manual': '#2196F3',
        'templates': '#4CAF50',
        'gpt4o': '#FF9800',
        'claude': '#9C27B0'
    }

    # Create bars
    for i, source in enumerate(sources_with_data):
        means = []
        stds = []
        for dataset in datasets:
            if dataset in stats[source]:
                means.append(stats[source][dataset][0])
                stds.append(stats[source][dataset][1])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(sources_with_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=source.capitalize(), color=colors.get(source, '#888888'),
                     capsize=3, alpha=0.8)

    # Styling
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Text Source Comparison for Zero-Shot Adaptation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_relative_improvement(stats: Dict, output_path: str, baseline: str = 'templates'):
    """Create relative improvement plot vs baseline.

    Args:
        stats: Statistics dictionary
        output_path: Path to save plot
        baseline: Baseline text source to compare against
    """

    if baseline not in stats or not stats[baseline]:
        print(f"Baseline '{baseline}' not found in results")
        return

    datasets = sorted(stats[baseline].keys())
    sources = [s for s in ['manual', 'gpt4o', 'claude'] if s in stats and stats[s]]

    if not sources:
        print("No sources to compare against baseline")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(sources)

    colors = {
        'manual': '#2196F3',
        'gpt4o': '#FF9800',
        'claude': '#9C27B0'
    }

    for i, source in enumerate(sources):
        improvements = []
        for dataset in datasets:
            baseline_acc = stats[baseline].get(dataset, (0, 0))[0]
            source_acc = stats[source].get(dataset, (0, 0))[0]
            if baseline_acc > 0:
                improvement = source_acc - baseline_acc
            else:
                improvement = 0
            improvements.append(improvement)

        offset = (i - len(sources)/2 + 0.5) * width
        ax.bar(x + offset, improvements, width, label=source.capitalize(),
               color=colors.get(source, '#888888'), alpha=0.8)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(f'Accuracy Improvement vs {baseline.capitalize()} (%)', fontsize=12)
    ax.set_title('Relative Performance vs Template Baseline', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    improvement_path = output_path.replace('.pdf', '_improvement.pdf')
    plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
    print(f"Improvement plot saved to: {improvement_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze text source ablation experiments")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="checkpoints/ViT-B-32/text_adapted",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_text_source.pdf",
        help="Output path for visualization"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="templates",
        choices=["manual", "templates", "gpt4o", "claude"],
        help="Baseline text source for relative comparison"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TEXT SOURCE ABLATION ANALYSIS")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output: {args.output}")

    # Load results
    print("\nLoading results...")
    results = load_results(args.results_dir)
    print(f"Found {len(results)} experiment results")

    if not results:
        print("\nNo results found. Run experiments first with:")
        print("  python src/eval_text_adaptation.py --text-source manual ...")
        print("  python src/eval_text_adaptation.py --text-source templates ...")
        print("  python src/eval_text_adaptation.py --text-source generated --text-variant gpt4o ...")
        return

    # Aggregate by text source
    print("\nAggregating results by text source...")
    aggregated = aggregate_by_text_source(results)

    # Compute statistics
    stats = compute_statistics(aggregated)

    # Print summary table
    print_summary_table(stats)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_comparison(stats, args.output)
    plot_relative_improvement(stats, args.output, baseline=args.baseline)

    # Save summary JSON
    summary_path = args.output.replace('.pdf', '_summary.json')
    summary = {
        'text_sources': list(stats.keys()),
        'datasets': list(set(d for s in stats.values() for d in s.keys())),
        'results': {
            source: {
                dataset: {'mean': m, 'std': s}
                for dataset, (m, s) in datasets.items()
            }
            for source, datasets in stats.items()
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
