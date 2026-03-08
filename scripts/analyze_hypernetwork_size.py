"""Analyze hypernetwork architecture ablations.

This script analyzes and visualizes results from comparing different hypernetwork
architectures:
- Different hidden layer sizes
- Number of layers
- Attention mechanisms
- Text encoder variants

Usage:
    python scripts/analyze_hypernetwork_size.py \
        --results-dir checkpoints/ViT-B-32/hypernetwork \
        --output ablation_hypernetwork_size.pdf
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_training_logs(results_dir: str) -> Dict[str, Dict]:
    """Load training logs from hypernetwork experiments.

    Args:
        results_dir: Directory containing experiment results.

    Returns:
        Dictionary mapping experiment configs to training metrics.
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return results

    # Find all training log files
    for log_file in results_path.rglob("training_log.json"):
        try:
            with open(log_file) as f:
                data = json.load(f)

            # Extract experiment identifier from path
            rel_path = log_file.parent.relative_to(results_path)
            exp_name = str(rel_path)
            results[exp_name] = data

        except Exception as e:
            print(f"Error loading {log_file}: {e}")

    # Also look for result JSON files
    for json_file in results_path.rglob("*_results.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            exp_name = json_file.stem.replace('_results', '')
            if exp_name not in results:
                results[exp_name] = data

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def parse_config_from_name(exp_name: str) -> Dict[str, Any]:
    """Parse hypernetwork configuration from experiment name.

    Args:
        exp_name: Experiment name (e.g., "hidden_512_256/layers_2")

    Returns:
        Dictionary with parsed configuration
    """
    config = {
        'hidden_dims': [512, 256],  # default
        'num_layers': 2,
        'use_attention': False,
        'text_encoder': 'clip',
        'dropout': 0.1
    }

    parts = exp_name.lower().split('/')

    for part in parts:
        # Parse hidden dimensions
        if 'hidden' in part:
            dims = []
            tokens = part.replace('hidden_', '').split('_')
            for t in tokens:
                try:
                    dims.append(int(t))
                except ValueError:
                    pass
            if dims:
                config['hidden_dims'] = dims

        # Parse number of layers
        if 'layer' in part:
            try:
                num = int(part.split('_')[-1])
                config['num_layers'] = num
            except ValueError:
                pass

        # Parse attention
        if 'attention' in part or 'attn' in part:
            config['use_attention'] = True

        # Parse text encoder
        if 'bert' in part:
            config['text_encoder'] = 'bert'
        elif 'sentence' in part or 'sbert' in part:
            config['text_encoder'] = 'sentence-transformers'

        # Parse dropout
        if 'dropout' in part or 'drop' in part:
            try:
                val = float(part.split('_')[-1])
                config['dropout'] = val
            except ValueError:
                pass

    return config


def compute_parameter_count(config: Dict) -> int:
    """Estimate parameter count for hypernetwork configuration.

    Args:
        config: Hypernetwork configuration

    Returns:
        Estimated parameter count
    """
    # Text encoder output dimension (CLIP default)
    text_dim = 512

    # Hidden dimensions
    hidden_dims = config.get('hidden_dims', [512, 256])

    # Output dimension (21 task vectors * ~150 blocks typical)
    output_dim = 21 * 150  # rough estimate

    # Count parameters
    total = 0

    # First layer
    if hidden_dims:
        total += text_dim * hidden_dims[0] + hidden_dims[0]  # weight + bias

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            total += hidden_dims[i] * hidden_dims[i+1] + hidden_dims[i+1]

        # Output layer
        total += hidden_dims[-1] * output_dim + output_dim
    else:
        total += text_dim * output_dim + output_dim

    # Attention layers (if used)
    if config.get('use_attention', False):
        # Rough estimate for attention mechanism
        d_model = hidden_dims[0] if hidden_dims else text_dim
        total += 4 * d_model * d_model  # Q, K, V, output projections

    return total


def extract_metrics(results: Dict) -> Dict[str, Dict[str, float]]:
    """Extract key metrics from results.

    Args:
        results: Raw results dictionary

    Returns:
        Dictionary mapping experiment names to metrics
    """
    metrics = {}

    for exp_name, data in results.items():
        config = parse_config_from_name(exp_name)
        param_count = compute_parameter_count(config)

        exp_metrics = {
            'param_count': param_count,
            'hidden_dims': config['hidden_dims'],
            'num_layers': len(config['hidden_dims']),
            'use_attention': config['use_attention'],
            'text_encoder': config['text_encoder']
        }

        # Extract accuracy metrics
        if isinstance(data, dict):
            if 'final_accuracy' in data:
                exp_metrics['accuracy'] = data['final_accuracy']
            elif 'test_accuracy' in data:
                exp_metrics['accuracy'] = data['test_accuracy']
            elif 'top1' in data:
                exp_metrics['accuracy'] = data['top1']

            if 'training_loss' in data:
                exp_metrics['final_loss'] = data['training_loss'][-1] if isinstance(data['training_loss'], list) else data['training_loss']

            if 'epochs' in data:
                exp_metrics['epochs'] = data['epochs']

            if 'training_time' in data:
                exp_metrics['training_time'] = data['training_time']

        metrics[exp_name] = exp_metrics

    return metrics


def print_summary_table(metrics: Dict):
    """Print formatted summary table.

    Args:
        metrics: Extracted metrics dictionary
    """
    print("\n" + "=" * 90)
    print("HYPERNETWORK ARCHITECTURE ABLATION RESULTS")
    print("=" * 90)

    # Sort by accuracy if available
    sorted_exps = sorted(
        metrics.keys(),
        key=lambda x: metrics[x].get('accuracy', 0),
        reverse=True
    )

    # Print header
    print(f"{'Configuration':<35} {'Hidden Dims':<20} {'Params':<12} {'Accuracy':<10}")
    print("-" * 90)

    for exp_name in sorted_exps:
        m = metrics[exp_name]
        hidden_str = str(m.get('hidden_dims', []))
        param_str = f"{m.get('param_count', 0):,}"
        acc_str = f"{m.get('accuracy', 0):.2f}%" if 'accuracy' in m else "--"

        print(f"{exp_name:<35} {hidden_str:<20} {param_str:<12} {acc_str:<10}")

    print("=" * 90)


def plot_size_vs_accuracy(metrics: Dict, output_path: str):
    """Plot parameter count vs accuracy.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save plot
    """

    # Extract data points
    param_counts = []
    accuracies = []
    labels = []

    for exp_name, m in metrics.items():
        if 'accuracy' in m and 'param_count' in m:
            param_counts.append(m['param_count'])
            accuracies.append(m['accuracy'])
            labels.append(exp_name)

    if not param_counts:
        print("No data points to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    scatter = ax.scatter(param_counts, accuracies, s=100, alpha=0.7, c='#2196F3')

    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (param_counts[i], accuracies[i]),
                   textcoords="offset points", xytext=(5, 5),
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('Parameter Count', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Hypernetwork Size vs Performance', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_architecture_comparison(metrics: Dict, output_path: str):
    """Create architecture comparison bar chart.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save plot
    """

    # Group by architecture type
    architectures = {
        'small': [],    # < 1M params
        'medium': [],   # 1M - 10M params
        'large': [],    # > 10M params
        'attention': [] # with attention
    }

    for exp_name, m in metrics.items():
        if 'accuracy' not in m:
            continue

        acc = m['accuracy']
        params = m.get('param_count', 0)

        if m.get('use_attention', False):
            architectures['attention'].append(acc)
        elif params < 1_000_000:
            architectures['small'].append(acc)
        elif params < 10_000_000:
            architectures['medium'].append(acc)
        else:
            architectures['large'].append(acc)

    # Filter out empty categories
    arch_names = []
    arch_means = []
    arch_stds = []

    for name, accs in architectures.items():
        if accs:
            arch_names.append(name.capitalize())
            arch_means.append(np.mean(accs))
            arch_stds.append(np.std(accs) if len(accs) > 1 else 0)

    if not arch_names:
        print("No architecture data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(arch_names))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

    bars = ax.bar(x, arch_means, yerr=arch_stds, capsize=5,
                  color=colors[:len(arch_names)], alpha=0.8)

    ax.set_xlabel('Architecture Type', fontsize=12)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Performance by Hypernetwork Architecture', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, arch_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    arch_path = output_path.replace('.pdf', '_architecture.pdf')
    plt.savefig(arch_path, dpi=150, bbox_inches='tight')
    print(f"Architecture comparison saved to: {arch_path}")
    plt.close()


def plot_hidden_dim_analysis(metrics: Dict, output_path: str):
    """Analyze effect of hidden dimension sizes.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save plot
    """

    # Extract hidden dim info
    first_hidden = []
    accuracies = []

    for exp_name, m in metrics.items():
        if 'accuracy' not in m:
            continue

        hidden_dims = m.get('hidden_dims', [])
        if hidden_dims:
            first_hidden.append(hidden_dims[0])
            accuracies.append(m['accuracy'])

    if not first_hidden:
        print("No hidden dimension data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(first_hidden, accuracies, s=100, alpha=0.7, c='#FF9800')

    ax.set_xlabel('First Hidden Layer Size', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Effect of Hidden Layer Size on Performance', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    hidden_path = output_path.replace('.pdf', '_hidden_dims.pdf')
    plt.savefig(hidden_path, dpi=150, bbox_inches='tight')
    print(f"Hidden dimension analysis saved to: {hidden_path}")
    plt.close()


def generate_recommendations(metrics: Dict) -> List[str]:
    """Generate architecture recommendations based on results.

    Args:
        metrics: Metrics dictionary

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if not metrics:
        return ["No results available for recommendations"]

    # Find best performing
    best_acc = 0
    best_config = None

    for exp_name, m in metrics.items():
        if m.get('accuracy', 0) > best_acc:
            best_acc = m['accuracy']
            best_config = exp_name

    if best_config:
        recommendations.append(f"Best performing configuration: {best_config} ({best_acc:.2f}%)")

    # Analyze trends
    small_accs = []
    large_accs = []

    for exp_name, m in metrics.items():
        if 'accuracy' not in m:
            continue

        params = m.get('param_count', 0)
        if params < 5_000_000:
            small_accs.append(m['accuracy'])
        else:
            large_accs.append(m['accuracy'])

    if small_accs and large_accs:
        small_avg = np.mean(small_accs)
        large_avg = np.mean(large_accs)

        if large_avg > small_avg + 1:
            recommendations.append("Larger architectures show meaningful improvement")
        elif small_avg > large_avg - 1:
            recommendations.append("Smaller architectures perform comparably - consider efficiency")

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze hypernetwork architecture ablations")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="checkpoints/ViT-B-32/hypernetwork",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_hypernetwork_size.pdf",
        help="Output path for visualization"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HYPERNETWORK ARCHITECTURE ABLATION ANALYSIS")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output: {args.output}")

    # Load results
    print("\nLoading training logs...")
    results = load_training_logs(args.results_dir)
    print(f"Found {len(results)} experiment results")

    if not results:
        print("\nNo results found. Run hypernetwork experiments first with any/all:")
        print("  python src/learn_text_to_coef.py --hidden-dims 256 128 ...")
        print("  python src/learn_text_to_coef.py --hidden-dims 512 256 ...")
        print("  python src/learn_text_to_coef.py --hidden-dims 1024 512 256 ...")
        return

    # Extract metrics
    print("\nExtracting metrics...")
    metrics = extract_metrics(results)

    # Print summary
    print_summary_table(metrics)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_size_vs_accuracy(metrics, args.output)
    plot_architecture_comparison(metrics, args.output)
    plot_hidden_dim_analysis(metrics, args.output)

    # Generate recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)
    recommendations = generate_recommendations(metrics)
    for rec in recommendations:
        print(f"  - {rec}")

    # Save summary
    summary_path = args.output.replace('.pdf', '_summary.json')
    summary = {
        'num_experiments': len(metrics),
        'metrics': {
            name: {k: v for k, v in m.items() if not isinstance(v, list)}
            for name, m in metrics.items()
        },
        'recommendations': recommendations
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
