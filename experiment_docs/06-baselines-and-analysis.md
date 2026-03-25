# 06 — Baselines and Analysis

## Overview

This document covers the evaluation scripts that provide baselines for comparison and the analysis tools for understanding results. These experiments don't introduce new methods — instead, they measure the performance of simpler approaches (scalar scaling, grid search) and provide diagnostic tools (task vector similarity, per-class analysis) that inform the interpretation of all other experiments.

## Baseline Evaluations

### Single-Task Scaling (`src/eval_single_task.py`)

**Scientific question**: How well does a simple scalar coefficient work for a single task vector?

**Method**: For each dataset, sweep a scalar coefficient $\alpha$ from 0 to some maximum (default: 21 evenly-spaced points) and evaluate accuracy. Find the optimal $\alpha$ on the validation set, then report test accuracy.

$$\theta_{\text{scaled}} = \theta_{\text{pretrained}} + \alpha \cdot \tau_D$$

This is the simplest possible task vector application and serves as the lower bound for what aTLAS should beat.

```bash
MODEL=ViT-B-32
python src/eval_single_task.py --model $MODEL --finetuning-mode standard
python src/eval_single_task.py --model $MODEL --finetuning-mode linear
python src/eval_single_task.py --model $MODEL --finetuning-mode posthoc
```

**Output**: `checkpoints/<MODEL>/ft_accuracies.json` (or `linear_ft_accuracies.json`, `posthoc_ft_accuracies.json`)

### Task Addition Baseline (`src/eval_task_addition.py`)

**Method**: Sum all task vectors with a single shared scalar, sweep scalar value, report multi-task accuracy.

```bash
python src/eval_task_addition.py --model $MODEL --finetuning-mode standard
```

**Output**: `checkpoints/<MODEL>/additions.json`

### Task Negation Baseline (`src/eval_task_negation.py`)

**Method**: Negate a task vector with a scalar, sweep scalar value, maintain control accuracy threshold.

```bash
python src/eval_task_negation.py --model $MODEL --finetuning-mode standard
```

**Output**: `checkpoints/<MODEL>/negations.json`

## Diagnostic Analyses

### Task Vector Orthogonality (`src/eval_orthogonality.py`)

**Scientific question**: Are task vectors from different tasks approximately orthogonal? This is a key theoretical assumption supporting task arithmetic — if vectors are orthogonal, their compositions should not interfere.

**Method**: Compute the cosine similarity matrix between task vectors from single-class MNIST variants (digits 0–9). Visualize as a heatmap.

```bash
python src/eval_orthogonality.py --model $MODEL --finetuning-mode standard
```

**Output**:
- `checkpoints/<MODEL>/task_vector_sim.json` — cosine similarity matrix
- `checkpoints/<MODEL>/task_vector_sim.png` — heatmap visualization

**Interpretation**: Off-diagonal values near 0 indicate approximate orthogonality. High off-diagonal values suggest interference risk when composing those task vectors.

### Per-Class Task Vector Evaluation (`src/eval_class_task_vector.py`)

**Scientific question**: How well do task vectors work at the *individual class* level?

**Method**: Fine-tune CLIP on individual MNIST digits, create per-class task vectors, evaluate how well each transfers to the full dataset.

```bash
python src/eval_class_task_vector.py --model $MODEL --finetuning-mode standard
```

**Output**: `checkpoints/<MODEL>/class_ft_accuracies.json`

## Analysis and Visualization Scripts

All scripts are in `scripts/` and produce publication-quality plots.

### Hypernetwork Architecture Ablation (`scripts/analyze_hypernetwork_size.py`)

Compare different hypernetwork sizes (small, medium, large) in terms of accuracy vs. parameter count.

```bash
python scripts/analyze_hypernetwork_size.py \
    --results-dir checkpoints/ViT-B-32/hypernetwork \
    --output ablation_hypernetwork_size.pdf
```

**Generates**: Architecture comparison plots, parameter count analysis, training convergence curves.

### Text Source Ablation (`scripts/analyze_text_ablation.py`)

Compare different text description sources: manual, CLIP templates, GPT-4o generated, Claude generated.

```bash
python scripts/analyze_text_ablation.py \
    --results-dir checkpoints/ViT-B-32/text_adapted \
    --output ablation_text_source.pdf
```

**Generates**: Per-source accuracy comparison, relative improvement over templates, per-dataset breakdown.

### Synthetic Image Quality (`scripts/analyze_synthetic_quality.py`)

Evaluate the quality of synthetic images from different text-to-image backends.

```bash
python scripts/analyze_synthetic_quality.py \
    --synthetic-dir data/synthetic_images \
    --real-dataset CIFAR10 \
    --metrics fid,is,lpips \
    --output synthetic_quality_comparison.pdf
```

**Metrics**: FID (Frechet Inception Distance), IS (Inception Score), LPIPS (perceptual similarity), task vector cosine similarity.

### Few-Shot Benchmark Visualization (`scripts/analyze_synthetic_benchmark.py`)

Produce grouped bar charts comparing zero-shot, aTLAS at various shot counts, and fine-tuned performance.

```bash
python scripts/analyze_synthetic_benchmark.py --model ViT-B-32 --output-dir results/analysis
```

**Generates**: Per-model grouped bar charts, cross-model comparison, accuracy vs. shot count curves.

### Multi-Modal vs. Text-Only Comparison (`scripts/compare_multimodal_vs_text.py`)

Publication-quality plots comparing multi-modal and text-only adaptation.

```bash
python scripts/compare_multimodal_vs_text.py \
    --multimodal-results checkpoints/ViT-B-32/multimodal_adapted/ \
    --textonly-results checkpoints/ViT-B-32/text_adapted/ \
    --datasets Flowers102,Cars,DTD \
    --shot-count 4 \
    --output figures/multimodal_vs_text.pdf
```

**Generates**: Shot-count sweep, per-dataset bar chart, improvement heatmap.

## How These Baselines Fit into the Paper

| Baseline | Compared Against | Key Claim |
|----------|-----------------|-----------|
| Scalar scaling (eval_single_task) | Blockwise aTLAS | Anisotropic > isotropic |
| Scalar addition (eval_task_addition) | Learned addition | Learning coefficients improves multi-task composition |
| Scalar negation (eval_task_negation) | Learned negation | Blockwise negation is more precise |
| Zero-shot CLIP | All methods | Task vectors add value over pretrained features |
| Fine-tuned (per-dataset) | Few-shot aTLAS | Upper bound reference |
| Orthogonality analysis | — | Supports theoretical motivation for task arithmetic |
