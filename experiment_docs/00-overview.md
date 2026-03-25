# aTLAS Experiment Overview

## What is aTLAS?

**aTLAS** (Anisotropic Task-Level Anisotropic Scaling) is a method for composing knowledge across vision tasks by manipulating **task vectors** — the weight-space difference between a fine-tuned model and its pretrained initialization. The core insight is that a single global scaling coefficient for a task vector is suboptimal; instead, aTLAS learns **per-block anisotropic coefficients**, giving each parameter group (e.g., each attention layer) its own scaling factor. This enables more precise control over how much task-specific knowledge to inject, remove, or combine.

The method builds on CLIP vision-language models and supports both standard and linearized (first-order Taylor approximation) fine-tuning modes.

## Experiment Dependency Graph

```
                    ┌─────────────────┐
                    │  01 Fine-Tuning  │
                    │ (create task     │
                    │  vectors)        │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌─────────────┐  ┌──────────────┐
     │ 02 Task    │  │ 03 Few-Shot │  │ 05 Test-Time │
     │ Arithmetic │  │ Adaptation  │  │ Adaptation   │
     └────────────┘  └──────┬──────┘  └──────────────┘
                            │
                     ┌──────┴──────┐
                     │             │
                     ▼             ▼
              ┌───────────┐ ┌───────────────┐
              │ 04 aTLAS  │ │ 07 Text-to-   │
              │ x K       │ │ Coefficient   │──────┐
              └───────────┘ └───────────────┘      │
                                                   ▼
                                            ┌───────────────┐
                                            │ 08 Multi-     │
                                            │ Modal Hyper-  │
                                            │ network       │
                                            └───────────────┘

        06 Baselines & Analysis ← (supports all experiments)
```

All experiments require task vectors from step 01. The text-based extensions (07, 08) additionally require text descriptions and/or synthetic data infrastructure.

## Models and Datasets

### Supported CLIP Architectures

| Model | Params | Notes |
|-------|--------|-------|
| ViT-B-32 | 88M | Primary model, fastest experiments |
| ViT-B-16 | 86M | Higher resolution patches |
| ViT-L-14 | 304M | Largest ViT variant |
| RN50 | 38M | ResNet backbone |
| RN101 | 56M | Deeper ResNet |

### Dataset Pool (22 datasets)

| Category | Datasets |
|----------|----------|
| General | CIFAR10, CIFAR100, STL10, ImageNet, Caltech101, Caltech256 |
| Fine-grained | Cars, FGVCAircraft, Flowers102, OxfordIIITPet, CUB200, Food101 |
| Texture/Scene | DTD, SUN397, Country211 |
| Remote Sensing | EuroSAT, RESISC45 |
| Specialized | GTSRB, MNIST, SVHN, UCF101, PascalVOC |

All datasets default to `~/data/`. See `DATASETS.md` for preparation instructions.

## Shared Configuration

All experiment scripts share a common argument parser (`src/args.py`). Key shared flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | CLIP architecture | ViT-B-32 |
| `--blockwise-coef` | Enable per-block coefficients (core aTLAS feature) | off |
| `--finetuning-mode` | standard, linear, posthoc, none | standard |
| `--data-location` | Root directory for datasets | ~/data |
| `--batch-size` | Training batch size | 128 |
| `--lr` | Learning rate | 0.001 |
| `--epochs` | Training epochs | 10 |
| `--seed` | Random seed | 0 |

## Checkpoint and Results Organization

```
checkpoints/<MODEL>/
├── <DATASET>Val/
│   ├── zeroshot.pt              # Pretrained CLIP weights
│   ├── finetuned.pt             # Fine-tuned weights
│   └── head_<DATASET>.pt       # Classification head
├── learned_additions.json       # Task addition results
├── learned_negations.json       # Task negation results
├── hypernetworks/
│   ├── text_to_coef/            # Text hypernetwork checkpoints
│   └── multimodal_to_coef/      # Multi-modal checkpoints
├── text_adapted/                # Text adaptation results
└── multimodal_adapted/          # Multi-modal results
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate atlas
export PYTHONPATH="$PYTHONPATH:/path/to/atlas"
```

Key dependencies: PyTorch 1.13.1 (CUDA 11.6), open-clip-torch, functorch, transformers.

## Experiment Index

| # | Experiment | Document | Script(s) | One-Line Summary |
|---|-----------|----------|-----------|-----------------|
| 01 | Fine-Tuning | [01-finetuning.md](01-finetuning.md) | `finetune.py` | Create task vectors by fine-tuning CLIP on individual datasets |
| 02 | Task Arithmetic | [02-task-arithmetic.md](02-task-arithmetic.md) | `learn_task_{addition,negation}.py` | Add or negate task vectors with learned anisotropic coefficients |
| 03 | Few-Shot Adaptation | [03-few-shot-adaptation.md](03-few-shot-adaptation.md) | `learn_few_shots.py` | K-shot learning via task vector composition |
| 04 | aTLAS x K | [04-parameter-efficient-finetuning.md](04-parameter-efficient-finetuning.md) | `learn_few_shots.py --partition` | Partitioned coefficients for parameter-efficient fine-tuning |
| 05 | Test-Time Adaptation | [05-test-time-adaptation.md](05-test-time-adaptation.md) | `learn_ufm.py` | Unsupervised adaptation using only test data |
| 06 | Baselines & Analysis | [06-baselines-and-analysis.md](06-baselines-and-analysis.md) | `eval_*.py`, `scripts/` | Baseline evaluations, orthogonality analysis, visualizations |
| 07 | Text-to-Coefficient | [07-text-to-coefficient.md](07-text-to-coefficient.md) | `learn_text_to_coef.py` | Predict aTLAS coefficients from text descriptions alone |
| 08 | Multi-Modal Hypernetwork | [08-multimodal-hypernetwork.md](08-multimodal-hypernetwork.md) | `learn_multimodal_to_coef.py` | Combine text + support images for coefficient prediction |
