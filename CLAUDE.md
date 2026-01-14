# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **aTLAS** (Anisotropic Task-Level Anisotropic Scaling) from the NeurIPS 2024 paper "Knowledge Composition using Task Vectors with Learned Anisotropic Scaling". The project explores task vector composition for knowledge transfer in vision models, specifically using CLIP models with learned anisotropic scaling coefficients.

## Environment Setup

```bash
# Create conda environment with dependencies
conda env create -f environment.yml

# Add project to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/path/to/atlas"

# Activate environment
conda activate atlas
```

The environment uses:
- PyTorch 1.13.1 with CUDA 11.6
- Python 3.10
- open-clip-torch for CLIP models
- functorch for linearization

## Dataset and Checkpoint Preparation

Before running experiments:
1. Follow instructions in `DATASETS.md` to prepare datasets (default location: `~/data/`)
2. Download checkpoints from HuggingFace as described in `CHECKPOINTS.md`
3. Several datasets require manual setup (Cars, DTD, EuroSAT, RESISC45, SUN397, ImageNet, UCF101)

## Common Commands

### Running Experiments

**Task Negation:**
```bash
MODEL=ViT-B-32
python src/learn_task_negation.py --model=$MODEL --blockwise-coef
```

**Task Addition:**
```bash
MODEL=ViT-B-32
python src/learn_task_addition.py --model=$MODEL --blockwise-coef
```

**Few-Shot Adaptation:**
```bash
MODEL=ViT-B-32
# Basic aTLAS for different shot settings
for SHOT in 1 2 4 8 16; do
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT
done

# aTLAS with adapters (LP++ or Tip)
for SHOT in 1 2 4 8 16; do
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter tip
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter lpp
done
```

**Test-Time Adaptation:**
```bash
MODEL=ViT-B-32
python src/learn_ufm.py --model=$MODEL --blockwise-coef
```

**Parameter-Efficient Fine-Tuning (aTLAS x K):**
```bash
MODEL=ViT-B-32
PARTITION=10
for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0; do
    python src/learn_few_shots.py --model=$MODEL --partition $PARTITION --subsample $PERC
done
```

### Key Script Arguments

Common arguments across scripts (see `src/args.py` for full list):
- `--model`: Model architecture (ViT-B-32, ViT-B-16, ViT-L-14, RN50, RN101)
- `--blockwise-coef`: Enable learned coefficients per parameter block (core aTLAS feature)
- `--subsample`: Float for percentage or int for number of shots
- `--partition`: Number of random partitions for aTLAS x K
- `--adapter`: Use adapter with aTLAS (choices: tip, lpp, tip_cot)
- `--finetuning-mode`: standard or linear (for linearized models)
- `--data-location`: Root directory for datasets (default: ~/data)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 10)

## Architecture Overview

### Core Concepts

**Task Vectors**: The difference between fine-tuned and pretrained model weights. Can be added, negated, or scaled to compose knowledge from multiple tasks.

**Anisotropic Scaling**: Instead of using a single scalar coefficient for an entire task vector, aTLAS learns different coefficients for different parameter blocks, allowing more fine-grained control over knowledge composition.

### Key Components

**1. Task Vector Classes (`src/task_vectors.py`)**
- `NonLinearTaskVector`: Task vectors for standard (non-linearized) models
- `LinearizedTaskVector`: Task vectors for linearized models using first-order Taylor approximation
- Task vectors support arithmetic operations: addition, negation, scaling, dot product
- Created by subtracting pretrained weights from fine-tuned weights

**2. Weighted Composition Models (`src/composition.py`)**
- `WeightedImageEncoder`: Wraps a CLIP image encoder to enable learning coefficients for task vector composition
  - Uses `functorch.make_functional_with_buffers` to decompose model into parameters and function
  - Supports blockwise coefficients (one per parameter block) or global coefficients
  - Supports partitioning for aTLAS x K variant
- `WeightedLinearizedModel`: Similar wrapper for linearized models using first-order Taylor expansion

**3. Linearization (`src/linearize.py`)**
- `LinearizedModel`: Creates first-order Taylor approximation around initialization point
- Uses `functorch.jvp` (Jacobian-vector product) for efficient linearized forward pass
- Enables tangent task arithmetic for improved performance in some settings

**4. Model Components (`src/modeling.py`)**
- `ImageEncoder`: Wraps OpenCLIP models, loads pretrained CLIP encoders
- `ClassificationHead`: Linear classification layer with optional normalization
- `ImageClassifier`: Combines encoder and head for end-to-end classification
- `MultiHeadImageClassifier`: Supports multiple classification heads for multi-task scenarios

**5. Dataset Registry (`src/datasets/registry.py`)**
- Central registry of 22 datasets (CIFAR10/100, ImageNet, Cars, DTD, etc.)
- `get_dataset()`: Factory function that handles dataset loading and preprocessing
- Support for "Val" suffix to automatically split train into train/val
- `extract_class_data()`: Isolate specific classes from datasets
- Each dataset module in `src/datasets/` implements dataset-specific logic

**6. Training Scripts**
Main experiment scripts follow the pattern `learn_*.py`:
- `learn_few_shots.py`: Few-shot learning with task vector composition
- `learn_task_negation.py`: Learn coefficients for task negation
- `learn_task_addition.py`: Learn coefficients for task addition
- `learn_ufm.py`: Unsupervised test-time adaptation
- `finetune.py`: Standard or linearized fine-tuning

Evaluation scripts follow `eval_*.py` pattern for assessing different task arithmetic operations.

### Data Flow

1. **Checkpoint Creation**: Fine-tune CLIP models on individual tasks, save as checkpoints
2. **Task Vector Extraction**: Load pretrained and fine-tuned checkpoints, compute difference
3. **Weighted Composition**: Create `WeightedImageEncoder` with multiple task vectors
4. **Coefficient Learning**: Train only the composition coefficients (freezing base parameters)
5. **Evaluation**: Test composed model on target dataset

### Key Design Patterns

- **Blockwise vs. Global Coefficients**: `--blockwise-coef` flag controls whether each parameter block gets its own coefficient (more expressive) or a single global coefficient is used
- **Partitioning (aTLAS x K)**: When `--partition` is set, parameters within each block are randomly assigned to K partitions, each with its own coefficient
- **Distributed Training**: Scripts use `src/distributed.py` utilities for multi-GPU training via DDP
- **Mixed Precision**: Training uses automatic mixed precision (AMP) with GradScaler

### Checkpoint Organization

Checkpoints are stored under `checkpoints/<MODEL>/<DATASET>/`:
- `zeroshot.pt`: Pretrained CLIP encoder (zero-shot)
- `finetuned.pt`: Fine-tuned encoder on specific dataset
- `head_<DATASET>.pt`: Classification head for dataset
- `*_accuracies.json`: Accuracy results for different settings

Results from learning experiments are saved as JSON files in checkpoint directories (e.g., `learned_additions.json`, `learned_negations.json`).
