# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **aTLAS** (Anisotropic Task-Level Anisotropic Scaling) from the NeurIPS 2024 paper "Knowledge Composition using Task Vectors with Learned Anisotropic Scaling". The project explores task vector composition for knowledge transfer in vision models, specifically using CLIP models with learned anisotropic scaling coefficients.

**Recent Extension**: Text-based zero-shot adaptation via hypernetworks that predict aTLAS coefficients from text descriptions alone, enabling adaptation without task-specific images.

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
- transformers (for text encoders in hypernetwork)

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

**7. Hypernetwork Architecture (`src/hypernetworks/`)**
- `BaseHypernetwork`: Abstract base class defining the hypernetwork interface
- `TextToCoefHypernetwork`: Maps text descriptions → aTLAS coefficients
  - Uses CLIP text encoder (frozen) to encode descriptions
  - MLP layers transform text embeddings to coefficient space
  - Supports blockwise or global coefficients
  - Architecture sizes: `small` [512,256], `medium` [512,512,256], `large` [768,512,512,256]
- `create_hypernetwork_from_args()`: Factory function for creating hypernetworks from CLI args

**8. Text Description Management (`src/text_descriptions/`)**
- `TextDescriptionLoader`: Load/save text descriptions from JSON files
  - Supports manual and LLM-generated descriptions
  - Format: `{"class_name": ["desc1", "desc2", ...], ...}`
- `OpenAIDescriptionGenerator`: Generate descriptions using GPT-4o
- `ClaudeDescriptionGenerator`: Generate descriptions using Claude
- `templates.py`: CLIP-style template-based description generation

**9. Text-to-Image Generation (`src/text2image/`)**
- `Text2ImageBackend`: Abstract base for T2I backends
- `StableDiffusionBackend`: Stable Diffusion / SDXL implementation
- `DalleBackend`: DALL-E 3 implementation
- `get_t2i_backend()`: Factory function with backend registry
- `register_t2i_backend()`: Register custom T2I backends

**10. Text-Conditioned Composition (`src/composition.py`)**
- `TextConditionedWeightedImageEncoder`: Extends `WeightedImageEncoder` with hypernetwork support
  - Can use either learnable coefficients or hypernetwork-predicted coefficients
  - `enable_coefficient_finetuning()`: Convert predicted coefficients to trainable parameters
  - Supports coefficient pass-through for gradient flow during meta-training

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

## Text-Based Zero-Shot Adaptation

### Overview

The text-based adaptation extension enables predicting aTLAS coefficients from text descriptions alone, without requiring any task-specific images. This is achieved through:

1. **Hypernetwork-based**: Meta-train a hypernetwork to predict coefficients from text
2. **Synthetic Data**: Generate images from text using T2I models, then create synthetic task vectors

### Meta-Training Hypernetwork

```bash
MODEL=ViT-B-32
python src/learn_text_to_coef.py \
    --model $MODEL \
    --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
    --meta-val-datasets Caltech101,Flowers102 \
    --hypernetwork-arch medium \
    --text-source manual \
    --meta-epochs 50 \
    --episodes-per-epoch 10 \
    --blockwise-coef
```

### Evaluating Text Adaptation

```bash
MODEL=ViT-B-32
python src/eval_text_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/$MODEL/hypernetworks/text_to_coef/meta_trained.pt \
    --text-source manual
```

### Generating Text Descriptions

```bash
# Using GPT-4o
python -m src.text_descriptions.generators \
    --dataset CIFAR10 \
    --classes airplane automobile bird cat deer dog frog horse ship truck \
    --llm gpt4o \
    --num-descriptions 15

# Using Claude
python -m src.text_descriptions.generators \
    --dataset CIFAR10 \
    --classes airplane automobile bird cat deer dog frog horse ship truck \
    --llm claude \
    --num-descriptions 15
```

### Text Description Format

Text descriptions are stored in `data/text_descriptions/` as JSON files:
```json
{
  "dataset": "CIFAR10",
  "source": "manual",
  "descriptions": {
    "airplane": ["a photo of an airplane", "aircraft in flight", ...],
    "automobile": ["a photo of a car", "vehicle on the road", ...]
  }
}
```

### New CLI Arguments for Text Adaptation

**Text-to-Image arguments:**
- `--t2i-backend`: T2I backend (stable_diffusion, sdxl, dalle)
- `--t2i-model`: Specific model ID for T2I backend
- `--num-images-per-class`: Number of synthetic images per class (default: 100)

**Text description arguments:**
- `--text-source`: Source of descriptions (manual, generated, templates)
- `--text-variant`: Variant for generated text (gpt4o, claude)
- `--text-aggregate`: How to aggregate descriptions (mean, max, median)

**Hypernetwork arguments:**
- `--hypernetwork-arch`: Architecture size (small, medium, large)
- `--hypernetwork-checkpoint`: Path to pre-trained hypernetwork
- `--freeze-text-encoder`: Freeze text encoder during training (default: True)
- `--init-from-hypernetwork`: Initialize coefficients from hypernetwork predictions

**Meta-learning arguments:**
- `--meta-train-datasets`: Comma-separated training datasets
- `--meta-val-datasets`: Comma-separated validation datasets
- `--meta-lr`: Meta-learning rate (default: 1e-4)
- `--meta-epochs`: Number of meta-training epochs (default: 100)
- `--episodes-per-epoch`: Episodes per epoch (default: 20)
- `--meta-batch-size`: Batch size for meta-training episodes (default: 4)

**Text adaptation mode:**
- `--text-adaptation-mode`: Which approach to use (synthetic, hypernetwork, both)

## Analysis Scripts

Analysis and visualization scripts are located in `scripts/`:

```bash
# Analyze hypernetwork architecture ablations
python scripts/analyze_hypernetwork_size.py \
    --results-dir checkpoints/ViT-B-32/hypernetwork \
    --output ablation_hypernetwork_size.pdf

# Analyze text source ablations
python scripts/analyze_text_ablation.py \
    --results-dir checkpoints/ViT-B-32/text_adapted \
    --output ablation_text_source.pdf

# Analyze synthetic image quality
python scripts/analyze_synthetic_quality.py \
    --synthetic-dir data/synthetic_images \
    --output synthetic_quality_comparison.pdf
```

## Future Extensions

See `EXTENSION_ROADMAP.md` for detailed plans on:
1. Ablation studies (text sources, T2I backends, hypernetwork sizes)
2. Full LoRA prediction (predicting weight matrices instead of scalars)
3. Multi-modal hypernetwork (text + few example images)
4. Cross-model generalization (transfer across architectures)
5. Domain adaptation (medical, satellite, microscopy)
6. Task composition (composing multiple text descriptions)
