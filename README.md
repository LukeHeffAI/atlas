
<!-- > [!NOTE]
> The repository is still being cleaned up. More documentation will be released soon. -->

# Task Vectors with Learned Anisotropic Scaling

This repository contains and extends the official PyTorch implementation for the NeurIPS'24 paper
> Frederic Z. Zhang, Paul Albert, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad.
_Knowledge Composition using Task Vectors with Learned Anisotropic Scaling_.
In Advances in Neural Information Processing Systems (NeurIPS), 2024.

<a href="http://arxiv.org/abs/2407.02880">Preprint</a>

Extension details begin further in this document, "Extension: Text-Based Zero-Shot Adaptation" section. 

## Abstract
> ...<br/>This paper builds on properties of task vectors and aims to answer (1) whether components of task vectors, particularly parameter blocks, exhibit similar characteristics, and (2) how such blocks can be used to enhance knowledge composition and transfer. To this end, we introduce aTLAS, an algorithm that linearly combines parameter blocks with different learned coefficients, resulting in anisotropic scaling at the task vector level. We show that such linear combinations explicitly exploit the low intrinsic dimensionality of pre-trained models, with only a few coefficients being the learnable parameters. Furthermore, composition of parameter blocks enables modular learning that effectively leverages the already learned representations, thereby reducing the dependency on large amounts of data. We demonstrate the effectiveness of our method in task arithmetic, few-shot recognition and test-time adaptation, with supervised or unsupervised objectives. In particular, we show that (1) learned anisotropic scaling allows task vectors to be more disentangled, causing less interference in composition; (2) task vector composition excels with scarce or no labelled data and is less prone to domain shift, thus leading to better generalisability; (3) mixing the most informative parameter blocks across different task vectors prior to training can reduce the memory footprint and improve the flexibility of knowledge transfer. Moreover, we show the potential of aTLAS as a parameter-efficient fine-tuning method, particularly with less data, and demonstrate that it can be easily scaled up for higher performance.

<img src="./assets/teaser_a.png" height="300">&nbsp;&nbsp;<img src="./assets/teaser_b.png" height="300">

## Citation
If you find our work useful for your research, please consider citing us
```bibtex
@inproceedings{atlas_neurips_2024,
  title     = {Knowledge Composition using Task Vectors with Learned Anisotropic Scaling},
  author    = {Zhang, Frederic Z and Albert, Paul and Rodriguez-Opazo, Cristian and van den Hengel, Anton and Abbasnejad, Ehsan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

## Prerequisites
1. Create a `conda` environment and install the dependencies.
```
conda env create -f environment.yml
```
2. Add project directory to `PYTHONPATH` in `.bashrc`
```bash
export PYTHONPATH="$PYTHONPATH:/path/to/atlas"
```
3. Download and prepare the [datasets](./DATASETS.md).
4. Download and prepare the [checkpoints](./CHECKPOINTS.md) for task vectors.

## Reproducing experiment results

### Running Experiments with Scripts

For convenience, we provide shell scripts that automate common workflows:

```bash
# Task negation
./scripts/run_task_negation.sh [MODEL]

# Task addition
./scripts/run_task_addition.sh [MODEL]

# Few-shot adaptation (all shot settings)
./scripts/run_few_shot.sh [MODEL]

# Test-time adaptation
./scripts/run_test_time_adaptation.sh [MODEL]

# Parameter-efficient fine-tuning
./scripts/run_parameter_efficient.sh [MODEL] [PARTITION]

# Meta-train hypernetwork
./scripts/run_hypernetwork_training.sh [MODEL] [ARCH] [EPOCHS]

# Evaluate text adaptation
./scripts/run_text_evaluation.sh [MODEL] [DATASET] [APPROACH]

# Run all core experiments
./scripts/run_all_experiments.sh [MODEL]
```

All scripts default to `ViT-B-32` if no model is specified.

### 1. Task negation
```bash
python src/learn_task_negation.py --model=ViT-B-32 --blockwise-coef
```
Detailed performance is saved at `/path/to/atlas/checkpoints/ViT-B-32/learned_negations.json`.
### 2. Task addition
```bash
python src/learn_task_addition.py --model=ViT-B-32 --blockwise-coef
```
Detailed performance is saved at `/path/to/atlas/checkpoints/ViT-B-32/learned_additions.json`.
### 3. Few-shot adaptation
```bash
# aTLAS for different few-shot settings
for SHOT in 1 2 4 8 16;do
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise-coef --subsample $SHOT
done
# aTLAS with LP++ or Tip
for SHOT in 1 2 4 8 16;do
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise-coef --subsample $SHOT --adapter tip
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise-coef --subsample $SHOT --adapter lpp
done
```
### 4. Test-time adaptation
```bash
python src/learn_ufm.py --model=ViT-B-32 --blockwise-coef
```
### 5. Parameter-efficient fine-tuning
```bash
PARTITION=10
# aTLAS with K partitions using different percentage of data (aTLAS x K)
for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0;do
    python src/learn_few_shots.py --model=ViT-B-32 --partition $PARTITION --subsample $PERC
done
```

## Extension: Text-Based Zero-Shot Adaptation

This section documents recent extensions to the aTLAS framework for text-based zero-shot adaptation. The goal is to predict aTLAS coefficients directly from text descriptions of a target task, enabling adaptation without any task-specific images.

### Overview

Two complementary approaches are implemented:

1. **Hypernetwork-based** (Primary): A neural network that takes text descriptions and predicts optimal aTLAS coefficients through meta-learning
2. **Synthetic Data** (Complementary): Generate task-specific images from text using text-to-image models, then fine-tune to create synthetic task vectors

### Quick Start

```bash
# 1. Meta-train the hypernetwork on diverse tasks
python src/learn_text_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
    --meta-val-datasets Caltech101,Flowers102 \
    --hypernetwork-arch medium \
    --text-source manual \
    --meta-epochs 50 \
    --episodes-per-epoch 10 \
    --blockwise-coef

# 2. Evaluate on a new task using only text descriptions
python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset Cars \
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
    --text-source manual
```

### Text-to-Coefficient Hypernetwork

The hypernetwork learns to map text descriptions to aTLAS coefficients through meta-learning on a diverse set of tasks.

#### Architecture
- **Text encoder**: CLIP text encoder (frozen by default)
- **MLP layers**: Transform text embeddings to coefficient space
- **Output layer**: Produces blockwise or global coefficients

Architecture sizes:
| Size | Hidden Dimensions | Use Case |
|------|-------------------|----------|
| `small` | [512, 256] | Rapid prototyping, single GPU |
| `medium` | [512, 512, 256] | Production use, best balance |
| `large` | [768, 512, 512, 256] | Research, maximum performance |

#### Meta-Training
```bash
python src/learn_text_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
    --meta-val-datasets Caltech101,Flowers102 \
    --hypernetwork-arch medium \
    --text-source manual \
    --meta-epochs 50 \
    --episodes-per-epoch 10 \
    --blockwise-coef
```

Key arguments:
- `--meta-train-datasets`: Comma-separated list of training tasks
- `--meta-val-datasets`: Comma-separated list of validation tasks
- `--hypernetwork-arch`: Architecture size (small/medium/large)
- `--text-source`: Source of text descriptions (manual/templates/generated)
- `--text-aggregate`: How to aggregate multiple descriptions (mean/max/median)
- `--meta-epochs`: Number of meta-training epochs
- `--episodes-per-epoch`: Episodes per epoch
- `--meta-lr`: Meta-learning rate (default: 1e-4)
- `--meta-batch-size`: Batch size for episodes (default: 4)

#### Meta-Training Process
1. **Episode sampling**: Randomly select a task from meta-train datasets
2. **Coefficient prediction**: Hypernetwork predicts coefficients from text descriptions
3. **Evaluation**: Apply predicted coefficients to compose task vectors
4. **Gradient flow**: Backpropagate through predicted coefficients to update hypernetwork
5. **Validation**: Every 5 epochs, evaluate on meta-val datasets and save best model

### Text Descriptions

Text descriptions provide domain knowledge about each dataset's visual characteristics.

#### Directory Structure
```
data/text_descriptions/
├── manual/           # Human-written descriptions
│   ├── CIFAR10.json
│   ├── EuroSAT.json
│   └── ...
├── generated/        # LLM-generated descriptions
│   ├── gpt4o/
│   └── claude/
└── templates/        # CLIP-style templates (auto-generated)
```

#### Format
Each JSON file maps class names to lists of descriptions:
```json
{
  "dataset": "CIFAR10",
  "source": "manual",
  "descriptions": {
    "airplane": ["a photo of an airplane", "aircraft in the sky", ...],
    "automobile": ["a photo of a car", "vehicle on the road", ...]
  }
}
```

#### Available Descriptions
Manual descriptions are provided for: CIFAR10, EuroSAT, DTD, GTSRB, SVHN, Food101, Caltech101, Flowers102, Cars

#### Generating Descriptions with LLMs

```bash
# Generate with GPT-4o
python -m src.text_descriptions.generators \
    --dataset CIFAR10 \
    --classes airplane automobile bird cat deer dog frog horse ship truck \
    --llm gpt4o \
    --num-descriptions 15 \
    --diversity medium \
    --output-dir data/text_descriptions

# Generate with Claude
python -m src.text_descriptions.generators \
    --dataset CIFAR10 \
    --classes airplane automobile bird cat deer dog frog horse ship truck \
    --llm claude \
    --num-descriptions 15 \
    --diversity medium \
    --output-dir data/text_descriptions
```

**Requirements**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable.

### Synthetic Data Generation

Generate task-specific images from text descriptions using text-to-image models:

```bash
python src/generate_synthetic_data.py \
    --dataset CIFAR10 \
    --text-source manual \
    --t2i-backend stable_diffusion \
    --t2i-model stabilityai/stable-diffusion-xl-base-1.0 \
    --num-images-per-class 100 \
    --output-dir data/synthetic_images
```

Supported T2I backends:
- `stable_diffusion` / `sdxl`: Stable Diffusion XL (local, free)
- `dalle` / `dall-e`: DALL-E 3 (API, paid)

### Analysis Scripts

Scripts for analyzing experiment results are located in `scripts/`:

```bash
# Analyze hypernetwork architecture ablations
python scripts/analyze_hypernetwork_size.py \
    --results-dir checkpoints/ViT-B-32/hypernetwork \
    --output ablation_hypernetwork_size.pdf

# Analyze text source ablations (manual vs generated vs templates)
python scripts/analyze_text_ablation.py \
    --results-dir checkpoints/ViT-B-32/text_adapted \
    --output ablation_text_source.pdf

# Analyze synthetic image quality (FID, IS, LPIPS)
python scripts/analyze_synthetic_quality.py \
    --synthetic-dir data/synthetic_images \
    --real-dataset CIFAR10 \
    --output synthetic_quality_comparison.pdf
```

### Evaluation

Evaluate text-adapted models on held-out datasets:

```bash
# Hypernetwork approach
python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset Flowers102 \
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
    --text-source manual \
    --text-aggregate mean

# Synthetic approach
python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset Flowers102 \
    --approach synthetic \
    --synthetic-backend stable_diffusion

# Both approaches combined
python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset Flowers102 \
    --approach both \
    --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
    --synthetic-backend stable_diffusion
```

### New Dependencies

The text-based extensions require additional packages:
```bash
# Required for text encoders in hypernetwork
pip install transformers==4.30.0  # Compatible with PyTorch 1.13

# Optional: for LLM-based description generation
pip install openai      # For GPT-4o descriptions
pip install anthropic   # For Claude descriptions

# Optional: for DALL-E image generation
pip install openai      # DALL-E 3 API
```

### File Structure (New)

```
src/
├── hypernetworks/
│   ├── __init__.py
│   ├── base.py               # Abstract base class for hypernetworks
│   └── text_to_coef.py       # Text-to-coefficient hypernetwork
├── text_descriptions/
│   ├── __init__.py
│   ├── loaders.py            # Load/save descriptions from files
│   ├── generators.py         # LLM-based description generation
│   └── templates.py          # CLIP-style template generation
├── text2image/
│   ├── __init__.py
│   ├── base.py               # Abstract base for T2I backends
│   ├── stable_diffusion.py   # Stable Diffusion / SDXL backend
│   ├── dalle.py              # DALL-E 3 backend
│   └── registry.py           # Backend factory and registration
├── datasets/
│   └── synthetic.py          # Synthetic dataset loader
├── learn_text_to_coef.py     # Meta-training script
├── eval_text_adaptation.py   # Evaluation script
└── generate_synthetic_data.py # Synthetic image generation

data/
└── text_descriptions/
    ├── manual/               # Human-written descriptions
    │   ├── CIFAR10.json
    │   ├── EuroSAT.json
    │   └── ...
    └── generated/            # LLM-generated descriptions
        ├── CIFAR10_gpt4o.json
        └── CIFAR10_claude.json

scripts/
├── analyze_hypernetwork_size.py  # Architecture ablation analysis
├── analyze_text_ablation.py      # Text source comparison
└── analyze_synthetic_quality.py  # Synthetic image quality metrics
```

## Acknowledgement

This repository is largely based on the code provided by [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and [Ortiz-Jimenez et al. (2023)](https://github.com/gortizji/tangent_task_arithmetic).
