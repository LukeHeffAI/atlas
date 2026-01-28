
<!-- > [!NOTE]
> The repository is still being cleaned up. More documentation will be released soon. -->

# Task Vectors with Learned Anisotropic Scaling

This repository contains the official PyTorch implementation for the NeurIPS'24 paper
> Frederic Z. Zhang, Paul Albert, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad.
_Knowledge Composition using Task Vectors with Learned Anisotropic Scaling_.
In Advances in Neural Information Processing Systems (NeurIPS), 2024.

<a href="http://arxiv.org/abs/2407.02880">Preprint</a>

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

### 1. Task negation
```bash
MODEL=ViT-B-32
python src/learn_task_negation.py --model=$MODEL --blockwise-coef 
```
Detailed performance is saved at `/path/to/atlas/checkpoints/$MODEL/learned_negations.json`.
### 2. Task addition
```bash
MODEL=ViT-B-32
python src/learn_task_addition.py --model=$MODEL --blockwise-coef 
```
Detailed performance is saved at `/path/to/atlas/checkpoints/$MODEL/learned_additions.json`.
### 3. Few-shot adaptation
```bash
MODEL=ViT-B-32
# aTLAS for different few-shot settings
for SHOT in 1 2 4 8 16;do
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT
done
# aTLAS with LP++ or Tip
for SHOT in 1 2 4 8 16;do
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter tip
    python src/learn_few_shots.py --model=$MODEL --blockwise-coef --subsample $SHOT --adapter lpp
done
```
### 4. Test-time adaptation
```bash
MODEL=ViT-B-32
python src/learn_ufm.py --model=$MODEL --blockwise-coef
```
### 5. Parameter-efficient fine-tuning
```bash
MODEL=ViT-B-32
PARTITION=10
# aTLAS with K partitions using different percentage of data (aTLAS x K)
for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0;do
    python src/learn_few_shots.py --model=$MODEL --partition $PATITION --subsample $PERC
done
```

## Extension: Text-Based Zero-Shot Adaptation

This section documents recent extensions to the aTLAS framework for text-based zero-shot adaptation. The goal is to predict aTLAS coefficients directly from text descriptions of a target task, enabling adaptation without any task-specific images.

### Overview

Two complementary approaches are implemented:

1. **Hypernetwork-based**: A neural network that takes text descriptions and predicts optimal aTLAS coefficients
2. **Synthetic Data**: Generate task-specific images from text using text-to-image models, then fine-tune to create synthetic task vectors

### Text-to-Coefficient Hypernetwork

The hypernetwork learns to map text descriptions to aTLAS coefficients through meta-learning on a diverse set of tasks.

#### Architecture
- Text encoder: CLIP text encoder (frozen)
- MLP: Predicts blockwise coefficients from text embeddings
- Architectures: `small` (256,128), `medium` (512,256), `large` (1024,512,256)

#### Meta-Training
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

Key arguments:
- `--meta-train-datasets`: Comma-separated list of training tasks
- `--meta-val-datasets`: Comma-separated list of validation tasks
- `--hypernetwork-arch`: Architecture size (small/medium/large)
- `--text-source`: Source of text descriptions (manual/templates/generated)
- `--meta-epochs`: Number of meta-training epochs
- `--episodes-per-epoch`: Episodes per epoch

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

### Analysis Scripts

Scripts for analyzing experiment results:

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

### New Dependencies

The text-based extensions require additional packages:
```bash
pip install transformers==4.30.0  # For text encoders (compatible with PyTorch 1.13)
```

### File Structure (New)

```
src/
├── hypernetworks/
│   ├── __init__.py
│   └── text_to_coef.py       # Hypernetwork architecture
├── text_descriptions/
│   ├── __init__.py
│   ├── loaders.py            # Load descriptions from files
│   └── templates.py          # Generate CLIP-style templates
├── text2image/               # Synthetic image generation (planned)
├── learn_text_to_coef.py     # Meta-training script
└── eval_text_adaptation.py   # Evaluation script

data/
└── text_descriptions/
    ├── manual/               # Human-written descriptions
    └── generated/            # LLM-generated descriptions

scripts/
├── analyze_hypernetwork_size.py
├── analyze_text_ablation.py
└── analyze_synthetic_quality.py
```

## Acknowledgement

This repository is largely based on the code provided by [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and [Ortiz-Jimenez et al. (2023)](https://github.com/gortizji/tangent_task_arithmetic).
