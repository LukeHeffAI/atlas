# 02 — Task Arithmetic: Addition and Negation

## Scientific Question

Can we compose or remove knowledge from pretrained models by performing arithmetic operations on task vectors in weight space? Specifically:

- **Task Addition**: Can we create a single model that performs well on multiple tasks simultaneously by *adding* their task vectors with learned coefficients?
- **Task Negation**: Can we *unlearn* specific knowledge (e.g., make a model forget a particular dataset) by *negating* a task vector, while preserving general capabilities?

The aTLAS contribution is that learning **anisotropic (per-block) coefficients** for these operations significantly outperforms using a single global scalar.

## Method Overview

### Task Addition

Given $N$ task vectors $\{\tau_1, \ldots, \tau_N\}$ from $N$ datasets, we compose a multi-task model:

$$\theta_{\text{multi}} = \theta_{\text{pretrained}} + \sum_{i=1}^{N} \alpha_i \odot \tau_i$$

where $\alpha_i$ are learnable coefficient vectors (one scalar per parameter block when using `--blockwise-coef`). We train the coefficients on all $N$ datasets simultaneously, using a `MultiHeadImageClassifier` with one classification head per dataset. The loss is cross-entropy summed across all tasks, and accuracy is **normalized** by each task's fine-tuning accuracy to account for varying difficulty.

The default task pool is 8 datasets: Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN.

### Task Negation

For negation, we want to make the model *forget* a target dataset $D_t$ while preserving performance on a control dataset (ImageNet). The composed model is:

$$\theta_{\text{negated}} = \theta_{\text{pretrained}} + \alpha \odot (-\tau_{D_t})$$

Training uses **gradient ascent on the target** (maximize loss → forget) and **gradient descent on the control** (minimize loss → preserve). A **control threshold** (default 0.95) ensures we don't degrade general performance too much — training stops early if ImageNet accuracy drops below 95% of its original value.

Each dataset has hand-tuned hyperparameters (learning rate multiplier, number of epochs) since the difficulty of negation varies widely across tasks.

## Relationship to Other Experiments

- **Prerequisites**: [01-finetuning.md](01-finetuning.md) — requires fine-tuned checkpoints for all target datasets.
- **Baselines from**: [06-baselines-and-analysis.md](06-baselines-and-analysis.md) — `eval_task_addition.py` and `eval_task_negation.py` provide the scalar (non-learned) baselines via grid search.
- **Extended by**: [03-few-shot-adaptation.md](03-few-shot-adaptation.md) — few-shot learning uses the same composition framework but with a different training objective.

## Key Implementation Details

**Scripts**:
- `src/learn_task_addition.py` — learn additive coefficients across 8 datasets
- `src/eval_task_addition.py` — baseline: grid search over a single global scalar
- `src/learn_task_negation.py` — learn negation coefficients per dataset
- `src/eval_task_negation.py` — baseline: grid search over negation scalar

**Key classes**:
- `WeightedImageEncoder` (`src/composition.py`) — wraps encoder with learnable coefficients
- `MultiHeadImageClassifier` (`src/modeling.py`) — multi-task classifier with per-dataset heads

**Task Addition hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-3 |
| Batch size | 128 (split across datasets) |
| Epochs | 20 |
| Regularization | Optional L1/L2 via `--lp-reg` |

**Task Negation per-dataset hyperparameters** (examples):
| Dataset | Epochs | LR Multiplier |
|---------|--------|---------------|
| Cars | 20 | 5x |
| DTD | 20 | 10x |
| EuroSAT | 3 | 5x |
| MNIST | 5 | 1x |

## How to Run

### Prerequisites
- Fine-tuned checkpoints for all target datasets (from [01-finetuning.md](01-finetuning.md))
- Classification heads saved as `head_<DATASET>.pt`

### Task Addition

```bash
MODEL=ViT-B-32

# Learn blockwise coefficients for multi-task composition
python src/learn_task_addition.py --model $MODEL --blockwise-coef

# Baseline: grid search over global scalar
python src/eval_task_addition.py --model $MODEL --finetuning-mode standard
```

### Task Negation

```bash
MODEL=ViT-B-32

# Learn per-dataset negation coefficients
python src/learn_task_negation.py --model $MODEL --blockwise-coef

# Baseline: grid search over negation scalar
python src/eval_task_negation.py --model $MODEL --finetuning-mode standard
```

### Variations

```bash
# Linearized models
python src/learn_task_addition.py --model $MODEL --blockwise-coef --finetuning-mode linear

# With L1 regularization on coefficients
python src/learn_task_addition.py --model $MODEL --blockwise-coef --lp-reg 1
```

## Expected Outputs

**Task Addition**:
- `checkpoints/<MODEL>/learned_additions.pt` — best coefficient checkpoint
- `checkpoints/<MODEL>/learned_additions.json` — validation and test metrics:
  ```json
  {
    "val": {"Cars": 0.82, "DTD": 0.71, ...},
    "test": {"Cars": 0.81, "DTD": 0.70, ...},
    "normalized_accuracy": 0.89
  }
  ```

**Task Negation**:
- `checkpoints/<MODEL>/<DATASET>/learned_negations.pt` — per-dataset coefficients
- `checkpoints/<MODEL>/learned_negations.json` — per-dataset results:
  ```json
  {
    "Cars": {"target_acc": 0.05, "control_acc": 0.96, "control_threshold": 0.95},
    "DTD": {"target_acc": 0.08, "control_acc": 0.95, ...}
  }
  ```

**What "good" looks like**:
- Addition: Normalized accuracy >0.85 with blockwise coefficients (vs ~0.75 for global scalar).
- Negation: Target accuracy drops to near-random while control accuracy stays above 95% of baseline.

## Common Issues

- **Negation too aggressive**: If the control threshold is too high, negation may fail to reduce target accuracy. Try lowering `--control-threshold` to 0.90.
- **Uneven dataset batch sizes**: Task addition divides the batch across all datasets. With 8 datasets and batch size 128, each dataset gets only 16 samples per step. Increase batch size if individual datasets underperform.
