# 04 — Parameter-Efficient Fine-Tuning (aTLAS x K)

## Scientific Question

How many learnable parameters do we actually need for effective task adaptation? Standard aTLAS learns one coefficient per parameter block (typically ~100–200 coefficients for a ViT). **aTLAS x K** asks: what if we split each block into $K$ partitions, each with its own coefficient? This trades off between the expressiveness of more coefficients and the risk of overfitting with limited data.

This experiment positions aTLAS as a **parameter-efficient fine-tuning (PEFT)** method, competing with approaches like LoRA, BitFit, and adapter layers, but operating in task vector space rather than on individual weights.

## Method Overview

In standard blockwise aTLAS, each parameter block $b$ has a single coefficient $\alpha_b$. In aTLAS x K:

1. Each parameter block is randomly partitioned into $K$ groups (using a fixed random assignment).
2. Each group gets its own coefficient, yielding $K$ times more learnable parameters.

$$\theta_{\text{adapted}} = \theta_{\text{pretrained}} + \sum_{i=1}^{N} \text{partition}(\alpha_i, K) \odot \tau_i$$

The total number of learnable parameters scales as $N \times B \times K$, where $N$ is the number of task vectors, $B$ is the number of blocks, and $K$ is the partition count.

The experiment sweeps over different **data fractions** (0.01 to 1.0 of the training set) to measure how the benefit of additional coefficients varies with data availability. The key finding is that higher $K$ helps when more data is available but can hurt with very few samples.

## Relationship to Other Experiments

- **Prerequisites**: [01-finetuning.md](01-finetuning.md) — requires fine-tuned checkpoints.
- **Variant of**: [03-few-shot-adaptation.md](03-few-shot-adaptation.md) — uses the same script with `--partition K`.
- **Complements**: [06-baselines-and-analysis.md](06-baselines-and-analysis.md) — compare against standard PEFT baselines.

## Key Implementation Details

**Core script**: `src/learn_few_shots.py` with `--partition K`

**How partitioning works** (`src/composition.py`):
- `WeightedImageEncoder` initializes a random partition assignment per block when `partition > 1`
- Each parameter within a block is assigned to one of $K$ groups
- The coefficient tensor shape becomes `[num_task_vectors, num_blocks, K]`
- During forward pass, each parameter is scaled by the coefficient of its assigned partition

**Key arguments**:
| Flag | Description |
|------|-------------|
| `--partition` | Number of partitions per block (K) |
| `--subsample` | Data fraction (float < 1) or shot count (int) |
| `--blockwise-coef` | Must be enabled for partitioning to work |

## How to Run

### Prerequisites
- Fine-tuned checkpoints for all 22 datasets

### Standard aTLAS x K sweep
```bash
MODEL=ViT-B-32
PARTITION=10

for PERC in 0.01 0.05 0.1 0.25 0.35 0.5 1.0; do
    python src/learn_few_shots.py \
        --model $MODEL \
        --partition $PARTITION \
        --subsample $PERC \
        --blockwise-coef
done
```

### Compare different partition sizes
```bash
MODEL=ViT-B-32
PERC=0.1

for K in 1 5 10 25 50; do
    python src/learn_few_shots.py \
        --model $MODEL \
        --partition $K \
        --subsample $PERC \
        --blockwise-coef
done
```

## Expected Outputs

Same output format as [03-few-shot-adaptation.md](03-few-shot-adaptation.md), with results nested by data fraction:

```json
{
  "0.1": {
    "Cars": {"val_acc": 0.75, "test_acc": 0.74},
    "DTD": {"val_acc": 0.68, "test_acc": 0.66},
    ...
  }
}
```

**What "good" looks like**:
- At 1% data: aTLAS x K ≈ standard aTLAS (few parameters are better to avoid overfitting)
- At 10% data: aTLAS x K (K=10) should outperform standard aTLAS by 2–5%
- At 100% data: aTLAS x K approaches full fine-tuning performance while using far fewer parameters

**Parameter count comparison** (ViT-B-32, 21 task vectors, ~150 blocks):
| Method | Learnable Parameters |
|--------|---------------------|
| Global scalar | 21 |
| Blockwise (K=1) | ~3,150 |
| aTLAS x 10 | ~31,500 |
| aTLAS x 50 | ~157,500 |
| Full fine-tuning | ~88M |

## Common Issues

- **High K with low data**: Partition count $K > 10$ with less than 5% of training data may overfit. Monitor validation accuracy carefully.
- **Random seed sensitivity**: Partition assignments are random. Results may vary across seeds; consider averaging over multiple runs.
