# 05 — Test-Time Adaptation (Unsupervised)

## Scientific Question

Can aTLAS adapt a pretrained model to a new task using *only unlabeled test data*? This is the most extreme low-resource setting: no labeled examples at all, no text descriptions, just the test images themselves. The experiment explores whether the structure of task vectors combined with pseudo-labeling can enable meaningful unsupervised adaptation.

## Method Overview

The approach, termed **UFM** (Unsupervised Fine-tuning Module), combines aTLAS coefficient learning with semi-supervised learning techniques:

1. **Initial pseudo-labels**: Run the pretrained CLIP model on test data to obtain zero-shot predictions.
2. **Trusted sample selection**: For each predicted class, select the top-$K$ most confident samples as "trusted." These serve as a noisy labeled set.
3. **FixMatch-style training**: Use two augmentation streams:
   - Weak augmentation → generate pseudo-labels (teacher)
   - Strong augmentation → train against pseudo-labels (student)
4. **Coefficient learning**: Only aTLAS coefficients are updated. The loss combines cross-entropy on trusted samples with an entropy minimization term on all samples.
5. **Adaptive thresholding**: Pseudo-labels are only used when the model's confidence exceeds a dynamic threshold, which adjusts based on the confidence distribution.

The final model is the composition of pretrained weights with task vectors weighted by the learned (unsupervised) coefficients. Optionally, adapters (TIP, LP++) can be applied post-adaptation for additional gains.

## Relationship to Other Experiments

- **Prerequisites**: [01-finetuning.md](01-finetuning.md) — requires fine-tuned checkpoints (task vector library).
- **Contrasts with**: [03-few-shot-adaptation.md](03-few-shot-adaptation.md) — this uses no labels at all.
- **Evaluation**: Results can be compared against zero-shot CLIP and few-shot aTLAS to understand the value of unlabeled data.

## Key Implementation Details

**Core script**: `src/learn_ufm.py`

**Key mechanisms**:
- Two-stream batching: each batch contains both trusted and untrusted samples
- Trusted samples selected by per-class confidence ranking
- Adaptive threshold: adjusts based on running average of prediction confidence
- Optional adapter integration post-adaptation

**Default hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 10 |
| Trusted samples | Top-K per class by confidence |
| Coefficient type | Blockwise |

## How to Run

### Prerequisites
- Fine-tuned checkpoints for all 22 datasets
- No labeled data needed — only the test set

### Basic test-time adaptation
```bash
MODEL=ViT-B-32
python src/learn_ufm.py --model $MODEL --blockwise-coef
```

### Linearized mode
```bash
python src/learn_ufm.py --model $MODEL --blockwise-coef --finetuning-mode linear
```

## Expected Outputs

Results saved under `results/<MODEL>/test_time/`:

| File | Description |
|------|-------------|
| `learned_composition.pt` | Best unsupervised coefficients |
| `learned_composition.json` | Per-dataset accuracy results |

**Result format**:
```json
{
  "Cars": {"test_acc": 0.58},
  "DTD": {"test_acc": 0.52},
  "EuroSAT": {"test_acc": 0.88},
  ...
}
```

**What "good" looks like**: UFM should consistently improve over zero-shot CLIP (by 2–8% depending on the dataset). It will typically underperform few-shot aTLAS with even 1–2 labeled shots, since pseudo-labels are noisy. The largest gains tend to appear on datasets where CLIP's zero-shot predictions are already reasonable (e.g., EuroSAT, GTSRB).

## Common Issues

- **Poor zero-shot accuracy**: If CLIP's initial predictions are very poor on a dataset (e.g., <30%), the pseudo-labels will be mostly wrong and adaptation may hurt. This is a fundamental limitation of unsupervised approaches.
- **Class imbalance in pseudo-labels**: The trusted sample selection assumes classes are roughly balanced in the test set. Heavily imbalanced datasets may bias the adaptation.
- **Slow convergence**: Since the signal is weak (noisy pseudo-labels), learning rates should be kept low. The default 1e-3 is intentionally conservative.
