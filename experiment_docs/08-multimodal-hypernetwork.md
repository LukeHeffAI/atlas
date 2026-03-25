# 08 — Multi-Modal Hypernetwork (Text + Images)

## Scientific Question

Text descriptions capture high-level semantics of a task, but they miss visual details — texture patterns, color distributions, spatial layouts — that even a handful of images can convey. Can we build a hypernetwork that combines **both text descriptions and a small set of support images** to predict aTLAS coefficients more accurately than either modality alone?

This experiment extends the text-only hypernetwork ([07-text-to-coefficient.md](07-text-to-coefficient.md)) with a dual-branch architecture that fuses semantic and visual signals. The key advantage is **graceful degradation**: the same model operates in both multi-modal mode (text + images) and text-only mode (zero-shot), enabling a unified system that improves as more images become available.

## Method Overview

### Architecture

The `MultiModalHypernetwork` has five components:

1. **Text branch**: CLIP text encoder (frozen) → linear projection → `[proj_dim]`
2. **Image branch**: OpenCLIP visual encoder (frozen) → shot pooling → linear projection → `[proj_dim]`
3. **Fusion module**: Combines text and image projections using one of three strategies
4. **Post-fusion MLP**: Maps the fused representation to the coefficient space
5. **Output layer**: Produces coefficients of shape `[num_task_vectors × num_blocks]`

### Shot Pooling

Given $K$ support images per class, the image branch must aggregate them into a fixed-size representation:
- **Mean pooling**: Simple average across all shots. Fast, no extra parameters.
- **Attention pooling**: A learnable query attends to each shot's features, producing a weighted sum. Better at ignoring outliers or noisy images.

### Fusion Modes

Three strategies for combining text and image projections:
- **Concat**: Concatenates projections → `[2 × proj_dim]`. Simple, doubles the post-fusion input size.
- **Add**: Element-wise sum → `[proj_dim]`. Forces both branches into a shared space.
- **Attention**: Cross-attention where text attends to per-shot image features → `[proj_dim]`. Most expressive, allows selective attention to relevant visual information.

### Graceful Degradation

When no support images are available, a learned `text_only_proj` layer maps the text projection directly to the fusion output space. This allows the **same trained model** to handle:
- 0 shots (text-only, zero-shot)
- 1–16 shots (multi-modal, few-shot)

### Variable-Shots Training

During meta-training, `--variable-shots` randomly varies $K \in [1, \text{num\_shots}]$ per episode. This forces the model to be robust across different shot counts rather than overfitting to a fixed $K$.

## Relationship to Other Experiments

- **Prerequisites**: [01-finetuning.md](01-finetuning.md) — requires task vector library; [07-text-to-coefficient.md](07-text-to-coefficient.md) — shares text description infrastructure.
- **Extends**: [07-text-to-coefficient.md](07-text-to-coefficient.md) — adds image branch to text-only hypernetwork.
- **Compared with**: [03-few-shot-adaptation.md](03-few-shot-adaptation.md) — both use few-shot data, but this uses a meta-learned predictor rather than per-task optimization.

## Key Implementation Details

**Scripts**:
- `src/learn_multimodal_to_coef.py` — meta-train multi-modal hypernetwork
- `src/eval_multimodal_adaptation.py` — evaluate with sweep across shot counts

**Key classes**:
- `MultiModalHypernetwork` (`src/hypernetworks/multimodal_to_coef.py`) — dual-branch architecture
- `MultiModalEpisodeSampler` (`src/meta_learning/multimodal_sampler.py`) — episode generation
- `AttentionShotPooling` — learnable shot aggregation (in multimodal_to_coef.py)
- `CrossAttentionFusion` — attention-based modality fusion (in multimodal_to_coef.py)

**Key arguments**:
| Flag | Description | Default |
|------|-------------|---------|
| `--fusion-mode` | Fusion strategy: concat, add, attention | concat |
| `--num-shots` | Support images per class | 4 |
| `--image-pooling` | Shot pooling: mean, attention | mean |
| `--text-input-mode` | dataset (aggregate) or per_class (aligned) | dataset |
| `--variable-shots` | Randomly vary K per episode | off |
| `--proj-dim` | Projection dimension for both branches | 256 |
| `--eval-mode` | multimodal, text_only, or sweep | multimodal |
| `--num-eval-seeds` | Seeds for sweep evaluation | 3 |

**Meta-training hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Meta learning rate | 1e-4 |
| LR schedule | Cosine annealing |
| Meta epochs | 100 |
| Episodes per epoch | 20 |
| Validation frequency | Every 5 epochs |
| Text encoder | CLIP (frozen) |
| Image encoder | OpenCLIP (frozen) |

## How to Run

### Prerequisites
- Fine-tuned checkpoints for all 22 datasets
- Text descriptions for meta-training datasets (manual or generated)

### Meta-Train

```bash
MODEL=ViT-B-32
python src/learn_multimodal_to_coef.py \
    --model $MODEL \
    --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
    --meta-val-datasets Caltech101,Flowers102 \
    --hypernetwork-arch medium \
    --fusion-mode concat \
    --num-shots 4 \
    --image-pooling mean \
    --text-input-mode dataset \
    --variable-shots \
    --meta-epochs 100 \
    --episodes-per-epoch 20 \
    --blockwise-coef
```

### Evaluate: Single Shot Count

```bash
python src/eval_multimodal_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --hypernetwork-checkpoint checkpoints/$MODEL/hypernetworks/multimodal_to_coef/meta_trained.pt \
    --num-shots 4 \
    --eval-mode multimodal
```

### Evaluate: Sweep Across Shot Counts

```bash
python src/eval_multimodal_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --hypernetwork-checkpoint checkpoints/$MODEL/hypernetworks/multimodal_to_coef/meta_trained.pt \
    --eval-mode sweep
```

This evaluates at 0, 1, 2, 4, 8, and 16 shots with multiple seeds.

### Evaluate: Text-Only Baseline

```bash
python src/eval_multimodal_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --hypernetwork-checkpoint checkpoints/$MODEL/hypernetworks/multimodal_to_coef/meta_trained.pt \
    --eval-mode text_only
```

### Compare Multi-Modal vs. Text-Only

```bash
python scripts/compare_multimodal_vs_text.py \
    --multimodal-results checkpoints/$MODEL/multimodal_adapted/ \
    --textonly-results checkpoints/$MODEL/text_adapted/ \
    --datasets Flowers102,Cars,DTD \
    --shot-count 4 \
    --output figures/multimodal_vs_text.pdf
```

### Fusion Mode Ablation

```bash
for FUSION in concat add attention; do
    python src/learn_multimodal_to_coef.py \
        --model $MODEL \
        --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
        --meta-val-datasets Caltech101,Flowers102 \
        --fusion-mode $FUSION \
        --num-shots 4 \
        --variable-shots \
        --blockwise-coef
done
```

## Expected Outputs

**Checkpoints** (in `checkpoints/<MODEL>/hypernetworks/multimodal_to_coef/`):
| File | Description |
|------|-------------|
| `meta_trained.pt` | Best validation checkpoint |
| `meta_trained_final.pt` | Last epoch checkpoint |
| `meta_results.json` | Training history with config |

**Evaluation results** (in `checkpoints/<MODEL>/multimodal_adapted/<DATASET>/`):
| File | Description |
|------|-------------|
| `multimodal_results.json` | Multi-modal evaluation results |
| `text_only_results.json` | Text-only baseline results |
| `sweep_results.json` | Per-shot, per-seed results |

**Sweep result format**:
```json
{
  "0": {"seed_0": 0.62, "seed_1": 0.61, "seed_2": 0.63, "mean": 0.62},
  "1": {"seed_0": 0.67, "seed_1": 0.66, "seed_2": 0.68, "mean": 0.67},
  "4": {"seed_0": 0.73, "seed_1": 0.72, "seed_2": 0.74, "mean": 0.73},
  "16": {"seed_0": 0.78, "seed_1": 0.77, "seed_2": 0.79, "mean": 0.78}
}
```

**What "good" looks like**:
- Multi-modal outperforms text-only by **5–15%** across datasets
- Largest gains at **low shot counts** (1–4 shots)
- Diminishing returns after **8–16 shots** (direct fine-tuning becomes competitive)
- **Attention fusion** likely best for diverse tasks; **concat** is a strong default
- `--variable-shots` training improves robustness across all shot counts

## Common Issues

- **Image encoder mismatch**: The multi-modal hypernetwork uses the OpenCLIP visual encoder. Ensure it matches the CLIP model variant (ViT-B-32 etc.).
- **Memory with high shot counts**: Encoding 16 shots × many classes can be memory-intensive. The implementation uses chunked processing (chunk_size=64) to mitigate this.
- **Variable shots hurt fixed-K performance**: If you only care about a specific shot count (e.g., always 4-shot), training *without* `--variable-shots` may yield slightly better results at that exact K, at the cost of worse performance at other shot counts.
- **Meta-train/test overlap**: As with the text hypernetwork, never include evaluation datasets in the meta-training set.
