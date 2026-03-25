# 07 — Text-to-Coefficient Prediction

## Scientific Question

Can we adapt a vision model to a new task using *only text descriptions*, with no task-specific images at all? This experiment explores two complementary approaches to this question:

1. **Hypernetwork approach**: Meta-train a network that maps text descriptions directly to aTLAS coefficients. At inference time, provide a textual description of the target task (e.g., "photos of different flower species") and the hypernetwork predicts how to weight the task vector library.

2. **Synthetic data approach**: Use text-to-image models to generate task-specific images from text, fine-tune on the synthetic images to create synthetic task vectors, then compose them with the aTLAS framework.

Both approaches enable **zero-shot task adaptation** — adapting to a task that was never seen during training, using only its natural language description.

## Method Overview

### Text-to-Coefficient Hypernetwork

The hypernetwork is a small MLP that maps CLIP text embeddings to aTLAS coefficients:

1. **Text encoding**: Each class description is encoded using CLIP's text encoder (frozen).
2. **Aggregation**: Multiple descriptions per class are aggregated (mean/max/median).
3. **MLP prediction**: The aggregated embedding passes through hidden layers with LayerNorm, ReLU, and Dropout.
4. **Output**: A coefficient vector of shape `[num_task_vectors, num_blocks]`.

**Meta-training** uses an episodic approach:
- Each episode samples a dataset from the training pool.
- The hypernetwork predicts coefficients from text descriptions.
- The predicted coefficients compose the pretrained model with task vectors.
- The composed model is evaluated on a held-out batch; cross-entropy loss is backpropagated through the hypernetwork.

This trains the hypernetwork to understand the relationship between text semantics and optimal task vector weightings.

### Synthetic Data Pipeline

An alternative approach that avoids meta-learning:
1. **Generate descriptions**: Use LLM (GPT-4o, Claude) or templates to create class descriptions.
2. **Generate images**: Feed descriptions to a T2I model (Stable Diffusion, DALL-E) to create synthetic images.
3. **Fine-tune**: Train CLIP on synthetic images → synthetic task vector.
4. **Compose**: Use the synthetic task vector in the standard aTLAS framework.

### Text Description Sources

Three types of descriptions are supported:
- **Manual**: Human-written, highest quality, stored in `data/text_descriptions/manual/`.
- **Generated**: LLM-produced (GPT-4o, Claude), more diverse, stored in `data/text_descriptions/generated/`.
- **Templates**: CLIP-style templates ("a photo of a {class}"), lowest quality but always available.

## Relationship to Other Experiments

- **Prerequisites**: [01-finetuning.md](01-finetuning.md) — requires task vector library from fine-tuned models.
- **Complements**: [03-few-shot-adaptation.md](03-few-shot-adaptation.md) — offers an alternative when labeled images aren't available.
- **Extended by**: [08-multimodal-hypernetwork.md](08-multimodal-hypernetwork.md) — adds image support for improved predictions.
- **Analyzed by**: [06-baselines-and-analysis.md](06-baselines-and-analysis.md) — text ablation and synthetic quality scripts.

## Key Implementation Details

**Scripts**:
- `src/learn_text_to_coef.py` — meta-train text hypernetwork
- `src/eval_text_adaptation.py` — evaluate text-based adaptation
- `src/generate_synthetic_data.py` — generate synthetic images from text
- `src/text_descriptions/generators.py` — generate text descriptions via LLMs

**Key classes**:
- `TextToCoefHypernetwork` (`src/hypernetworks/text_to_coef.py`) — maps text → coefficients
- `TextConditionedWeightedImageEncoder` (`src/composition.py`) — applies predicted coefficients
- `TextDescriptionLoader` (`src/text_descriptions/loaders.py`) — loads/saves descriptions
- `StableDiffusionBackend` / `DalleBackend` (`src/text2image/`) — T2I generation

**Hypernetwork architectures**:
| Size | Hidden Dims | Approx. Params |
|------|-------------|---------------|
| small | [512, 256] | ~400K |
| medium | [512, 512, 256] | ~650K |
| large | [768, 512, 512, 256] | ~1.1M |

**Meta-training hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Meta learning rate | 1e-4 |
| Meta epochs | 50–100 |
| Episodes per epoch | 10–20 |
| Batch size per episode | 4 |
| Text encoder | CLIP (frozen) |
| Validation frequency | Every 5 epochs |

## How to Run

### Step 1: Generate Text Descriptions (if needed)

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

### Step 2: Meta-Train Hypernetwork

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

### Step 3: Evaluate on Held-Out Datasets

```bash
# Hypernetwork approach
python src/eval_text_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/$MODEL/hypernetworks/text_to_coef/meta_trained.pt \
    --text-source manual

# Synthetic data approach
python src/eval_text_adaptation.py \
    --model $MODEL \
    --dataset Flowers102 \
    --approach synthetic \
    --synthetic-backend stable_diffusion
```

### Alternative: Generate Synthetic Images

```bash
python src/generate_synthetic_data.py \
    --dataset CIFAR10 \
    --t2i-backend stable_diffusion \
    --num-images-per-class 100 \
    --output-dir data/synthetic_images
```

## Expected Outputs

**Hypernetwork checkpoints** (in `checkpoints/<MODEL>/hypernetworks/text_to_coef/`):
| File | Description |
|------|-------------|
| `meta_trained.pt` | Best validation checkpoint |
| `meta_results.json` | Training history with per-epoch metrics |

**Evaluation results**:
- Top-1 and top-5 accuracy on target datasets
- Comparison between hypernetwork and synthetic approaches

**Text descriptions** (in `data/text_descriptions/`):
```
data/text_descriptions/
├── manual/
│   ├── CIFAR10.json
│   └── Flowers102.json
├── generated/
│   ├── CIFAR10_gpt4o.json
│   └── CIFAR10_claude.json
└── templates/
    └── CIFAR10.json
```

**What "good" looks like**:
- Hypernetwork should outperform zero-shot CLIP by 3–10% on held-out datasets.
- Manual descriptions typically outperform templates by 2–5%.
- LLM-generated descriptions are competitive with manual ones (within 1–2%).
- Synthetic data approach is competitive with hypernetwork but requires more compute.

## Common Issues

- **Text encoder mismatch**: Ensure the CLIP text encoder used in the hypernetwork matches the one from the target model. Mixing ViT-B-32 and ViT-L-14 text encoders will produce poor results.
- **Meta-train/test overlap**: Never include evaluation datasets in `--meta-train-datasets`. The hypernetwork must generalize to unseen tasks.
- **LLM API keys**: Generated descriptions require API keys for GPT-4o or Claude. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variables.
- **Synthetic image quality**: SDXL generally produces higher-quality images than base Stable Diffusion. DALL-E 3 is best quality but expensive.
