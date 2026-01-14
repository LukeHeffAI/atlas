# Extension Roadmap for Text-Based LoRA Adaptation

This document contains detailed implementation plans for extending the text-based LoRA adaptation system beyond the core implementation.

Created: 2026-01-13
Last Updated: 2026-01-13

---

## Table of Contents

1. [Ablation Studies](#1-ablation-studies)
2. [Full LoRA Prediction](#2-full-lora-prediction)
3. [Multi-Modal Hypernetwork](#3-multi-modal-hypernetwork)
4. [Cross-Model Generalization](#4-cross-model-generalization)
5. [Domain Adaptation](#5-domain-adaptation)
6. [Task Composition](#6-task-composition)

---

## 1. Ablation Studies

### Overview
Conduct systematic ablation studies to understand which components contribute most to performance and identify optimal configurations.

### 1.1 Text Source Comparison

**Objective**: Compare manual descriptions, template-based descriptions, and LLM-generated descriptions.

**Methodology**:
1. **Datasets**: Select 5-8 diverse datasets (CIFAR10, EuroSAT, MNIST, DTD, GTSRB, Cars, SUN397)
2. **Text Sources**:
   - Manual: Hand-curated descriptions (baseline)
   - Templates: CLIP-style templates
   - GPT-4o generated: 10, 15, 20 descriptions per class
   - Claude generated: 10, 15, 20 descriptions per class
3. **Metrics**: Zero-shot accuracy, few-shot accuracy (1, 2, 4, 8, 16-shot), convergence speed

**Implementation Steps**:
```bash
# 1. Generate descriptions for all sources
for DATASET in CIFAR10 EuroSAT MNIST DTD GTSRB; do
    # Manual: already exists

    # Templates
    python src/text_descriptions/templates.py \
        --dataset $DATASET \
        --classes <class_list> \
        --output-dir data/text_descriptions

    # GPT-4o
    for NUM_DESC in 10 15 20; do
        python src/text_descriptions/generators.py \
            --dataset $DATASET \
            --classes <class_list> \
            --llm gpt4o \
            --num-descriptions $NUM_DESC \
            --output-dir data/text_descriptions
    done

    # Claude
    for NUM_DESC in 10 15 20; do
        python src/text_descriptions/generators.py \
            --dataset $DATASET \
            --classes <class_list> \
            --llm claude \
            --num-descriptions $NUM_DESC \
            --output-dir data/text_descriptions
    done
done

# 2. Meta-train hypernetworks with each text source
MODEL=ViT-B-32
for TEXT_SOURCE in manual templates; do
    python src/learn_text_to_coef.py \
        --model $MODEL \
        --meta-train-datasets CIFAR10,EuroSAT,MNIST,DTD,GTSRB \
        --meta-val-datasets Cars,SUN397 \
        --text-source $TEXT_SOURCE \
        --hypernetwork-arch small \
        --meta-epochs 100 \
        --save checkpoints/$MODEL/hypernetworks/text_to_coef_${TEXT_SOURCE}
done

for TEXT_VARIANT in gpt4o_10 gpt4o_15 gpt4o_20 claude_10 claude_15 claude_20; do
    python src/learn_text_to_coef.py \
        --model $MODEL \
        --meta-train-datasets CIFAR10,EuroSAT,MNIST,DTD,GTSRB \
        --meta-val-datasets Cars,SUN397 \
        --text-source generated \
        --text-variant $TEXT_VARIANT \
        --hypernetwork-arch small \
        --meta-epochs 100 \
        --save checkpoints/$MODEL/hypernetworks/text_to_coef_${TEXT_VARIANT}
done

# 3. Evaluate all variants on held-out datasets
for CHECKPOINT in checkpoints/$MODEL/hypernetworks/text_to_coef_*/meta_trained.pt; do
    for TEST_DATASET in Flowers102 UCF101 CUB200; do
        python src/eval_text_adaptation.py \
            --model $MODEL \
            --dataset $TEST_DATASET \
            --approach hypernetwork \
            --hypernetwork-checkpoint $CHECKPOINT \
            --text-source <corresponding_source>
    done
done

# 4. Analyze results
python scripts/analyze_text_ablation.py \
    --results-dir checkpoints/$MODEL/text_adapted/ \
    --output ablation_text_source.pdf
```

**Expected Findings**:
- Manual descriptions likely best quality but time-consuming
- GPT-4o generates more diverse descriptions than templates
- 15 descriptions per class likely optimal balance
- LLM-generated descriptions may capture more visual details

**Files to Create**:
- `scripts/analyze_text_ablation.py`: Analysis and visualization script

---

### 1.2 T2I Backend Comparison

**Objective**: Compare different text-to-image models for synthetic image generation.

**Methodology**:
1. **Backends to Compare**:
   - Stable Diffusion XL 1.5 (baseline)
   - DALL-E 3
   - SD3.5 Large (if available)
   - GPT Image 1 (if available)
2. **Metrics**: Synthetic TV similarity to real TV, few-shot accuracy improvement, image quality (FID, IS)
3. **Cost Analysis**: Time and $ cost per dataset

**Implementation Steps**:
```bash
# 1. Generate synthetic images with each backend
MODEL=ViT-B-32
DATASET=CIFAR10

for BACKEND in stable_diffusion dalle sd3.5 gpt_image; do
    python src/generate_synthetic_data.py \
        --dataset $DATASET \
        --text-source manual \
        --t2i-backend $BACKEND \
        --num-images-per-class 100 \
        --output-dir data/synthetic_images \
        --seed 42

    # Fine-tune on synthetic images
    python src/finetune.py \
        --model $MODEL \
        --dataset Synthetic${DATASET} \
        --synthetic-backend $BACKEND \
        --epochs 10 \
        --save checkpoints/$MODEL/$DATASET/synthetic_${BACKEND}_finetuned.pt
done

# 2. Evaluate synthetic task vectors
for BACKEND in stable_diffusion dalle sd3.5 gpt_image; do
    python src/eval_text_adaptation.py \
        --model $MODEL \
        --dataset $DATASET \
        --approach synthetic \
        --synthetic-backend $BACKEND
done

# 3. Compare image quality
python scripts/analyze_synthetic_quality.py \
    --synthetic-dir data/synthetic_images/ \
    --real-dataset $DATASET \
    --metrics fid,is,lpips \
    --output synthetic_quality_comparison.pdf
```

**Expected Findings**:
- SDXL 1.5 best balance of quality and cost (local, free)
- DALL-E 3 highest quality but expensive ($40 per 1000 images)
- SD3.5 may have better text rendering but higher memory requirements
- GPT Image 1 quality unclear, needs testing

**Files to Create**:
- `scripts/analyze_synthetic_quality.py`: FID, IS, LPIPS computation
- `scripts/visualize_synthetic_images.py`: Side-by-side comparison tool

---

### 1.3 Hypernetwork Architecture Comparison

**Objective**: Determine optimal hypernetwork size for different scenarios.

**Methodology**:
1. **Architectures**:
   - Tiny: [256, 128] (ultra-fast, minimal performance)
   - Small: [512, 256] (baseline, RTX 4090)
   - Medium: [512, 512, 256] (4x A100)
   - Large: [768, 512, 512, 256] (8x A100, best performance)
   - XL: [1024, 768, 512, 256] (extreme, research only)
2. **Metrics**: Zero-shot accuracy, meta-training time, parameter count, memory usage

**Implementation Steps**:
```bash
# Create new architecture configs
cat > configs/hypernetworks/tiny.yaml << EOF
hidden_dims: [256, 128]
dropout: 0.1
meta_lr: 1e-4
meta_epochs: 100
episodes_per_epoch: 20
EOF

cat > configs/hypernetworks/xl.yaml << EOF
hidden_dims: [1024, 768, 512, 256]
dropout: 0.1
meta_lr: 5e-5
meta_epochs: 200
episodes_per_epoch: 40
EOF

# Meta-train all architectures
for ARCH in tiny small medium large xl; do
    python src/learn_text_to_coef.py \
        --model ViT-B-32 \
        --meta-train-datasets CIFAR10,EuroSAT,MNIST,DTD,GTSRB,Food101,STL10,SVHN \
        --meta-val-datasets Cars,SUN397,RESISC45 \
        --text-source manual \
        --hypernetwork-arch $ARCH \
        --blockwise-coef \
        --meta-epochs <from_config> \
        --episodes-per-epoch <from_config> \
        --save checkpoints/ViT-B-32/hypernetworks/text_to_coef_${ARCH}
done

# Evaluate on diverse test sets
for ARCH in tiny small medium large xl; do
    for DATASET in Flowers102 UCF101 CUB200 Country211; do
        python src/eval_text_adaptation.py \
            --model ViT-B-32 \
            --dataset $DATASET \
            --approach hypernetwork \
            --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef_${ARCH}/meta_trained.pt
    done
done

# Analyze trade-offs
python scripts/analyze_hypernetwork_size.py \
    --results-dir checkpoints/ViT-B-32/hypernetworks/ \
    --plot-tradeoffs \
    --output hypernetwork_size_analysis.pdf
```

**Expected Findings**:
- Small: Good for rapid prototyping, 90% of large performance
- Medium: Best balance for production use
- Large/XL: Marginal gains (2-5%), only for final paper results
- Diminishing returns after medium size

**Files to Create**:
- `scripts/analyze_hypernetwork_size.py`: Performance vs size analysis

---

## 2. Full LoRA Prediction

### Overview
Extend hypernetwork to predict full LoRA weight matrices instead of just scalar coefficients. This enables learning entirely new features rather than just composing existing task vectors.

### 2.1 Architecture Design

**Key Differences from Coefficient Prediction**:
- **Output dimension**: ~100K-1M parameters instead of ~4K coefficients
- **LoRA structure**: Predict low-rank matrices A (rank × d_in) and B (d_out × rank) for each layer
- **Rank**: Start with rank=4, then 8, 16 (higher rank = more expressive but harder to meta-learn)

**Implementation**:

**File: `src/hypernetworks/text_to_lora.py`**

```python
class TextToLoRAHypernetwork(BaseHypernetwork):
    """Hypernetwork that predicts LoRA weights from text descriptions.

    Instead of predicting scalar coefficients, this predicts low-rank
    weight matrices for each layer, enabling more expressive adaptation.
    """

    def __init__(
        self,
        text_encoder_name: str,
        target_model_config: Dict,  # Info about target model architecture
        lora_rank: int = 8,
        lora_layers: List[str] = ["all"],  # Which layers to apply LoRA
        hidden_dims: List[int] = [512, 512, 256],
        dropout: float = 0.1,
        freeze_text_encoder: bool = True
    ):
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.lora_rank = lora_rank
        self.lora_layers = lora_layers
        self.hidden_dims = hidden_dims

        # Load text encoder
        self._load_text_encoder()
        text_dim = self.text_encoder.config.hidden_size

        # Build shared MLP
        self._build_mlp(text_dim)

        # Build LoRA predictors for each layer
        self.lora_predictors = nn.ModuleDict()
        for layer_name in target_model_config['lora_layers']:
            d_in = target_model_config[layer_name]['d_in']
            d_out = target_model_config[layer_name]['d_out']

            # Predict A matrix (rank × d_in) and B matrix (d_out × rank)
            total_params = lora_rank * d_in + d_out * lora_rank

            self.lora_predictors[layer_name] = nn.Sequential(
                nn.Linear(hidden_dims[-1], 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, total_params)
            )

            # Initialize near zero
            nn.init.normal_(self.lora_predictors[layer_name][-1].weight, std=0.001)
            nn.init.zeros_(self.lora_predictors[layer_name][-1].bias)

    def forward(self, text_descriptions: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass: text descriptions → LoRA weights.

        Returns:
            Dictionary mapping layer names to (A_matrix, B_matrix) tuples
        """
        # Encode text
        tokens = self.tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)

        with torch.set_grad_enabled(not self.freeze_text_encoder):
            text_features = self.text_encoder(**tokens).pooler_output

        # Shared MLP
        hidden = self.mlp(text_features)

        # Predict LoRA weights for each layer
        lora_weights = {}
        for layer_name, predictor in self.lora_predictors.items():
            flat_weights = predictor(hidden)

            # Split into A and B matrices
            config = self.target_model_config[layer_name]
            d_in, d_out = config['d_in'], config['d_out']

            a_size = self.lora_rank * d_in
            b_size = d_out * self.lora_rank

            a_flat = flat_weights[:, :a_size]
            b_flat = flat_weights[:, a_size:a_size+b_size]

            A = a_flat.view(-1, self.lora_rank, d_in)
            B = b_flat.view(-1, d_out, self.lora_rank)

            lora_weights[layer_name] = (A, B)

        return lora_weights
```

### 2.2 LoRA Application

**File: `src/lora_adaptation.py`**

```python
class LoRAImageEncoder(nn.Module):
    """Image encoder with LoRA adaptation."""

    def __init__(self, base_model, lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.base_model = base_model
        self.lora_weights = lora_weights

        # Inject LoRA matrices into model
        for layer_name, (A, B) in lora_weights.items():
            self._inject_lora(layer_name, A, B)

    def _inject_lora(self, layer_name: str, A: torch.Tensor, B: torch.Tensor):
        """Inject LoRA matrices into a layer: W' = W + B @ A"""
        # Navigate to layer
        module = self.base_model
        for attr in layer_name.split('.'):
            module = getattr(module, attr)

        # Get original weight
        orig_weight = module.weight

        # Compute LoRA update: B @ A
        lora_update = torch.matmul(B, A)

        # Apply: W' = W + B @ A
        module.weight.data = orig_weight + lora_update

    def forward(self, x):
        return self.base_model(x)
```

### 2.3 Meta-Training Script

**File: `src/learn_text_to_lora.py`**

```python
def meta_train_lora(args):
    """Meta-train text-to-LoRA hypernetwork."""

    # Get target model config (layer dimensions)
    target_model_config = get_model_config(args.model)

    # Create hypernetwork
    hypernetwork = TextToLoRAHypernetwork(
        text_encoder_name="openai/clip-vit-base-patch32",
        target_model_config=target_model_config,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        hidden_dims=[512, 512, 256],
        dropout=0.1
    )
    hypernetwork = hypernetwork.cuda()

    # Meta-training loop (similar to coefficient version)
    for epoch in range(args.meta_epochs):
        for episode in range(args.episodes_per_epoch):
            # Sample task
            dataset_name = random.choice(args.meta_train_datasets)

            # Load text descriptions
            text_descriptions = load_text_descriptions(dataset_name, args)

            # Predict LoRA weights
            lora_weights = hypernetwork.predict_for_dataset(text_descriptions)

            # Apply LoRA to base model
            base_model = ImageEncoder(args)
            lora_model = LoRAImageEncoder(base_model, lora_weights)

            # Evaluate
            loss, acc = evaluate_model(lora_model, dataset_name, args)

            # Backprop
            loss.backward()
            optimizer.step()
```

### 2.4 Challenges and Solutions

**Challenge 1: Large Output Space**
- **Problem**: Predicting ~100K-1M parameters is much harder than ~4K coefficients
- **Solution**:
  - Start with low rank (rank=4 or 8)
  - Use progressive training: first train rank=4, then expand to rank=8, then rank=16
  - Use more meta-training data (15-20 datasets instead of 5-8)

**Challenge 2: Overfitting**
- **Problem**: Hypernetwork may memorize meta-training tasks instead of generalizing
- **Solution**:
  - Strong regularization (dropout=0.2-0.3, weight decay=0.1)
  - Data augmentation on text (paraphrase, back-translation)
  - Early stopping on meta-validation set

**Challenge 3: Computational Cost**
- **Problem**: 10-100x more parameters to predict and optimize
- **Solution**:
  - Use mixed precision training (FP16)
  - Gradient checkpointing
  - Distributed training across 4-8 GPUs
  - Reduce episodes per epoch (10-20 instead of 40)

### 2.5 Evaluation Plan

```bash
# Meta-train LoRA hypernetwork
python src/learn_text_to_lora.py \
    --model ViT-B-32 \
    --meta-train-datasets <15_datasets> \
    --meta-val-datasets <3_datasets> \
    --lora-rank 8 \
    --lora-layers attn,ffn \
    --hypernetwork-arch medium \
    --meta-epochs 200 \
    --world-size 4

# Evaluate on held-out datasets
for DATASET in <held_out_datasets>; do
    python src/eval_lora_adaptation.py \
        --model ViT-B-32 \
        --dataset $DATASET \
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_lora/meta_trained.pt
done

# Compare with coefficient-based approach
python scripts/compare_lora_vs_coef.py \
    --lora-results checkpoints/ViT-B-32/lora_adapted/ \
    --coef-results checkpoints/ViT-B-32/text_adapted/ \
    --output lora_vs_coef_comparison.pdf
```

**Expected Results**:
- LoRA should outperform coefficients by 3-10% on out-of-distribution tasks
- LoRA more expensive to meta-train (5-10x longer)
- LoRA better for tasks very different from meta-training set

**Files to Create**:
- `src/hypernetworks/text_to_lora.py`
- `src/lora_adaptation.py`
- `src/learn_text_to_lora.py`
- `src/eval_lora_adaptation.py`
- `scripts/compare_lora_vs_coef.py`

---

## 3. Multi-Modal Hypernetwork

### Overview
Extend hypernetwork to use BOTH text descriptions AND few example images for coefficient/LoRA prediction. This combines the zero-shot capability of text with the adaptation power of few-shot learning.

### 3.1 Architecture Design

**Inputs**:
1. **Text descriptions**: Dataset-level semantic information
2. **Support images**: Few example images per class (1-16 shots)

**Architecture**:
```
Text Branch:               Image Branch:
  Text Descriptions    →     Support Images
        ↓                          ↓
  CLIP Text Encoder      CLIP Image Encoder
        ↓                          ↓
    [512-dim]                  [512-dim]
        ↓                          ↓
    MLP [512→256]             MLP [512→256]
        ↓                          ↓
        └───────── Fusion ─────────┘
                      ↓
                  [512-dim]
                      ↓
              MLP [512→256→128]
                      ↓
            Output: Coefficients/LoRA
```

**Implementation**:

**File: `src/hypernetworks/multimodal_to_coef.py`**

```python
class MultiModalHypernetwork(BaseHypernetwork):
    """Hypernetwork that uses both text and images."""

    def __init__(
        self,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        image_encoder_name: str = "openai/clip-vit-base-patch32",
        num_task_vectors: int = 21,
        num_blocks: int = 200,
        fusion_mode: str = "concat",  # "concat", "add", "attention"
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        use_blockwise: bool = True
    ):
        super().__init__()

        # Text encoder branch
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        # Image encoder branch
        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        image_dim = self.image_encoder.config.hidden_size

        # Freeze encoders
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # Projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion module
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            fusion_dim = 512
        elif fusion_mode == "add":
            fusion_dim = 256
        elif fusion_mode == "attention":
            fusion_dim = 256
            self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # Post-fusion MLP
        layers = []
        prev_dim = fusion_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.fusion_mlp = nn.Sequential(*layers)

        # Output layer
        if use_blockwise:
            output_dim = num_task_vectors * num_blocks
        else:
            output_dim = num_task_vectors
        self.output = nn.Linear(prev_dim, output_dim)

        # Initialize near zero
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        text_descriptions: List[str],
        support_images: torch.Tensor  # [batch, num_shots, C, H, W]
    ) -> torch.Tensor:
        """Forward pass: text + images → coefficients.

        Args:
            text_descriptions: List of text descriptions
            support_images: Support images [batch, num_shots, C, H, W]

        Returns:
            Coefficients tensor
        """
        batch_size = support_images.shape[0]
        num_shots = support_images.shape[1]

        # Encode text
        tokens = self.text_tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)

        text_features = self.text_encoder(**tokens).pooler_output  # [batch, text_dim]
        text_proj = self.text_proj(text_features)  # [batch, 256]

        # Encode images
        # Flatten batch and shots: [batch * num_shots, C, H, W]
        support_images_flat = support_images.view(-1, *support_images.shape[2:])
        image_features = self.image_encoder(support_images_flat).pooler_output  # [batch*num_shots, image_dim]

        # Reshape back: [batch, num_shots, image_dim]
        image_features = image_features.view(batch_size, num_shots, -1)

        # Pool across shots (mean pooling)
        image_features_pooled = image_features.mean(dim=1)  # [batch, image_dim]
        image_proj = self.image_proj(image_features_pooled)  # [batch, 256]

        # Fusion
        if self.fusion_mode == "concat":
            fused = torch.cat([text_proj, image_proj], dim=-1)  # [batch, 512]
        elif self.fusion_mode == "add":
            fused = text_proj + image_proj  # [batch, 256]
        elif self.fusion_mode == "attention":
            # Cross-attention: text attends to images
            text_proj_unsq = text_proj.unsqueeze(1)  # [batch, 1, 256]
            image_proj_unsq = image_proj.unsqueeze(1)  # [batch, 1, 256]
            attended, _ = self.attention(text_proj_unsq, image_proj_unsq, image_proj_unsq)
            fused = attended.squeeze(1)  # [batch, 256]

        # Post-fusion MLP
        hidden = self.fusion_mlp(fused)

        # Output
        coef_flat = self.output(hidden)

        # Reshape
        if self.use_blockwise:
            coef = coef_flat.view(-1, self.num_task_vectors, self.num_blocks)
        else:
            coef = coef_flat.view(-1, self.num_task_vectors)

        return coef
```

### 3.2 Data Loading for Meta-Training

**Key Challenge**: Need both text descriptions AND support images for each episode.

**File: `src/meta_learning/multimodal_sampler.py`**

```python
class MultiModalEpisodeSampler:
    """Sample episodes with both text and images."""

    def __init__(
        self,
        datasets: List[str],
        num_shots: int,
        args
    ):
        self.datasets = datasets
        self.num_shots = num_shots
        self.args = args

    def sample_episode(self):
        """Sample one episode: dataset + text + support images.

        Returns:
            dataset_name: str
            text_descriptions: Dict[str, List[str]]
            support_images: torch.Tensor [num_classes, num_shots, C, H, W]
            support_labels: torch.Tensor [num_classes, num_shots]
        """
        # Sample dataset
        dataset_name = random.choice(self.datasets)

        # Load text descriptions
        text_descriptions = load_text_descriptions(dataset_name, self.args)

        # Load dataset
        preprocess = ... # Get from model
        dataset = get_dataset(dataset_name, preprocess, self.args.data_location)

        # Sample support images (N-way K-shot)
        support_images = []
        support_labels = []

        for class_idx in range(len(dataset.classnames)):
            # Get images from this class
            class_mask = (dataset.targets == class_idx)
            class_indices = torch.where(class_mask)[0]

            # Sample K shots
            shot_indices = torch.randperm(len(class_indices))[:self.num_shots]
            shot_images = [dataset[idx.item()][0] for idx in class_indices[shot_indices]]

            support_images.append(torch.stack(shot_images))
            support_labels.append(torch.full((self.num_shots,), class_idx))

        support_images = torch.stack(support_images)  # [num_classes, num_shots, C, H, W]
        support_labels = torch.stack(support_labels)  # [num_classes, num_shots]

        return dataset_name, text_descriptions, support_images, support_labels
```

### 3.3 Meta-Training Loop

**File: `src/learn_multimodal_to_coef.py`**

```python
def meta_train_multimodal(args):
    """Meta-train multimodal hypernetwork."""

    # Create hypernetwork
    hypernetwork = MultiModalHypernetwork(
        fusion_mode=args.fusion_mode,
        num_task_vectors=len(task_vectors),
        num_blocks=num_blocks,
        hidden_dims=[512, 256, 128],
        use_blockwise=args.blockwise_coef
    )
    hypernetwork = hypernetwork.cuda()

    # Create episode sampler
    sampler = MultiModalEpisodeSampler(
        datasets=args.meta_train_datasets,
        num_shots=args.num_shots,
        args=args
    )

    # Meta-training loop
    for epoch in range(args.meta_epochs):
        for episode in range(args.episodes_per_epoch):
            # Sample episode
            dataset_name, text_descriptions, support_images, support_labels = sampler.sample_episode()

            # Predict coefficients from text + images
            predicted_coef = hypernetwork(
                text_descriptions=list(text_descriptions.values()),  # Flatten
                support_images=support_images
            )

            # Create weighted encoder with predicted coefficients
            # (same as before)
            weighted_encoder = create_weighted_encoder_with_coef(predicted_coef, task_vectors, base_model)

            # Evaluate on query set
            loss, acc = evaluate_on_query_set(weighted_encoder, dataset_name, args)

            # Backprop
            loss.backward()
            optimizer.step()
```

### 3.4 Evaluation

```bash
# Meta-train multimodal hypernetwork
python src/learn_multimodal_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets <datasets> \
    --fusion-mode attention \
    --num-shots 4 \
    --meta-epochs 150

# Evaluate with varying numbers of shots
for NUM_SHOTS in 1 2 4 8 16; do
    python src/eval_multimodal_adaptation.py \
        --model ViT-B-32 \
        --dataset Flowers102 \
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/multimodal/meta_trained.pt \
        --num-shots $NUM_SHOTS \
        --fusion-mode attention
done

# Compare with text-only baseline
python scripts/compare_multimodal_vs_text.py \
    --multimodal-results checkpoints/ViT-B-32/multimodal_adapted/ \
    --textonly-results checkpoints/ViT-B-32/text_adapted/ \
    --output multimodal_vs_text.pdf
```

**Expected Results**:
- Multimodal should significantly outperform text-only (5-15% improvement)
- Improvement most pronounced at low shot counts (1-4 shots)
- Attention fusion likely better than concat/add
- Diminishing returns after 8-16 shots

**Files to Create**:
- `src/hypernetworks/multimodal_to_coef.py`
- `src/meta_learning/multimodal_sampler.py`
- `src/learn_multimodal_to_coef.py`
- `src/eval_multimodal_adaptation.py`
- `scripts/compare_multimodal_vs_text.py`

---

## 4. Cross-Model Generalization

### Overview
Meta-train hypernetwork on one model architecture (e.g., ViT-B/32) and test generalization to other architectures (e.g., ViT-L/14, RN50). This tests whether the hypernetwork learns general principles of task composition rather than model-specific patterns.

### 4.1 Experimental Design

**Source Models** (Meta-Training):
- ViT-B/32 (baseline)

**Target Models** (Transfer):
- ViT-B/16 (similar architecture, higher resolution)
- ViT-L/14 (larger model, more parameters)
- RN50 (different architecture family)
- RN101 (larger ResNet)

**Key Challenge**: Different models have different numbers of parameter blocks and block dimensions.

### 4.2 Architecture-Agnostic Hypernetwork

**Strategy**: Make hypernetwork predict coefficients normalized by block size, then scale appropriately for target model.

**File: `src/hypernetworks/architecture_agnostic.py`**

```python
class ArchitectureAgnosticHypernetwork(BaseHypernetwork):
    """Hypernetwork that generalizes across model architectures.

    Key idea: Instead of predicting absolute coefficients for fixed number of blocks,
    predict relative importance scores that can be mapped to any model architecture.
    """

    def __init__(
        self,
        text_encoder_name: str,
        num_task_vectors: int,
        num_layer_types: int = 4,  # attention, ffn, layernorm, embedding
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size

        # MLP
        layers = []
        prev_dim = text_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Output: predict coefficients for each (task_vector, layer_type) pair
        # This is architecture-agnostic: same output regardless of model size
        self.output = nn.Linear(prev_dim, num_task_vectors * num_layer_types)

        # Initialize near zero
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

        self.num_task_vectors = num_task_vectors
        self.num_layer_types = num_layer_types

    def forward(self, text_descriptions: List[str]) -> torch.Tensor:
        """Predict architecture-agnostic coefficients.

        Returns:
            coef: [batch, num_task_vectors, num_layer_types]
        """
        # Encode text
        tokens = self.text_tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)

        text_features = self.text_encoder(**tokens).pooler_output

        # Predict coefficients
        hidden = self.mlp(text_features)
        coef_flat = self.output(hidden)

        # Reshape to [batch, num_task_vectors, num_layer_types]
        coef = coef_flat.view(-1, self.num_task_vectors, self.num_layer_types)

        return coef

    def map_to_model(
        self,
        agnostic_coef: torch.Tensor,  # [1, num_task_vectors, num_layer_types]
        model_config: Dict  # Info about target model's parameter blocks
    ) -> torch.Tensor:
        """Map architecture-agnostic coefficients to specific model.

        Args:
            agnostic_coef: Coefficients for layer types
            model_config: Configuration of target model
                {
                    'blocks': [
                        {'name': 'attn.q', 'type': 'attention', 'dim': 768},
                        {'name': 'attn.k', 'type': 'attention', 'dim': 768},
                        {'name': 'ffn.fc1', 'type': 'ffn', 'dim': 3072},
                        ...
                    ]
                }

        Returns:
            model_specific_coef: [num_task_vectors, num_blocks] for this model
        """
        # Map layer type names to indices
        layer_type_to_idx = {
            'attention': 0,
            'ffn': 1,
            'layernorm': 2,
            'embedding': 3
        }

        num_blocks = len(model_config['blocks'])
        model_specific_coef = torch.zeros(self.num_task_vectors, num_blocks)

        # For each block in the model, assign coefficient based on its type
        for block_idx, block_info in enumerate(model_config['blocks']):
            layer_type = block_info['type']
            type_idx = layer_type_to_idx[layer_type]

            # Copy coefficients for this layer type
            model_specific_coef[:, block_idx] = agnostic_coef[0, :, type_idx]

        return model_specific_coef
```

### 4.3 Model Configuration Extraction

**File: `src/utils/model_config.py`**

```python
def extract_model_config(model_name: str) -> Dict:
    """Extract model configuration for architecture-agnostic mapping.

    Args:
        model_name: Model identifier (e.g., "ViT-B-32", "ViT-L-14", "RN50")

    Returns:
        Configuration dictionary with block information
    """
    # Load model
    from src.modeling import ImageEncoder
    args = argparse.Namespace(model=model_name, openclip_cachedir="~/openclip-cachedir")
    model = ImageEncoder(args)

    # Analyze state dict to identify block types
    blocks = []
    for name, param in model.model.state_dict().items():
        # Classify block type based on name
        if 'attn' in name or 'attention' in name:
            layer_type = 'attention'
        elif 'mlp' in name or 'ffn' in name or 'fc' in name:
            layer_type = 'ffn'
        elif 'ln' in name or 'norm' in name:
            layer_type = 'layernorm'
        elif 'embed' in name:
            layer_type = 'embedding'
        else:
            layer_type = 'other'

        blocks.append({
            'name': name,
            'type': layer_type,
            'shape': list(param.shape),
            'dim': param.numel()
        })

    return {
        'model_name': model_name,
        'num_blocks': len(blocks),
        'blocks': blocks
    }
```

### 4.4 Evaluation Protocol

```bash
# 1. Meta-train on ViT-B/32
python src/learn_text_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets <datasets> \
    --hypernetwork-type architecture_agnostic \
    --meta-epochs 100 \
    --save checkpoints/cross_model/ViT-B-32_source

# 2. Extract model configs for all target models
for MODEL in ViT-B-16 ViT-L-14 RN50 RN101; do
    python scripts/extract_model_config.py \
        --model $MODEL \
        --output configs/model_configs/${MODEL}.json
done

# 3. Transfer to each target model
for TARGET_MODEL in ViT-B-16 ViT-L-14 RN50 RN101; do
    echo "Transferring to $TARGET_MODEL..."

    # Load source hypernetwork, map to target model
    python src/eval_cross_model_transfer.py \
        --source-model ViT-B-32 \
        --target-model $TARGET_MODEL \
        --source-checkpoint checkpoints/cross_model/ViT-B-32_source/meta_trained.pt \
        --target-config configs/model_configs/${TARGET_MODEL}.json \
        --eval-datasets Flowers102,UCF101,CUB200 \
        --output checkpoints/cross_model/${TARGET_MODEL}_transfer_results.json
done

# 4. Analyze transfer performance
python scripts/analyze_cross_model_transfer.py \
    --source-model ViT-B-32 \
    --target-models ViT-B-16,ViT-L-14,RN50,RN101 \
    --results-dir checkpoints/cross_model/ \
    --plot-transfer-matrix \
    --output cross_model_transfer_analysis.pdf
```

### 4.5 Expected Results

**Transfer Performance (% of source model performance)**:
- ViT-B/16: 95-98% (very similar architecture)
- ViT-L/14: 85-90% (same family, larger scale)
- RN50: 70-80% (different architecture family)
- RN101: 75-85% (larger ResNet)

**Key Findings**:
- Transfer works best within same architecture family
- Larger models benefit more from transfer than smaller models
- Layer-type based mapping is crucial for cross-architecture transfer
- Fine-tuning on target architecture recovers most performance (5-10% gain)

**Files to Create**:
- `src/hypernetworks/architecture_agnostic.py`
- `src/utils/model_config.py`
- `src/eval_cross_model_transfer.py`
- `scripts/extract_model_config.py`
- `scripts/analyze_cross_model_transfer.py`

---

## 5. Domain Adaptation

### Overview
Adapt the text-to-coefficient/LoRA system to specialized domains like medical imaging, satellite imagery, or scientific microscopy where training data is scarce but text descriptions are available.

### 5.1 Domain-Specific Challenges

**Medical Imaging**:
- Limited public datasets (privacy constraints)
- Highly specialized vocabulary (pathology terms)
- Fine-grained classification (100+ disease subtypes)
- Class imbalance (rare diseases)

**Satellite Imagery**:
- Large image sizes (multi-spectral, high resolution)
- Temporal changes (seasonal variations)
- Geospatial context matters
- Limited labeled data (expensive annotation)

**Scientific Microscopy**:
- Novel organism/cell types (no existing task vectors)
- Highly domain-specific (fluorescence patterns, etc.)
- Small datasets (<1000 images)

### 5.2 Domain Adaptation Strategy

**Approach 1: Domain-Specific Text Encoder Fine-Tuning**

Instead of using general CLIP text encoder, fine-tune on domain-specific text:

**File: `src/domain_adaptation/finetune_text_encoder.py`**

```python
def finetune_text_encoder_for_domain(
    text_encoder_name: str,
    domain_texts: List[str],  # Domain-specific descriptions
    domain_labels: List[int],  # Optional: class labels
    output_path: str,
    epochs: int = 10
):
    """Fine-tune text encoder on domain-specific text.

    Args:
        text_encoder_name: Base text encoder
        domain_texts: List of domain-specific descriptions
        domain_labels: Optional labels for supervised fine-tuning
        output_path: Where to save fine-tuned encoder
    """
    # Load base encoder
    text_encoder = AutoModel.from_pretrained(text_encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)

    # Create contrastive learning objective
    # Positive pairs: different descriptions of same class
    # Negative pairs: descriptions of different classes

    optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in create_contrastive_batches(domain_texts, domain_labels):
            # Encode text
            text1, text2, label = batch

            tokens1 = tokenizer(text1, ...)
            tokens2 = tokenizer(text2, ...)

            emb1 = text_encoder(**tokens1).pooler_output
            emb2 = text_encoder(**tokens2).pooler_output

            # Contrastive loss
            if label == 1:  # Same class
                loss = 1 - F.cosine_similarity(emb1, emb2).mean()
            else:  # Different class
                loss = F.cosine_similarity(emb1, emb2).mean()

            loss.backward()
            optimizer.step()

    # Save
    text_encoder.save_pretrained(output_path)
    print(f"Fine-tuned text encoder saved to {output_path}")
```

**Approach 2: Domain-Specific Synthetic Data Generation**

Use specialized T2I models for medical/scientific imagery:

**Medical**:
- Use medical image synthesis models (e.g., MedSyn, ControlNet + medical data)
- Prompt engineering with medical terminology

**Satellite**:
- Use EarthGPT or similar geo-aware models
- Incorporate geospatial metadata in prompts

**Implementation**:

```bash
# Example: Medical domain
# 1. Fine-tune text encoder on medical terminology
python src/domain_adaptation/finetune_text_encoder.py \
    --base-encoder openai/clip-vit-base-patch32 \
    --domain medical \
    --domain-texts data/medical/terminology.txt \
    --output checkpoints/text_encoders/medical_clip

# 2. Generate synthetic medical images
python src/generate_synthetic_data.py \
    --dataset ChestXRay \
    --text-source medical \
    --t2i-backend medsyn \  # Specialized medical T2I
    --num-images-per-class 100 \
    --output-dir data/synthetic_images/medical

# 3. Meta-train hypernetwork on medical datasets
python src/learn_text_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets ChestXRay,Dermoscopy,Retinopathy,PathologySlides \
    --text-encoder checkpoints/text_encoders/medical_clip \
    --text-source medical \
    --hypernetwork-arch small \
    --meta-epochs 100

# 4. Evaluate on held-out medical tasks
python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset BloodCellTypes \  # New medical task
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/medical/meta_trained.pt \
    --text-source medical
```

### 5.3 Domain-Specific Text Description Generation

**File: `src/domain_adaptation/medical_text_generator.py`**

```python
class MedicalTextGenerator(TextDescriptionGenerator):
    """Generate medical image descriptions using domain-specific LLM."""

    def __init__(self, api_key: str, use_medical_knowledge: bool = True):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.use_medical_knowledge = use_medical_knowledge

    def generate_class_descriptions(
        self,
        class_name: str,  # e.g., "pneumonia"
        dataset_context: str,  # e.g., "chest X-ray dataset"
        num_descriptions: int = 10,
        diversity: str = "medium"
    ) -> List[str]:
        """Generate medical descriptions with domain knowledge."""

        # System prompt with medical expertise
        system_prompt = """You are a medical imaging expert. Generate detailed,
        clinically accurate descriptions of medical images for computer vision training.
        Include visual features visible in images (textures, patterns, anatomical landmarks)
        rather than clinical interpretations."""

        # User prompt
        user_prompt = f"""Generate {num_descriptions} diverse visual descriptions of
        {class_name} as seen in {dataset_context}.

        Focus on:
        - Visual appearance (brightness, texture, patterns)
        - Anatomical features
        - Image characteristics (contrast, clarity)
        - Comparative features (vs. normal)

        Keep descriptions visual and objective, suitable for text-to-image generation.
        One description per line."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        descriptions = response.choices[0].message.content.strip().split('\n')
        return [d.strip() for d in descriptions if d.strip()]
```

### 5.4 Evaluation on Domain-Specific Benchmarks

```bash
# Medical domain evaluation
DOMAIN=medical
python scripts/evaluate_domain_adaptation.py \
    --domain $DOMAIN \
    --source-datasets ChestXRay,Dermoscopy,Retinopathy \
    --target-datasets BloodCellTypes,LiverPathology,BrainMRI \
    --hypernetwork-checkpoint checkpoints/${DOMAIN}/meta_trained.pt \
    --baseline-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
    --output ${DOMAIN}_adaptation_results.json

# Compare domain-specific vs general hypernetwork
python scripts/plot_domain_adaptation_gains.py \
    --results ${DOMAIN}_adaptation_results.json \
    --output ${DOMAIN}_gains.pdf
```

### 5.5 Expected Results

**Medical Domain**:
- Domain-specific text encoder: +5-10% over general CLIP
- Domain-specific T2I: +10-15% over general Stable Diffusion
- Combined: +15-25% total improvement
- Critical for rare disease classification

**Satellite Domain**:
- Geospatial context crucial for good performance
- Multi-spectral image handling needed
- Temporal augmentation helps (+5-8%)

**Files to Create**:
- `src/domain_adaptation/finetune_text_encoder.py`
- `src/domain_adaptation/medical_text_generator.py`
- `src/domain_adaptation/satellite_text_generator.py`
- `scripts/evaluate_domain_adaptation.py`
- `scripts/plot_domain_adaptation_gains.py`

---

## 6. Task Composition

### Overview
Learn to compose multiple text descriptions or tasks together to create hybrid tasks or fine-grained specializations. For example, "outdoor + daytime + sunny weather" for satellite imagery or "elderly + female + chest X-ray" for medical imaging.

### 6.1 Compositional Text Encoding

**Architecture**: Hierarchical text encoder that understands task composition.

**File: `src/hypernetworks/compositional_text_encoder.py`**

```python
class CompositionalTextEncoder(nn.Module):
    """Text encoder that understands task composition.

    Instead of encoding entire descriptions holistically, this model
    learns to compose atomic concepts (e.g., "object", "lighting", "viewpoint").
    """

    def __init__(
        self,
        base_encoder_name: str,
        num_concepts: int = 50,  # Number of atomic concepts
        concept_dim: int = 128
    ):
        super().__init__()

        # Base text encoder
        self.base_encoder = AutoModel.from_pretrained(base_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_encoder_name)

        # Concept decomposition module
        # Maps text embedding to concept weights
        self.concept_decomposer = nn.Sequential(
            nn.Linear(self.base_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_concepts),
            nn.Softmax(dim=-1)  # Weights over concepts
        )

        # Concept embeddings (learnable)
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )

        # Composition module
        # Combines concept embeddings weighted by importance
        self.composer = nn.Sequential(
            nn.Linear(concept_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, text_descriptions: List[str]) -> torch.Tensor:
        """Encode text compositionally.

        Args:
            text_descriptions: List of text descriptions

        Returns:
            Compositional embeddings [batch, 512]
        """
        # Base encoding
        tokens = self.tokenizer(text_descriptions, padding=True, truncation=True, return_tensors="pt")
        base_emb = self.base_encoder(**tokens).pooler_output  # [batch, hidden_size]

        # Decompose into concepts
        concept_weights = self.concept_decomposer(base_emb)  # [batch, num_concepts]

        # Weighted combination of concept embeddings
        composed = torch.matmul(concept_weights, self.concept_embeddings)  # [batch, concept_dim]

        # Final composition
        output = self.composer(composed)  # [batch, 512]

        return output, concept_weights

    def compose_multiple_tasks(
        self,
        task_descriptions: List[List[str]],  # List of task description lists
        composition_weights: torch.Tensor  # [num_tasks]
    ) -> torch.Tensor:
        """Compose multiple tasks with given weights.

        Args:
            task_descriptions: List of description lists for each task
            composition_weights: Weights for each task

        Returns:
            Composed embedding
        """
        task_embeddings = []
        task_concepts = []

        # Encode each task
        for descriptions in task_descriptions:
            emb, concepts = self.forward(descriptions)
            # Pool across descriptions
            task_embeddings.append(emb.mean(dim=0))
            task_concepts.append(concepts.mean(dim=0))

        task_embeddings = torch.stack(task_embeddings)  # [num_tasks, 512]
        task_concepts = torch.stack(task_concepts)  # [num_tasks, num_concepts]

        # Weighted composition
        composed_emb = (task_embeddings * composition_weights.unsqueeze(-1)).sum(dim=0)
        composed_concepts = (task_concepts * composition_weights.unsqueeze(-1)).sum(dim=0)

        return composed_emb, composed_concepts
```

### 6.2 Task Composition Examples

**Example 1: Fine-Grained Classification**

Task: Classify bird species by composition of "bird type + plumage color + habitat"

```python
# Define atomic tasks
tasks = {
    'bird_type': ['sparrow', 'eagle', 'penguin', 'hummingbird'],
    'plumage': ['brown plumage', 'colorful plumage', 'black and white plumage'],
    'habitat': ['forest', 'ocean', 'desert', 'urban']
}

# Compose: "sparrow" + "brown" + "urban" = house sparrow
composed_descriptions = compositional_encoder.compose_multiple_tasks(
    task_descriptions=[
        ['a sparrow', 'a small songbird'],  # bird_type
        ['brown plumage', 'earth-toned feathers'],  # plumage
        ['urban habitat', 'city environment']  # habitat
    ],
    composition_weights=torch.tensor([0.5, 0.3, 0.2])  # Emphasize bird type
)

# Use composed description for coefficient prediction
coefs = hypernetwork.forward_from_embedding(composed_descriptions)
```

**Example 2: Multi-Attribute Composition**

Task: Medical imaging with "anatomy + pathology + imaging modality"

```python
# Compose: "lung" + "pneumonia" + "X-ray"
descriptions = compositional_encoder.compose_multiple_tasks(
    task_descriptions=[
        ['lung tissue', 'pulmonary region'],
        ['pneumonia', 'lung infection', 'consolidation'],
        ['X-ray image', 'radiograph', 'chest X-ray']
    ],
    composition_weights=torch.tensor([0.3, 0.5, 0.2])
)
```

### 6.3 Meta-Learning for Task Composition

Train the hypernetwork to handle compositional descriptions:

```python
def meta_train_compositional(args):
    """Meta-train with compositional tasks."""

    # Create compositional text encoder
    text_encoder = CompositionalTextEncoder(
        base_encoder_name="openai/clip-vit-base-patch32",
        num_concepts=50,
        concept_dim=128
    )

    # Create hypernetwork with compositional encoder
    hypernetwork = CompositionalHypernetwork(
        text_encoder=text_encoder,
        num_task_vectors=21,
        num_blocks=200,
        hidden_dims=[512, 256]
    )

    # Meta-training loop with compositional episodes
    for epoch in range(args.meta_epochs):
        for episode in range(args.episodes_per_epoch):
            # Sample a base task
            base_task = random.choice(args.meta_train_datasets)

            # Sample compositional attributes
            # E.g., "Cars" + "red color" + "side view"
            attribute_tasks = random.sample(args.attribute_tasks, k=2)

            # Compose task descriptions
            composed_desc = text_encoder.compose_multiple_tasks(
                task_descriptions=[
                    load_text_descriptions(base_task),
                    load_text_descriptions(attribute_tasks[0]),
                    load_text_descriptions(attribute_tasks[1])
                ],
                composition_weights=torch.tensor([0.6, 0.2, 0.2])
            )

            # Predict coefficients for composed task
            coefs = hypernetwork.forward_from_embedding(composed_desc)

            # Evaluate
            loss = evaluate_with_composition(coefs, base_task, attribute_tasks)
            loss.backward()
            optimizer.step()
```

### 6.4 Evaluation Protocol

```bash
# 1. Meta-train compositional hypernetwork
python src/learn_compositional_text_to_coef.py \
    --model ViT-B-32 \
    --meta-train-datasets <base_tasks> \
    --attribute-tasks <attributes> \
    --num-concepts 50 \
    --meta-epochs 150

# 2. Evaluate compositional generalization
# Test if model can compose unseen combinations
python src/eval_compositional_generalization.py \
    --model ViT-B-32 \
    --base-tasks Cars,Birds,Dogs \
    --attributes color,viewpoint,background,lighting \
    --hypernetwork-checkpoint checkpoints/compositional/meta_trained.pt \
    --output compositional_generalization.json

# 3. Fine-grained classification test
# Create synthetic fine-grained datasets via composition
python scripts/create_finegrained_via_composition.py \
    --base-dataset Birds \
    --attributes species,plumage,habitat,season \
    --num-compositions 100 \
    --output data/finegrained/birds_composed

python src/eval_text_adaptation.py \
    --model ViT-B-32 \
    --dataset BirdsComposed \
    --approach hypernetwork \
    --hypernetwork-checkpoint checkpoints/compositional/meta_trained.pt \
    --use-compositional-encoding
```

### 6.5 Expected Results

**Compositional Generalization**:
- Model should handle novel attribute combinations
- Performance: 70-80% of supervised fine-tuning on fine-grained tasks
- Most improvement on long-tail classes (rare attribute combinations)

**Concept Discovery**:
- Learned concepts should be interpretable (color, shape, texture, etc.)
- Visualization: t-SNE of concept embeddings clusters by semantic similarity

**Ablation**:
- Compositional encoder vs baseline: +10-15% on fine-grained tasks
- More concepts (50 vs 20): +3-5% but diminishing returns after 50

**Files to Create**:
- `src/hypernetworks/compositional_text_encoder.py`
- `src/hypernetworks/compositional_hypernetwork.py`
- `src/learn_compositional_text_to_coef.py`
- `src/eval_compositional_generalization.py`
- `scripts/create_finegrained_via_composition.py`
- `scripts/visualize_learned_concepts.py`

---

## Summary

This extension roadmap provides detailed implementation plans for 6 major directions to extend the text-based LoRA adaptation system. Each section includes:

1. **Clear objectives** and motivation
2. **Architecture designs** with code snippets
3. **Implementation steps** with bash commands
4. **Evaluation protocols** and expected results
5. **List of files to create**

**Priority Recommendations**:

1. **High Priority** (Next 3 months):
   - Ablation studies (understand what works)
   - Multi-modal hypernetwork (significant practical improvement)

2. **Medium Priority** (6 months):
   - Full LoRA prediction (more expressive, research contribution)
   - Domain adaptation (practical application to specialized fields)

3. **Long-term** (1 year):
   - Cross-model generalization (theoretical contribution)
   - Task composition (advanced capability, niche applications)

**Next Steps**:
1. Choose one extension based on research goals
2. Implement core components following this plan
3. Run ablations and comparisons
4. Write paper sections documenting findings
