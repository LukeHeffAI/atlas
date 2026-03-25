"""Multi-Modal Hypernetwork for aTLAS Coefficient Prediction.

This module implements a hypernetwork that predicts aTLAS coefficients from
both text descriptions AND few example images. It extends the text-only
TextToCoefHypernetwork by adding an image branch and a learned fusion module,
enabling the model to leverage complementary semantic (text) and visual (image)
signals for more accurate coefficient prediction.

Architecture overview:

    Text Branch:                Image Branch:
      Text Descriptions    →      Support Images (N classes × K shots)
            ↓                              ↓
      CLIP Text Encoder          CLIP Image Encoder
            ↓                              ↓
        [text_dim]                   [image_dim]
            ↓                              ↓
       Text Projection             Shot Pooling (mean / attention)
        [proj_dim]                     [image_dim]
            ↓                       Image Projection
            ↓                         [proj_dim]
            └─────────── Fusion ──────────┘
                            ↓
                      [fusion_dim]
                            ↓
                     Post-Fusion MLP
                            ↓
                  Output: Coefficients
                  [num_task_vectors × num_blocks]

When no support images are provided, the model gracefully degrades to
text-only prediction, matching the interface of TextToCoefHypernetwork.

References:
    - aTLAS: "Knowledge Composition using Task Vectors with Learned
      Anisotropic Scaling" (NeurIPS 2024)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .base import BaseHypernetwork


class _HFVisionEncoder(nn.Module):
    """Thin wrapper that combines HF CLIPVisionModel + visual_projection.

    HuggingFace splits the vision encoder and projection layer into
    separate modules.  This wrapper combines them so that calling
    ``encoder(images)`` returns projected features of the same
    dimensionality as OpenCLIP's ``clip_model.visual(images)``.
    """

    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # [B, hidden_size]
        return self.visual_projection(pooled)  # [B, proj_dim]


class AttentionShotPooling(nn.Module):
    """Attention-based pooling over K support images per class.

    Instead of simple mean pooling, this module learns to weight the
    contribution of each support image via scaled dot-product attention
    with a learnable query vector.

    Args:
        embed_dim: Dimension of image embeddings.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.query, std=0.02)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool across the shot dimension using learned attention.

        Args:
            x: Image embeddings of shape [batch, num_shots, embed_dim].

        Returns:
            Pooled embeddings of shape [batch, embed_dim].
        """
        # Expand query to batch size
        query = self.query.expand(x.size(0), -1, -1)  # [B, 1, D]

        # Attention scores
        attn = torch.bmm(query, x.transpose(1, 2)) * self.scale  # [B, 1, K]
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum
        pooled = torch.bmm(attn, x).squeeze(1)  # [B, D]
        return pooled


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion where text attends to per-shot image features.

    Implements multi-head cross-attention where the text embedding serves as
    the query and individual shot-level image embeddings serve as keys/values.
    This is more expressive than attending to a single pooled image vector,
    because the attention mechanism can learn to weight different support
    images differently depending on the text context.

    When ``per_shot_features`` are not available (i.e. images have already
    been pooled), the module falls back to single-token cross-attention.

    Args:
        embed_dim: Dimension of projected embeddings.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor,
        per_shot_image_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse text and image via cross-attention.

        Args:
            text_proj: Text projection of shape [batch, proj_dim].
            image_proj: Pooled image projection of shape [batch, proj_dim]
                (used as fallback if per_shot_image_proj is None).
            per_shot_image_proj: Optional per-shot image projections of shape
                [batch, num_shots, proj_dim]. When available, these serve as
                keys/values for richer cross-attention.

        Returns:
            Fused representation of shape [batch, proj_dim].
        """
        text_q = text_proj.unsqueeze(1)  # [B, 1, D]

        if per_shot_image_proj is not None:
            image_kv = per_shot_image_proj  # [B, K, D]
        else:
            image_kv = image_proj.unsqueeze(1)  # [B, 1, D]

        attended, _ = self.cross_attn(text_q, image_kv, image_kv)
        # Residual connection + layer norm
        fused = self.norm(attended.squeeze(1) + text_proj)
        return fused


class MultiModalHypernetwork(BaseHypernetwork):
    """Hypernetwork that predicts aTLAS coefficients from text + images.

    This architecture combines a text branch (CLIP text encoder) with an
    image branch (CLIP image encoder) through a learned fusion module. It
    can operate in three modes:

    1. **Multi-modal**: Both text descriptions and support images provided.
       The two modalities are fused before predicting coefficients.
    2. **Text-only fallback**: Only text descriptions provided (no images).
       The image branch is bypassed and a learned text-only projection is used.
    3. **Dataset-level vs. per-class text**: Text can be aggregated across
       the entire dataset (one embedding) or kept per-class (aligned with
       per-class image embeddings).

    Args:
        text_encoder_name: HuggingFace model name for the text encoder.
        num_task_vectors: Number of source task vectors in the pool.
        num_blocks: Number of parameter blocks (required for blockwise mode).
        proj_dim: Dimension of the projection space for both modalities.
        fusion_mode: How to fuse text and image branches.
            One of "concat", "add", or "attention".
        hidden_dims: Hidden layer dimensions for the post-fusion MLP.
        dropout: Dropout rate applied in projection and MLP layers.
        use_blockwise: If True, predict per-block coefficients; otherwise global.
        freeze_text_encoder: If True, freeze text encoder weights.
        freeze_image_encoder: If True, freeze image encoder weights.
        image_pooling: How to pool across K support shots.
            One of "mean" or "attention".
        text_input_mode: How to handle text descriptions.
            "dataset" aggregates all text into one embedding;
            "per_class" keeps per-class text (aligned with per-class images).
    """

    def __init__(
        self,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        num_task_vectors: int = 22,
        num_blocks: Optional[int] = None,
        proj_dim: int = 256,
        fusion_mode: str = "concat",
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_blockwise: bool = True,
        freeze_text_encoder: bool = True,
        freeze_image_encoder: bool = True,
        image_pooling: str = "mean",
        text_input_mode: str = "dataset",
        clip_backend: str = "clip",
    ):
        super().__init__()

        # Store config
        self.text_encoder_name = text_encoder_name
        self._clip_backend = clip_backend
        self.num_task_vectors = num_task_vectors
        self.num_blocks = num_blocks
        self.proj_dim = proj_dim
        self.fusion_mode = fusion_mode
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        self.use_blockwise = use_blockwise
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_image_encoder = freeze_image_encoder
        self.image_pooling = image_pooling
        self.text_input_mode = text_input_mode

        # --- Text encoder branch ---
        self._load_text_encoder()

        # --- Image encoder branch (uses the same CLIP model's visual tower) ---
        self._load_image_encoder()

        # --- Projection layers ---
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Text-only fallback projection ---
        # When no images are available, map text projection directly to
        # the fusion output space so the downstream MLP sees consistent dims.
        if fusion_mode == "concat":
            fusion_dim = proj_dim * 2
        else:
            fusion_dim = proj_dim
        self.text_only_proj = nn.Sequential(
            nn.Linear(proj_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Shot pooling ---
        if image_pooling == "attention":
            self.shot_pooling = AttentionShotPooling(self.image_dim)
        elif image_pooling == "mean":
            self.shot_pooling = None  # Use simple mean
        else:
            raise ValueError(f"Unknown image_pooling: {image_pooling}")

        # --- Fusion module ---
        self.fusion_dim = fusion_dim
        if fusion_mode == "attention":
            self.fusion = CrossAttentionFusion(proj_dim, num_heads=8, dropout=dropout)
        elif fusion_mode in ("concat", "add"):
            self.fusion = None  # Handled in forward
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        # --- Post-fusion MLP ---
        layers = []
        prev_dim = fusion_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.post_fusion_mlp = nn.Sequential(*layers)

        # --- Output layer ---
        if use_blockwise:
            if num_blocks is None:
                raise ValueError("num_blocks required for blockwise coefficients")
            output_dim = num_task_vectors * num_blocks
        else:
            output_dim = num_task_vectors
        self.output = nn.Linear(prev_dim, output_dim)

        # Initialize output near zero for stable training start
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def _load_text_encoder(self):
        """Load and optionally freeze the CLIP text encoder."""
        try:
            from transformers import AutoTokenizer, CLIPTextModel, AutoModel
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        print(f"[MultiModal] Loading text encoder: {self.text_encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

        if "clip" in self.text_encoder_name.lower():
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name)
        else:
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

        # Determine text embedding dimension
        if hasattr(self.text_encoder.config, "text_config"):
            self.text_dim = self.text_encoder.config.text_config.hidden_size
        elif hasattr(self.text_encoder.config, "hidden_size"):
            self.text_dim = self.text_encoder.config.hidden_size
        else:
            self.text_dim = self.text_encoder.config.projection_dim

        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("[MultiModal] Text encoder frozen")

    def _load_image_encoder(self):
        """Load and optionally freeze the CLIP image encoder.

        Uses the configured CLIP backend (HuggingFace or OpenCLIP) for
        consistency with the rest of the aTLAS codebase. The image encoder
        is a standalone module that encodes support images into fixed-size
        embeddings.
        """
        from clip_backends import load_clip_model, HFCLIPWrapper

        # Map HuggingFace-style names to internal model names
        hf_to_internal = {
            "openai/clip-vit-base-patch32": "ViT-B-32",
            "openai/clip-vit-base-patch16": "ViT-B-16",
            "openai/clip-vit-large-patch14": "ViT-L-14",
        }

        model_name = hf_to_internal.get(self.text_encoder_name, "ViT-B-32")
        backend = getattr(self, "_clip_backend", "clip")

        print(f"[MultiModal] Loading image encoder: {model_name} (backend={backend})")
        clip_model, _, self.image_preprocess = load_clip_model(
            model_name, pretrained="openai", backend=backend
        )

        if isinstance(clip_model, HFCLIPWrapper):
            self.image_encoder = clip_model.clip_model.vision_model
            # HF vision_model output is hidden_size; use the visual_projection for final dim
            self.image_dim = clip_model.clip_model.visual_projection.out_features
            # Wrap to apply projection after the vision model
            vision_model = clip_model.clip_model.vision_model
            visual_projection = clip_model.clip_model.visual_projection
            self.image_encoder = _HFVisionEncoder(vision_model, visual_projection)
        else:
            self.image_encoder = clip_model.visual
            self.image_dim = self.image_encoder.output_dim

        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("[MultiModal] Image encoder frozen")

    def _encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        """Encode text descriptions into embeddings.

        Args:
            text_descriptions: List of N text strings.

        Returns:
            Text embeddings of shape [N, text_dim].
        """
        tokens = self.tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(next(self.text_encoder.parameters()).device)

        with torch.set_grad_enabled(not self.freeze_text_encoder):
            text_features = self.text_encoder(**tokens).pooler_output

        return text_features

    def _encode_images(
        self,
        support_images: torch.Tensor,
        return_per_shot: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Encode support images and pool across shots.

        Uses chunked encoding to avoid OOM on large support sets (e.g.
        ImageNet with 1000 classes × 16 shots = 16,000 images).

        Args:
            support_images: Tensor of shape [batch, num_shots, C, H, W].
            return_per_shot: If True, also return unpooled per-shot features
                (useful for cross-attention fusion).
            chunk_size: Maximum number of images to encode in one forward
                pass through the image encoder.

        Returns:
            If return_per_shot is False:
                Pooled image embeddings of shape [batch, image_dim].
            If return_per_shot is True:
                Tuple of (pooled [batch, image_dim],
                          per_shot [batch, num_shots, image_dim]).
        """
        batch_size, num_shots = support_images.shape[:2]

        # Flatten: [B * K, C, H, W]
        flat_images = support_images.view(-1, *support_images.shape[2:])
        total = flat_images.shape[0]

        # Chunked encoding to prevent OOM
        with torch.set_grad_enabled(not self.freeze_image_encoder):
            if total <= chunk_size:
                image_features = self.image_encoder(flat_images)
            else:
                chunks = []
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    chunks.append(self.image_encoder(flat_images[start:end]))
                image_features = torch.cat(chunks, dim=0)

        # Reshape: [B, K, image_dim]
        image_features = image_features.view(batch_size, num_shots, -1)

        # Pool across shots
        if self.shot_pooling is not None:
            pooled = self.shot_pooling(image_features)
        else:
            pooled = image_features.mean(dim=1)

        if return_per_shot:
            return pooled, image_features
        return pooled

    def _fuse(
        self,
        text_proj: torch.Tensor,
        image_proj: torch.Tensor,
        per_shot_image_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse text and image projections.

        Args:
            text_proj: Text projection of shape [batch, proj_dim].
            image_proj: Pooled image projection of shape [batch, proj_dim].
            per_shot_image_proj: Optional per-shot image projections
                of shape [batch, num_shots, proj_dim] for attention fusion.

        Returns:
            Fused representation of shape [batch, fusion_dim].
        """
        if self.fusion_mode == "concat":
            return torch.cat([text_proj, image_proj], dim=-1)
        elif self.fusion_mode == "add":
            return text_proj + image_proj
        elif self.fusion_mode == "attention":
            return self.fusion(text_proj, image_proj, per_shot_image_proj)
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(
        self,
        text_descriptions: List[str],
        support_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: text (+ optional images) → coefficients.

        Args:
            text_descriptions: List of text descriptions (batch_size strings).
            support_images: Optional support images of shape
                [batch, num_shots, C, H, W]. If None, falls back to text-only.

        Returns:
            Coefficients tensor of shape:
                - [batch_size, num_task_vectors, num_blocks] if blockwise
                - [batch_size, num_task_vectors] if global
        """
        # --- Text branch ---
        text_features = self._encode_text(text_descriptions)  # [B, text_dim]
        text_proj = self.text_proj(text_features)  # [B, proj_dim]

        # --- Image branch (or fallback) ---
        if support_images is not None:
            # For attention fusion, we need per-shot features as KV tokens
            need_per_shot = (self.fusion_mode == "attention")
            encode_result = self._encode_images(
                support_images, return_per_shot=need_per_shot
            )
            if need_per_shot:
                image_pooled, per_shot_features = encode_result
                per_shot_proj = self.image_proj(per_shot_features)  # [B, K, proj_dim]
            else:
                image_pooled = encode_result
                per_shot_proj = None

            image_proj = self.image_proj(image_pooled)  # [B, proj_dim]
            fused = self._fuse(text_proj, image_proj, per_shot_proj)  # [B, fusion_dim]
        else:
            # Text-only fallback
            fused = self.text_only_proj(text_proj)  # [B, fusion_dim]

        # --- Post-fusion MLP + output ---
        hidden = self.post_fusion_mlp(fused)
        coef_flat = self.output(hidden)

        # Reshape to coefficient tensor
        if self.use_blockwise:
            coef = coef_flat.view(-1, self.num_task_vectors, self.num_blocks)
        else:
            coef = coef_flat.view(-1, self.num_task_vectors)

        return coef

    def predict_for_dataset(
        self,
        dataset_descriptions: Dict[str, List[str]],
        aggregate: str = "mean",
        support_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict coefficients for an entire dataset.

        Supports two text input modes controlled by ``self.text_input_mode``:

        - **dataset**: All descriptions are concatenated and encoded together,
          producing a single dataset-level embedding that is then aggregated.
        - **per_class**: Descriptions for each class are encoded separately,
          producing per-class embeddings that can be aligned with per-class
          support images before aggregation.

        Args:
            dataset_descriptions: Dict mapping class names to text descriptions.
                Format: {"class1": ["desc1", ...], "class2": [...], ...}
            aggregate: How to aggregate multiple predictions ("mean", "max", "median").
            support_images: Optional support images.
                Accepted shapes:
                - [num_classes, num_shots, C, H, W] — per-class support images
                  (dataset mode pools across classes; per_class mode uses directly)
                - [1, num_shots, C, H, W] — single set of shots (broadcast)

        Returns:
            Single coefficient tensor for the dataset.
        """
        if self.text_input_mode == "dataset":
            return self._predict_dataset_level(
                dataset_descriptions, aggregate, support_images
            )
        elif self.text_input_mode == "per_class":
            return self._predict_per_class(
                dataset_descriptions, aggregate, support_images
            )
        else:
            raise ValueError(f"Unknown text_input_mode: {self.text_input_mode}")

    def _predict_dataset_level(
        self,
        dataset_descriptions: Dict[str, List[str]],
        aggregate: str,
        support_images: Optional[torch.Tensor],
        text_chunk_size: int = 64,
    ) -> torch.Tensor:
        """Predict by aggregating all descriptions into one embedding.

        To avoid OOM with many descriptions, text is processed in chunks
        while image features are pre-computed once and reused.
        """
        all_descriptions = []
        for descriptions in dataset_descriptions.values():
            all_descriptions.extend(descriptions)

        if not all_descriptions:
            raise ValueError("No descriptions provided")

        # Pre-compute image features once (not per-description).
        # If support_images is [num_classes, K, C, H, W], reshape to
        # [1, num_classes*K, C, H, W] so all class shots are pooled together
        # into a single dataset-level representation.
        image_pooled = None
        per_shot_proj = None
        if support_images is not None:
            if support_images.dim() == 5 and support_images.shape[0] > 1:
                nc, k = support_images.shape[:2]
                support_images = support_images.view(1, nc * k, *support_images.shape[2:])

            need_per_shot = (self.fusion_mode == "attention")
            encode_result = self._encode_images(
                support_images, return_per_shot=need_per_shot
            )
            if need_per_shot:
                image_pooled_raw, per_shot_features = encode_result
                per_shot_proj = self.image_proj(per_shot_features)
            else:
                image_pooled_raw = encode_result
            image_pooled = self.image_proj(image_pooled_raw)

        # Process text in chunks, fusing with pre-computed image features.
        # NOTE: no torch.no_grad() here — gradient flow is needed during
        # meta-training.  Callers handle no_grad for inference/validation.
        all_coefs = []
        for start in range(0, len(all_descriptions), text_chunk_size):
            chunk_descs = all_descriptions[start:start + text_chunk_size]
            text_features = self._encode_text(chunk_descs)
            text_proj = self.text_proj(text_features)

            if image_pooled is not None:
                # Expand pre-computed image features to match chunk size
                img_exp = image_pooled.expand(len(chunk_descs), -1)
                ps_exp = (per_shot_proj.expand(len(chunk_descs), -1, -1)
                          if per_shot_proj is not None else None)
                fused = self._fuse(text_proj, img_exp, ps_exp)
            else:
                fused = self.text_only_proj(text_proj)

            hidden = self.post_fusion_mlp(fused)
            coef_flat = self.output(hidden)
            if self.use_blockwise:
                coef = coef_flat.view(-1, self.num_task_vectors, self.num_blocks)
            else:
                coef = coef_flat.view(-1, self.num_task_vectors)
            all_coefs.append(coef)

        all_coefs = torch.cat(all_coefs, dim=0)
        return self._aggregate_coefs(all_coefs, aggregate)

    def _predict_per_class(
        self,
        dataset_descriptions: Dict[str, List[str]],
        aggregate: str,
        support_images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Predict by encoding each class separately, then aggregating."""
        class_coefs = []

        for class_idx, (class_name, descriptions) in enumerate(
            dataset_descriptions.items()
        ):
            if not descriptions:
                continue

            # Get per-class images if available
            class_images = None
            if support_images is not None and class_idx < support_images.shape[0]:
                class_images = support_images[class_idx].unsqueeze(0)  # [1, K, C, H, W]
                class_images = class_images.expand(len(descriptions), -1, -1, -1, -1)

            coefs = self.forward(descriptions, support_images=class_images)
            # Average across descriptions within this class
            class_coef = coefs.mean(dim=0, keepdim=True)
            class_coefs.append(class_coef)

        if not class_coefs:
            raise ValueError("No descriptions provided for any class")

        all_class_coefs = torch.cat(class_coefs, dim=0)
        return self._aggregate_coefs(all_class_coefs, aggregate)

    @staticmethod
    def _aggregate_coefs(coefs: torch.Tensor, aggregate: str) -> torch.Tensor:
        """Aggregate coefficient predictions."""
        if aggregate == "mean":
            return coefs.mean(dim=0, keepdim=True)
        elif aggregate == "max":
            return coefs.max(dim=0, keepdim=True).values
        elif aggregate == "median":
            return coefs.median(dim=0, keepdim=True).values
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

    def get_config(self) -> Dict:
        """Get configuration dictionary for saving/loading."""
        return {
            "text_encoder_name": self.text_encoder_name,
            "num_task_vectors": self.num_task_vectors,
            "num_blocks": self.num_blocks,
            "proj_dim": self.proj_dim,
            "fusion_mode": self.fusion_mode,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_blockwise": self.use_blockwise,
            "freeze_text_encoder": self.freeze_text_encoder,
            "freeze_image_encoder": self.freeze_image_encoder,
            "image_pooling": self.image_pooling,
            "text_input_mode": self.text_input_mode,
        }


def create_multimodal_hypernetwork_from_args(args, num_blocks: int):
    """Factory function to create a MultiModalHypernetwork from CLI arguments.

    Mirrors ``create_hypernetwork_from_args`` but for the multi-modal variant.

    Args:
        args: Parsed command-line arguments.
        num_blocks: Number of parameter blocks in the target model.

    Returns:
        MultiModalHypernetwork instance.
    """
    arch_configs = {
        "small": [512, 256],
        "medium": [512, 512, 256],
        "large": [768, 512, 512, 256],
    }
    hidden_dims = arch_configs.get(args.hypernetwork_arch, arch_configs["small"])

    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397",
        "SVHN", "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101",
        "Caltech101", "Caltech256", "FGVCAircraft", "Flowers102",
        "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    ]
    num_task_vectors = len(pool)

    hypernetwork = MultiModalHypernetwork(
        text_encoder_name="openai/clip-vit-base-patch32",
        num_task_vectors=num_task_vectors,
        num_blocks=num_blocks,
        proj_dim=getattr(args, "proj_dim", 256),
        fusion_mode=getattr(args, "fusion_mode", "concat"),
        hidden_dims=hidden_dims,
        dropout=0.1,
        use_blockwise=args.blockwise_coef,
        freeze_text_encoder=getattr(args, "freeze_text_encoder", True),
        freeze_image_encoder=getattr(args, "freeze_image_encoder", True),
        image_pooling=getattr(args, "image_pooling", "mean"),
        text_input_mode=getattr(args, "text_input_mode", "dataset"),
        clip_backend=getattr(args, "clip_backend", "clip"),
    )

    return hypernetwork
