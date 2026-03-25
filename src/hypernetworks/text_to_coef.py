"""Text-to-Coefficient Hypernetwork.

This module implements a hypernetwork that predicts aTLAS coefficients from text descriptions.
The hypernetwork is meta-trained across multiple tasks and can then predict coefficients
for new tasks using only text descriptions (zero-shot) or be further fine-tuned (few-shot).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .base import BaseHypernetwork


class TextToCoefHypernetwork(BaseHypernetwork):
    """Hypernetwork that predicts aTLAS coefficients from text descriptions.

    Architecture:
    1. Text encoder (CLIP text encoder) - encodes text to embeddings
    2. MLP layers - maps text embeddings to coefficient space
    3. Output layer - produces blockwise or global coefficients

    This hypernetwork is designed to be meta-trained across multiple tasks,
    then used for zero-shot or few-shot adaptation on new tasks.
    """

    def __init__(
        self,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        num_task_vectors: int = 21,
        num_blocks: Optional[int] = None,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_blockwise: bool = True,
        freeze_text_encoder: bool = True
    ):
        """Initialize text-to-coefficient hypernetwork.

        Args:
            text_encoder_name: HuggingFace model name for text encoder
            num_task_vectors: Number of source task vectors in the pool
            num_blocks: Number of parameter blocks (auto-detected if None)
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
            dropout: Dropout rate (default: 0.1)
            use_blockwise: Predict blockwise coefficients vs global (default: True)
            freeze_text_encoder: Freeze text encoder weights (default: True)
        """
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.num_task_vectors = num_task_vectors
        self.num_blocks = num_blocks
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        self.use_blockwise = use_blockwise
        self.freeze_text_encoder = freeze_text_encoder

        # Load text encoder
        self._load_text_encoder()

        # Get text embedding dimension
        # Handle different model config structures (CLIP vs BERT)
        if hasattr(self.text_encoder.config, 'text_config'):
            # CLIP-style models have nested config
            self.text_dim = self.text_encoder.config.text_config.hidden_size
        elif hasattr(self.text_encoder.config, 'hidden_size'):
            # Standard transformers models
            self.text_dim = self.text_encoder.config.hidden_size
        else:
            # Fallback to projection dim for CLIP
            self.text_dim = self.text_encoder.config.projection_dim

        # Build MLP
        self._build_mlp()

        # Build output layer
        self._build_output_layer()

        # Initialize weights
        self._initialize_weights()

    def _load_text_encoder(self):
        """Load and configure text encoder."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        print(f"Loading text encoder: {self.text_encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

        # Use CLIPTextModel for CLIP models (they need text-only model)
        if 'clip' in self.text_encoder_name.lower():
            from transformers import CLIPTextModel
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_name)
        else:
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)

        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Text encoder frozen")

    def _build_mlp(self):
        """Build MLP layers."""
        layers = []
        prev_dim = self.text_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def _build_output_layer(self):
        """Build output layer for coefficient prediction."""
        prev_dim = self.hidden_dims[-1]

        if self.use_blockwise:
            if self.num_blocks is None:
                raise ValueError("num_blocks required for blockwise coefficients")
            output_dim = self.num_task_vectors * self.num_blocks
        else:
            output_dim = self.num_task_vectors

        self.output = nn.Linear(prev_dim, output_dim)

    def _initialize_weights(self):
        """Initialize weights near zero for stability."""
        # Output layer: small initialization near zero
        # This ensures the model starts near the pretrained model
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, text_descriptions: List[str]) -> torch.Tensor:
        """Forward pass: text descriptions → coefficients.

        Args:
            text_descriptions: List of text descriptions (batch_size strings)

        Returns:
            Coefficients tensor of shape:
                - [batch_size, num_task_vectors, num_blocks] if blockwise
                - [batch_size, num_task_vectors] if global
        """
        # Tokenize text
        tokens = self.tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP max length
            return_tensors="pt"
        ).to(next(self.text_encoder.parameters()).device)

        # Encode text
        with torch.set_grad_enabled(not self.freeze_text_encoder):
            text_features = self.text_encoder(**tokens).pooler_output

        # Predict coefficients
        hidden = self.mlp(text_features)
        coef_flat = self.output(hidden)

        # Reshape
        if self.use_blockwise:
            coef = coef_flat.view(-1, self.num_task_vectors, self.num_blocks)
        else:
            coef = coef_flat.view(-1, self.num_task_vectors)

        return coef

    def predict_for_dataset(
        self,
        dataset_descriptions: Dict[str, List[str]],
        aggregate: str = "mean"
    ) -> torch.Tensor:
        """Predict coefficients for an entire dataset.

        Args:
            dataset_descriptions: Dict mapping class names to text descriptions
                Format: {"class1": ["desc1", ...], "class2": [...], ...}
            aggregate: How to aggregate multiple descriptions
                Options: "mean", "max", "median"

        Returns:
            Single coefficient tensor for the dataset
        """
        # Collect all descriptions
        all_descriptions = []
        for class_name, descriptions in dataset_descriptions.items():
            all_descriptions.extend(descriptions)

        if not all_descriptions:
            raise ValueError("No descriptions provided")

        # Get coefficients for all descriptions.
        # NOTE: no torch.no_grad() here — gradient flow is needed during
        # meta-training.  Callers handle no_grad for inference/validation.
        all_coefs = self.forward(all_descriptions)

        # Aggregate across descriptions
        if aggregate == "mean":
            dataset_coef = all_coefs.mean(dim=0, keepdim=True)
        elif aggregate == "max":
            dataset_coef = all_coefs.max(dim=0, keepdim=True).values
        elif aggregate == "median":
            dataset_coef = all_coefs.median(dim=0, keepdim=True).values
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        return dataset_coef

    def get_config(self) -> Dict:
        """Get configuration dictionary for saving/loading.

        Returns:
            Configuration dictionary
        """
        return {
            'text_encoder_name': self.text_encoder_name,
            'num_task_vectors': self.num_task_vectors,
            'num_blocks': self.num_blocks,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'use_blockwise': self.use_blockwise,
            'freeze_text_encoder': self.freeze_text_encoder,
        }


def create_hypernetwork_from_args(args, num_blocks: int):
    """Create hypernetwork from command-line arguments.

    Args:
        args: Parsed arguments
        num_blocks: Number of parameter blocks in the model

    Returns:
        TextToCoefHypernetwork instance
    """
    # Determine hidden dimensions based on architecture size
    arch_configs = {
        'small': [512, 256],
        'medium': [512, 512, 256],
        'large': [768, 512, 512, 256],
    }

    hidden_dims = arch_configs.get(args.hypernetwork_arch, arch_configs['small'])

    # Count task vectors (assume all datasets in pool)
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    ]
    num_task_vectors = len(pool)

    hypernetwork = TextToCoefHypernetwork(
        text_encoder_name="openai/clip-vit-base-patch32",
        num_task_vectors=num_task_vectors,
        num_blocks=num_blocks,
        hidden_dims=hidden_dims,
        dropout=0.1,
        use_blockwise=args.blockwise_coef,
        freeze_text_encoder=args.freeze_text_encoder
    )

    return hypernetwork
