"""CLIP model backend abstraction.

Provides a unified interface for loading CLIP models from either
HuggingFace transformers (default) or OpenCLIP. This allows the
codebase to switch between backends while keeping the rest of the
pipeline unchanged.

The HuggingFace backend uses the original OpenAI CLIP weights hosted
on HuggingFace Hub, matching the model used in the base aTLAS paper.
The OpenCLIP backend is retained for backward compatibility with
existing checkpoints and for models not available via HuggingFace
(e.g., ResNet-based CLIP variants).
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

# Internal model name → HuggingFace model ID
HF_MODEL_MAP = {
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-L-14": "openai/clip-vit-large-patch14",
    "ViT-L-14-336": "openai/clip-vit-large-patch14-336",
}


class HFCLIPWrapper(nn.Module):
    """Wraps a HuggingFace CLIPModel to expose the same API as OpenCLIP.

    This wrapper provides ``encode_image``, ``encode_text``, ``logit_scale``,
    and ``tokenize`` so that downstream code (heads, composition, etc.) can
    use either backend interchangeably.
    """

    def __init__(self, clip_model, tokenizer):
        super().__init__()
        self.clip_model = clip_model
        self._tokenizer = tokenizer
        # Expose logit_scale at top level (nn.Parameter, same as OpenCLIP)
        self.logit_scale = clip_model.logit_scale
        # Expose text_model as 'transformer' so existing keep_lang logic works
        self.transformer = clip_model.text_model

    # ------------------------------------------------------------------
    # Tokenization (mirrors open_clip.tokenize)
    # ------------------------------------------------------------------
    def tokenize(self, texts):
        """Tokenize a list of strings, returning input_ids tensor [N, 77]."""
        tokens = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return tokens["input_ids"]

    # ------------------------------------------------------------------
    # Encoding helpers matching OpenCLIP's model API
    # ------------------------------------------------------------------
    def encode_image(self, images):
        """Encode preprocessed images → feature vectors."""
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        return self.clip_model.visual_projection(vision_outputs.pooler_output)

    def encode_text(self, token_ids):
        """Encode tokenized text → feature vectors.

        Args:
            token_ids: Long tensor of shape [N, 77] (from ``self.tokenize``).
        """
        attention_mask = (token_ids != self._tokenizer.pad_token_id).long()
        text_outputs = self.clip_model.text_model(
            input_ids=token_ids, attention_mask=attention_mask
        )
        return self.clip_model.text_projection(text_outputs.pooler_output)

    def forward(self, *args, **kwargs):
        return self.clip_model(*args, **kwargs)


# ----------------------------------------------------------------------
# Transform helpers
# ----------------------------------------------------------------------

def _build_hf_transforms(image_processor):
    """Build torchvision train/val transforms from a HF CLIPImageProcessor."""
    # Extract config from the processor
    if hasattr(image_processor, "crop_size"):
        crop = image_processor.crop_size
        crop_size = crop.get("height", crop) if isinstance(crop, dict) else crop
    else:
        crop_size = 224

    if hasattr(image_processor, "size"):
        sz = image_processor.size
        resize_size = sz.get("shortest_edge", sz.get("height", 224)) if isinstance(sz, dict) else sz
    else:
        resize_size = crop_size

    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])

    val_transform = T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    train_transform = T.Compose([
        T.RandomResizedCrop(crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


# ----------------------------------------------------------------------
# Public loading API
# ----------------------------------------------------------------------

def load_clip_model(name, pretrained="openai", backend="clip", cache_dir=None):
    """Load a CLIP model using the specified backend.

    Args:
        name: Model architecture string (e.g. ``"ViT-B-32"``).
        pretrained: Pretrained source.  For the HuggingFace backend this is
            ignored (always loads the canonical OpenAI weights).  For OpenCLIP
            this is forwarded as-is (default ``"openai"``).
        backend: ``"clip"`` for HuggingFace transformers (default) or
            ``"openclip"`` for the open-clip-torch package.
        cache_dir: Directory used to cache downloaded model files.

    Returns:
        ``(model, train_preprocess, val_preprocess)`` – the model object
        exposes ``.encode_image()``, ``.encode_text()``, ``.logit_scale``,
        and ``.tokenize()`` regardless of backend.
    """
    if backend == "openclip":
        return _load_openclip(name, pretrained, cache_dir)
    else:
        return _load_hf_clip(name, pretrained, cache_dir)


def _load_openclip(name, pretrained, cache_dir):
    import open_clip

    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, cache_dir=cache_dir
    )
    # Attach a tokenize convenience method so the API is uniform
    model.tokenize = open_clip.tokenize
    return model, train_preprocess, val_preprocess


def _load_hf_clip(name, pretrained, cache_dir):
    from transformers import CLIPModel, CLIPProcessor, CLIPConfig

    hf_name = HF_MODEL_MAP.get(name)
    if hf_name is None:
        raise ValueError(
            f"Model '{name}' is not available via the HuggingFace CLIP backend. "
            f"Supported models: {list(HF_MODEL_MAP.keys())}. "
            f"Use --clip-backend openclip for other architectures."
        )

    if pretrained is None:
        # Random initialization (no pretrained weights)
        print(f"Initializing HuggingFace CLIP model from scratch: {hf_name}")
        config = CLIPConfig.from_pretrained(hf_name, cache_dir=cache_dir)
        clip_model = CLIPModel(config)
    else:
        print(f"Loading HuggingFace CLIP model: {hf_name}")
        clip_model = CLIPModel.from_pretrained(hf_name, cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(hf_name, cache_dir=cache_dir)

    tokenizer = processor.tokenizer
    train_preprocess, val_preprocess = _build_hf_transforms(processor.image_processor)

    wrapper = HFCLIPWrapper(clip_model, tokenizer)
    return wrapper, train_preprocess, val_preprocess


def get_hf_model_name(internal_name):
    """Map an internal model name to its HuggingFace model ID.

    Returns ``None`` if the model is not available via HuggingFace.
    """
    return HF_MODEL_MAP.get(internal_name)
