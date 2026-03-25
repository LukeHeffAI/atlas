"""Definition of image encoders and classifier models.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import torch

import utils
from clip_backends import load_clip_model


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        elif "__init__" in args.model:
            print("Using random initialization.")
            name, pretrained = args.model.split("__init__")[0], None
        else:
            name = args.model
            pretrained = "openai"

        backend = getattr(args, "clip_backend", "clip")
        cache_dir = getattr(args, "clip_cache_dir", None)
        # Fall back to legacy openclip-cachedir when using the openclip backend
        if backend == "openclip" and cache_dir is None:
            cache_dir = getattr(args, "openclip_cachedir", None)

        (
            self.model,
            self.train_preprocess,
            self.val_preprocess,
        ) = load_clip_model(
            name, pretrained=pretrained, backend=backend, cache_dir=cache_dir
        )

        self.cache_dir = args.cache_dir

        if not keep_lang:
            # Remove text components to save memory
            if hasattr(self.model, "transformer"):
                delattr(self.model, "transformer")
            # For HF CLIP wrapper, also remove underlying text model
            if hasattr(self.model, "clip_model"):
                clip = self.model.clip_model
                for attr in ("text_model", "text_projection"):
                    if hasattr(clip, attr):
                        delattr(clip, attr)

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)
        return cls.load(model_name, state_dict)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs, return_features=False):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        if return_features:
            return outputs, features / features.norm(dim=-1, keepdim=True)
        return outputs

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)

class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, num):
        """Process inputs with different classification heads.

        Parameters:
        -----------
        inputs: Tensor
            Input images.
        num: List[int]
            Number of images for each head to process. The length must be
            equal to the number of heads.
        """
        features = self.image_encoder(inputs)
        split_features = features.split(num)
        outputs = [
            head(feat) for feat, head in
            zip(split_features, self.classification_heads)
        ]
        return outputs

    def __call__(self, inputs, num):
        return self.forward(inputs, num)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
