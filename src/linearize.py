"""
Image encoders with first-order Taylor approximation.

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Guillermo Ortiz-Jimenez et al.,
https://github.com/gortizji/tangent_task_arithmetic
"""

import abc
import os

import torch
import torch.nn as nn
from torch.func import jvp, functional_call

from modeling import ImageEncoder
from utils import DotDict


def make_functional_with_buffers(model, disable_autograd_tracking=False):
    """Compatibility shim: replicate the old functorch API using torch.func utilities."""
    param_names = []
    params = []
    buffer_names = []
    buffers = []
    for name, p in model.named_parameters():
        param_names.append(name)
        params.append(p if not disable_autograd_tracking else p.detach().requires_grad_(p.requires_grad))
    for name, b in model.named_buffers():
        buffer_names.append(name)
        buffers.append(b)

    def func(params_list, buffers_list, *args, **kwargs):
        state = {}
        for n, p in zip(param_names, params_list):
            state[n] = p
        for n, b in zip(buffer_names, buffers_list):
            state[n] = b
        return functional_call(model, state, args, kwargs)

    return func, params, buffers


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, buffers, x: func0(params, buffers, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def _apply(self, fn, recurse=True):
        """Override method to relocate buffer list."""
        new_self = super()._apply(fn=fn, recurse=recurse)
        new_self.buffers0 = [fn(x) for x in new_self.buffers0]
        return new_self

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class LinearizedImageEncoder(abc.ABC, nn.Module):
    """Creates a linearized version of an image encoder."""

    def __init__(
        self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
    ):
        super().__init__()
        if image_encoder is None:
            image_encoder = ImageEncoder(args, keep_lang)
        if init_encoder is None:
            init_encoder = image_encoder

        # Copy the attributes from the image encoder.
        self.train_preprocess = image_encoder.train_preprocess
        self.val_preprocess = image_encoder.val_preprocess
        self.cache_dir = image_encoder.cache_dir

        self._model_name = self._get_name(args.model)
        self._clip_backend = getattr(args, "clip_backend", "clip")
        self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

    def _get_name(self, model_name):
        if "__pretrained__" in model_name:
            model_name, _ = model_name.split("__pretrained__", "")
        return model_name

    def forward(self, x):
        # use the taylorized version of the model.
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def save(self, filename):
        """Saves the linearized image encoder.

        We save the model name in the state dict so that we can load the
        correct model when loading the linearized image encoder. Directly using
        torch.save would not work becuse func0 is not serializable.

        Args:
            filename (str): The path to save the taylorized image encoder.
        """
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        state_dict = self.state_dict()
        state_dict["model_name"] = self._model_name
        state_dict["clip_backend"] = getattr(self, "_clip_backend", "openclip")

        torch.save(state_dict, filename)

    @classmethod
    def load(cls, filename):
        """Loads a linearized image encoder.

        It first loads the state dict with the model name and then creates the
        correct model and loads the state dict.

        Args:
            filename (str): The path to the taylorized image encoder.

        Returns:
            LinearizedImageEncoder: The loaded taylorized image encoder.
        """
        print(f"Loading image encoder from {filename}")
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)

        # ImageEncoder expects a DotDict
        clip_backend = state_dict.pop("clip_backend", "openclip")
        args = DotDict({
            "model": state_dict["model_name"],
            "clip_backend": clip_backend,
        })
        taylorized_encoder = cls(args)

        # Remove the model name from the state dict so that we can load the
        # model.
        state_dict.pop("model_name")
        taylorized_encoder.load_state_dict(state_dict)
        return taylorized_encoder
