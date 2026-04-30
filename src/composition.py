"""Models with learnable weights on task vectors 

Fred Zhang <frederic.zhang@adelaide.edu.au>
Paul Albert <paul.albert@adelaide.edu.au>

Australian Institute for Machine Learning
"""

import torch

from torch import nn
from torch.func import jvp, functional_call

def make_functional_with_buffers(model, disable_autograd_tracking=False):
    """Compatibility shim for the old functorch API, built on torch.func.

    Returns ``(func, params, buffers, param_names)``.  ``param_names`` is the
    ordered list matching ``params`` and lets callers align auxiliary tensors
    (e.g. task-vector deltas) by name rather than position — required when
    ``state_dict()`` and ``named_parameters()`` disagree (shared Parameters are
    listed once by the latter, multiple times by the former).
    """
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

    return func, params, buffers, param_names


def _align_task_vectors_by_name(task_vectors, params, param_names, context=""):
    """Return deltas aligned to ``param_names``, one inner list per task vector.

    Missing keys become zero tensors matching the corresponding param's shape
    and dtype.  Raises if fewer than half the names match — that almost always
    means the task vector was built for a different CLIP backend (e.g.
    OpenCLIP checkpoints used with ``--clip-backend=clip``, or vice versa).
    """
    aligned = []
    for tv_idx, tv in enumerate(task_vectors):
        matched = 0
        deltas = []
        for name, p in zip(param_names, params):
            v = tv.vector.get(name)
            if v is None:
                deltas.append(torch.zeros_like(p, dtype=torch.float16))
            else:
                if v.shape != p.shape:
                    raise RuntimeError(
                        f"Task vector {tv_idx} key '{name}' has shape "
                        f"{tuple(v.shape)} but model parameter has shape "
                        f"{tuple(p.shape)}. {context}"
                    )
                deltas.append(v)
                matched += 1
        if matched < max(1, len(param_names) // 2):
            raise RuntimeError(
                f"Task vector {tv_idx}: only {matched}/{len(param_names)} "
                f"parameter names matched. The checkpoint almost certainly "
                f"came from a different CLIP backend (HuggingFace vs OpenCLIP). "
                f"Check --clip-backend and --checkpoint-root. {context}"
            )
        aligned.append(deltas)
    return aligned

def mask_multiply(coefs, mask, params):
    if params.ndim != 3:
        return (coefs*0.).sum()
    if params.ndim == 1:
        return (coefs.sum(dim=-1) * params).sum(dim=0) #Classic block wise for 1-dim parameters
    if params.ndim == 2:
        coef_mask = torch.einsum('ij,jk->ik', coefs, mask.to(coefs))
        return torch.einsum('ik,ik->k', coef_mask, params)
    if params.ndim == 5: #Conv layer
        coef_mask = torch.einsum('ij,jdkcb->idkcb', coefs, mask.to(coefs))
        return torch.einsum('idkcb,idkcb->dkcb', coef_mask, params)
    
    coef_mask = torch.einsum('ij,jbk->ibk', coefs.to(mask), mask)
    return torch.einsum('ibk,ibk->bk', coef_mask, params)

class WeightedImageEncoder(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True, partition=None) -> None:
        """A wrapper class to enable compositions of task vectors

        Parameter:
        ----------
        model: nn.Module
            CLIP image encoder model.
        task_vectors: List[NonLinearTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        func, params, self.buffer, param_names = make_functional_with_buffers(model)
        # NOTE This is important to avoid the following error
        # NotImplementedError: Cannot copy out of meta tensor; no data!
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # Copy the attributes from the image encoder.
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = model.cache_dir

        # Align deltas to named_parameters order — state_dict order is NOT a
        # safe proxy (e.g. shared logit_scale appears twice in state_dict but
        # once in named_parameters; stale checkpoints may carry text keys).
        self.dparams = _align_task_vectors_by_name(
            task_vectors, params, param_names,
            context="(WeightedImageEncoder)"
        )
        self.blockwise = blockwise
        self.partition = partition
        if self.partition is not None:
            self.mask_mats = {}
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params), self.partition))
            for p in self.params:
                mask = torch.randint(self.partition, p.shape)
                self.mask_mats[p.shape] = torch.nn.Parameter(torch.nn.functional.one_hot(mask).moveaxis(-1, 0).half(), requires_grad=False)
        elif blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def _apply(self, fn, recurse=True):
        """Override method to relocate buffer list and task vector deltas."""
        new_self = super()._apply(fn=fn, recurse=recurse)
        new_self.buffer = [fn(x) for x in new_self.buffer]
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        if hasattr(self, 'mask_mats'):
            new_self.mask_mats = {k: fn(v) for k,v in new_self.mask_mats.items()}
        return new_self
    
    def train(self, mode=True):
        super().train(mode)

    def forward(self, x) -> torch.Tensor:
        if self.partition is not None:
            dparams = [mask_multiply(self.coef[:,i,], self.mask_mats[dp[0].shape], torch.cat([d.unsqueeze(0) for d in dp], dim=0)) for i, dp in enumerate(zip(*self.dparams))]
        elif self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, self.buffer, x)

class WeightedLinearizedModel(nn.Module):
    def __init__(self, model, task_vectors, blockwise=True) -> None:
        """A wrapper class to enable compositions of task vectors for linearised models
        
        Parameters:
        -----------
        model: nn.Module
            Linearised model using first-order Taylor expansion.
        task_vectors: List[LinearizedTaskVector]
            List of task vectors to learn coefficients for.
        blockwise: bool, default: True
            Learn a coefficient for each parameter block.
        """
        super().__init__()

        self.params0 = model.params0
        self.func0 = model.func0
        self.buffers0 = model.buffers0
        self._model_name = model._model_name

        self.dparams = [[tv.vector[k] for k in tv.vector if k.startswith('model.params.')] for tv in task_vectors]
        self.blockwise = blockwise
        if blockwise:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params0)))
        else:
            self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))

    def _apply(self, fn, recurse=True):
        """Override method to relocate buffer list and task vector deltas."""
        new_self = super()._apply(fn=fn, recurse=recurse)
        new_self.buffers0 = [fn(x) for x in new_self.buffers0]
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x) -> torch.Tensor:
        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, self.coef)]) for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, self.coef)]) for dp in zip(*self.dparams)]
        out, dp = jvp(
            lambda param: self.func0(param, self.buffers0, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


class TextConditionedWeightedImageEncoder(nn.Module):
    """Weighted image encoder with text-conditioned coefficients from hypernetwork.

    This extends WeightedImageEncoder to use a hypernetwork for predicting
    coefficients from text descriptions instead of learning them directly.
    """

    def __init__(
        self,
        model,
        task_vectors,
        hypernetwork=None,
        text_descriptions=None,
        text_aggregate="mean",
        blockwise=True
    ):
        """Initialize text-conditioned weighted encoder.

        Args:
            model: CLIP image encoder model
            task_vectors: List of task vectors
            hypernetwork: Optional pre-trained hypernetwork for coefficient prediction
            text_descriptions: Dict mapping class names to text descriptions
            text_aggregate: How to aggregate multiple descriptions ("mean", "max", "median")
            blockwise: Use blockwise coefficients (must match hypernetwork)
        """
        super().__init__()

        # Decompose model
        func, params, self.buffer, param_names = make_functional_with_buffers(model)
        self.func = lambda p, b, x: func(p, b, x)
        self.params = torch.nn.ParameterList(params)
        for p in self.params:
            p.requires_grad = False

        # Copy model attributes
        self.train_preprocess = model.train_preprocess
        self.val_preprocess = model.val_preprocess
        self.cache_dir = getattr(model, 'cache_dir', None)

        # Store task vector deltas aligned to named_parameters() order.
        self.dparams = _align_task_vectors_by_name(
            task_vectors, params, param_names,
            context="(TextConditionedWeightedImageEncoder)"
        )
        self.blockwise = blockwise

        # Hypernetwork for coefficient prediction
        self.hypernetwork = hypernetwork
        self.text_aggregate = text_aggregate

        if hypernetwork is None:
            # Standard learnable coefficients (same as WeightedImageEncoder)
            if blockwise:
                self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors), len(self.params)))
            else:
                self.coef = torch.nn.Parameter(torch.zeros(len(task_vectors),))
        else:
            # Use hypernetwork to predict coefficients
            if text_descriptions is None:
                raise ValueError("text_descriptions required when using hypernetwork")

            # Freeze hypernetwork during inference
            for p in self.hypernetwork.parameters():
                p.requires_grad = False

            # Predict coefficients from text descriptions
            with torch.no_grad():
                predicted_coef = self.hypernetwork.predict_for_dataset(
                    text_descriptions,
                    aggregate=text_aggregate
                )
                # Remove batch dimension and store as buffer (not trainable by default)
                self.register_buffer('coef', predicted_coef.squeeze(0))

            print(f"Initialized coefficients from hypernetwork (aggregate: {text_aggregate})")
            print(f"Coefficient shape: {self.coef.shape}")

    def _apply(self, fn, recurse=True):
        """Override to relocate buffer list and dparams when moving to device."""
        new_self = super()._apply(fn=fn, recurse=recurse)
        new_self.buffer = [fn(x) for x in new_self.buffer]
        new_self.dparams = [[fn(x) for x in tv] for tv in new_self.dparams]
        return new_self

    def forward(self, x, coef=None):
        """Forward pass with weighted task vector composition.

        Args:
            x: Input images
            coef: Optional external coefficients (for gradient flow from hypernetwork).
                  If None, uses self.coef.

        Returns:
            Model output
        """
        # Use external coef if provided (allows gradient flow from hypernetwork)
        active_coef = coef if coef is not None else self.coef

        if self.blockwise:
            dparams = [sum([p * c[i] for p, c in zip(dp, active_coef)])
                      for i, dp in enumerate(zip(*self.dparams))]
        else:
            dparams = [sum([p * c for p, c in zip(dp, active_coef)])
                      for dp in zip(*self.dparams)]

        new_params = [dp + p for dp, p in zip(dparams, self.params)]
        return self.func(new_params, self.buffer, x)

    def enable_coefficient_finetuning(self):
        """Enable fine-tuning of coefficients after hypernetwork initialization.

        This converts the buffer to a trainable parameter, allowing few-shot
        fine-tuning of the hypernetwork-initialized coefficients.
        """
        if hasattr(self, 'coef') and not isinstance(self.coef, torch.nn.Parameter):
            # Convert buffer to parameter
            coef_data = self.coef.clone()
            delattr(self, 'coef')
            self.coef = torch.nn.Parameter(coef_data)
            print("Enabled coefficient fine-tuning")
        else:
            print("Coefficients already trainable")
