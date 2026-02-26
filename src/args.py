"""
Argument list

Fred Zhang <frederic.zhang@adelaide.edu.au>
Australian Institute for Machine Learning

Modified from the codebase by Ilharco et al. and Guillermo Ortiz-Jimenez et al.,
at https://github.com/mlfoundations/task_vectors and
https://github.com/gortizji/tangent_task_arithmetic
"""

import argparse
import os

import torch

def int_or_float(value):
    if '.' in value:
        return float(value)
    return int(value)

def int_or_float_list(value):
    """Parse a single int/float or a comma-separated list of int/floats.

    Returns a scalar for single values (backward compatible) or a list for
    comma-separated values (e.g. '1,2,4,8,16').
    """
    if ',' in value:
        return [int_or_float(v.strip()) for v in value.split(',')]
    return int_or_float(value)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--eval-on-full",
        default=False,
        action="store_true",
        help="Evaluate on the full dataset, when the model is trained on one class."
    )
    parser.add_argument(
        "--loss-fn",
        default='entropy',
        type=str,
        help="Loss function to use.",
        choices=["entropy", "cross_entropy"]
    )
    parser.add_argument(
        "--lp-reg",
        default=None,
        type=int,
        choices=[1, 2],
        help="Regularisation applied to the learned coefficients."
    )
    parser.add_argument(
        "--blockwise-coef",
        default=False,
        action="store_true",
        help="Use different coefficients on different parameter blocks."
    )
    parser.add_argument(
        "--subsample",
        default=1.0,
        type=int_or_float_list,
        help="Subsample the datasets with a float or specify the number of shots with an integer. "
             "Supports comma-separated values (e.g. 1,2,4,8,16) to run multiple shot settings in one invocation."
    )
    parser.add_argument(
        "--control-threshold",
        default=0.95,
        type=float,
        help="Percentage of accuracy on the control dataset to maintain."
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Where to load zero-shot weights and task vectors",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default='results/',
        help="Where to save results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="Port for distributed training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Adapter trained with aTLAS",
        choices=["tip", "lpp", "tip_cot"],
    )
    parser.add_argument(
        "--finetuning-mode",
        default='standard',
        choices=["standard", "linear", "posthoc", "none"],
        help="Whether to use linearized models or not.",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Run atlas x K where the task vectors are randomly partitioned n times (few-shot only)",
    )

    # Text-to-Image arguments
    parser.add_argument(
        "--t2i-backend",
        type=str,
        default="stable_diffusion",
        choices=["stable_diffusion", "sdxl", "dalle", "dall-e"],
        help="Text-to-image backend to use for synthetic data generation",
    )
    parser.add_argument(
        "--t2i-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Specific model ID for T2I backend",
    )
    parser.add_argument(
        "--t2i-config",
        type=str,
        default=None,
        help="Path to T2I configuration YAML file",
    )
    parser.add_argument(
        "--num-images-per-class",
        type=int,
        default=50,
        help="Number of synthetic images to generate per class",
    )

    # Text description arguments
    parser.add_argument(
        "--text-source",
        type=str,
        default="manual",
        choices=["manual", "generated", "templates"],
        help="Source of text descriptions for T2I generation or hypernetwork",
    )
    parser.add_argument(
        "--text-variant",
        type=str,
        default=None,
        help="Variant of generated text (e.g., gpt4o, claude) when text-source=generated",
    )
    parser.add_argument(
        "--text-aggregate",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="How to aggregate multiple text descriptions for hypernetwork",
    )

    # Synthetic task vector arguments
    parser.add_argument(
        "--use-synthetic-tv",
        action="store_true",
        default=False,
        help="Use synthetic task vector generated from T2I images in addition to real task vectors",
    )
    parser.add_argument(
        "--synthetic-backend",
        type=str,
        default="stable_diffusion",
        help="T2I backend used to generate synthetic images (for loading synthetic checkpoints)",
    )
    parser.add_argument(
        "--synthetic-data-location",
        type=str,
        default="data/synthetic_images",
        help="Root directory for synthetic images generated by T2I models",
    )
    parser.add_argument(
        "--task-vector-source",
        type=str,
        default="real",
        choices=["real", "synthetic", "mixed"],
        help="Source of task vectors: real (standard fine-tuning), synthetic (T2I-generated), or mixed",
    )
    parser.add_argument(
        "--synthetic-epochs-scale",
        type=float,
        default=0.5,
        help="Scale factor for epochs when fine-tuning on synthetic data (default: 0.5)",
    )

    # Hypernetwork arguments
    parser.add_argument(
        "--hypernetwork-arch",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Hypernetwork architecture size for text-to-coefficient prediction",
    )
    parser.add_argument(
        "--hypernetwork-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained hypernetwork checkpoint",
    )
    parser.add_argument(
        "--freeze-text-encoder",
        action="store_true",
        default=True,
        help="Freeze text encoder during hypernetwork training",
    )
    parser.add_argument(
        "--init-from-hypernetwork",
        action="store_true",
        default=False,
        help="Initialize coefficients from hypernetwork predictions before fine-tuning",
    )

    # Meta-learning arguments
    parser.add_argument(
        "--meta-train-datasets",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated list of datasets for meta-training hypernetwork",
    )
    parser.add_argument(
        "--meta-val-datasets",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated list of datasets for meta-validation",
    )
    parser.add_argument(
        "--meta-lr",
        type=float,
        default=1e-4,
        help="Learning rate for meta-training (outer loop)",
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=1e-3,
        help="Inner loop learning rate for MAML-style meta-learning",
    )
    parser.add_argument(
        "--meta-epochs",
        type=int,
        default=100,
        help="Number of meta-training epochs",
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=20,
        help="Number of episodes (task samples) per meta-training epoch",
    )
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=4,
        help="Batch size for meta-training episodes",
    )

    # Target datasets for benchmarking
    parser.add_argument(
        "--target-datasets",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated list of target datasets for benchmarking",
    )

    # Text adaptation mode
    parser.add_argument(
        "--text-adaptation-mode",
        type=str,
        default="hypernetwork",
        choices=["synthetic", "hypernetwork", "both", "synthetic_pool"],
        help="Which text-based adaptation approach to use",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
