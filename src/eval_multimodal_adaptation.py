"""Evaluate multi-modal hypernetwork adaptation.

This script evaluates a pre-trained MultiModalHypernetwork on held-out
datasets with varying numbers of support images.  It supports three
evaluation modes:

1. **multimodal**: Predict coefficients from text + K support images.
2. **text_only**: Predict coefficients from text alone (hypernetwork's
   graceful fallback mode), as a zero-shot baseline.
3. **sweep**: Evaluate across multiple shot counts (e.g. 0, 1, 2, 4, 8, 16)
   and produce a single results file suitable for plotting.

For each setting the script:
    - Loads the hypernetwork checkpoint.
    - Loads text descriptions and (optionally) samples support images.
    - Predicts aTLAS coefficients.
    - Composes the pretrained model using the predicted coefficients.
    - Evaluates on the full test set of the target dataset.

Usage:
    # Single evaluation with 4-shot support
    python src/eval_multimodal_adaptation.py \\
        --model ViT-B-32 \\
        --dataset Flowers102 \\
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/multimodal_to_coef/meta_trained.pt \\
        --num-shots 4 \\
        --eval-mode multimodal

    # Sweep across shot counts
    python src/eval_multimodal_adaptation.py \\
        --model ViT-B-32 \\
        --dataset Flowers102 \\
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/multimodal_to_coef/meta_trained.pt \\
        --eval-mode sweep

    # Text-only baseline (zero-shot)
    python src/eval_multimodal_adaptation.py \\
        --model ViT-B-32 \\
        --dataset Flowers102 \\
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/multimodal_to_coef/meta_trained.pt \\
        --eval-mode text_only
"""

import os
import json
import random

import torch
from args import parse_arguments
from modeling import ImageEncoder, ImageClassifier
from task_vectors import NonLinearTaskVector
from hypernetworks.multimodal_to_coef import MultiModalHypernetwork
from composition import TextConditionedWeightedImageEncoder
from datasets.registry import get_dataset
from heads import get_classification_head
from utils import load_text_descriptions
from eval import eval_single_dataset
from learn_few_shots import load_task_vectors


# Default task vector pool (all 22 datasets)
TASK_VECTOR_POOL = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397",
    "SVHN", "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101",
    "Caltech101", "Caltech256", "FGVCAircraft", "Flowers102",
    "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
]


def sample_support_images(dataset_name, num_shots, preprocess, args, seed=42):
    """Sample K-shot support images from a dataset's training split.

    Args:
        dataset_name: Target dataset name.
        num_shots: Number of images per class.
        preprocess: Image preprocessing transform.
        args: Parsed CLI arguments.
        seed: Random seed for reproducibility.

    Returns:
        support_images: Tensor of shape [num_classes, K, C, H, W].
    """
    dataset = get_dataset(
        dataset_name + "Val",
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )

    raw = dataset.train_dataset if hasattr(dataset, "train_dataset") else dataset

    # Build class → indices mapping
    if hasattr(raw, "targets"):
        targets = raw.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
    else:
        targets = [int(raw[i][1]) for i in range(len(raw))]

    class_indices = {}
    for idx, label in enumerate(targets):
        class_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    support_images = []

    for class_idx in sorted(class_indices.keys()):
        indices = class_indices[class_idx]
        if len(indices) >= num_shots:
            shot_indices = rng.sample(indices, num_shots)
        else:
            shot_indices = rng.choices(indices, k=num_shots)

        class_images = []
        for idx in shot_indices:
            img, _ = raw[idx]
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            class_images.append(img)
        support_images.append(torch.stack(class_images))

    return torch.stack(support_images)  # [C, K, ch, H, W]


def evaluate_multimodal(
    hypernetwork, pretrained_model, task_vectors, text_descriptions,
    support_images, args,
):
    """Run evaluation with multi-modal coefficient prediction.

    Args:
        hypernetwork: Trained MultiModalHypernetwork.
        pretrained_model: Pretrained CLIP image encoder.
        task_vectors: List of task vectors (excluding target).
        text_descriptions: Dict of class → descriptions.
        support_images: Support images tensor or None (for text-only).
        args: Parsed CLI arguments.

    Returns:
        Dictionary of evaluation results.
    """
    hypernetwork.eval()

    with torch.no_grad():
        predicted_coef = hypernetwork.predict_for_dataset(
            text_descriptions,
            aggregate=args.text_aggregate,
            support_images=support_images,
        )

    # Build composed model
    weighted_encoder = TextConditionedWeightedImageEncoder(
        model=pretrained_model,
        task_vectors=task_vectors,
        hypernetwork=None,
        blockwise=args.blockwise_coef,
    )
    weighted_encoder = weighted_encoder.cuda()

    # Set coefficients from prediction
    coef_data = predicted_coef.squeeze(0)
    weighted_encoder.register_buffer("coef", coef_data)

    # Build classifier
    classification_head = get_classification_head(args, args.dataset)
    image_classifier = ImageClassifier(weighted_encoder, classification_head)
    image_classifier = image_classifier.cuda()

    # Evaluate
    results = eval_single_dataset(
        weighted_encoder,
        args.dataset,
        args,
        model=image_classifier,
    )

    return results


def main():
    args = parse_arguments()

    if args.hypernetwork_checkpoint is None:
        raise ValueError("--hypernetwork-checkpoint required")
    if args.save is None:
        args.save = f"checkpoints/{args.model}"

    eval_mode = getattr(args, "eval_mode", "multimodal")
    num_shots = getattr(args, "num_shots", 4)

    # --- Load hypernetwork ---
    print(f"Loading hypernetwork from: {args.hypernetwork_checkpoint}")
    hypernetwork = MultiModalHypernetwork.load(
        args.hypernetwork_checkpoint, device="cuda"
    )
    hypernetwork.eval()

    # --- Load pretrained model ---
    print("Loading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # --- Load task vectors (exclude target dataset) ---
    print("Loading task vectors...")
    task_vectors_list = []
    for ds in TASK_VECTOR_POOL:
        if ds == args.dataset:
            continue
        pre = f"{args.save}/{ds}Val/zeroshot.pt"
        ft = f"{args.save}/{ds}Val/finetuned.pt"
        if os.path.exists(pre) and os.path.exists(ft):
            task_vectors_list.append(NonLinearTaskVector(pre, ft))
    print(f"Loaded {len(task_vectors_list)} task vectors")

    # --- Load text descriptions ---
    print(f"Loading text descriptions for {args.dataset}...")
    text_descriptions = load_text_descriptions(args.dataset, args)

    # --- Evaluation ---
    if eval_mode == "text_only":
        print("\n" + "=" * 80)
        print(f"Evaluating: TEXT-ONLY (zero-shot) on {args.dataset}")
        print("=" * 80)

        results = evaluate_multimodal(
            hypernetwork, pretrained_model, task_vectors_list,
            text_descriptions, support_images=None, args=args,
        )
        print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
        all_results = {"text_only": results}

    elif eval_mode == "multimodal":
        print("\n" + "=" * 80)
        print(f"Evaluating: MULTIMODAL ({num_shots}-shot) on {args.dataset}")
        print("=" * 80)

        support_images = sample_support_images(
            args.dataset, num_shots, pretrained_model.val_preprocess, args
        ).cuda()
        print(f"  Support images shape: {support_images.shape}")

        results = evaluate_multimodal(
            hypernetwork, pretrained_model, task_vectors_list,
            text_descriptions, support_images=support_images, args=args,
        )
        print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
        all_results = {"multimodal": {f"{num_shots}-shot": results}}

    elif eval_mode == "sweep":
        shot_counts = [0, 1, 2, 4, 8, 16]
        print("\n" + "=" * 80)
        print(f"Evaluating: SWEEP ({shot_counts} shots) on {args.dataset}")
        print("=" * 80)

        all_results = {"sweep": {}}

        for k in shot_counts:
            print(f"\n--- {k}-shot ---")
            if k == 0:
                support_images = None
            else:
                support_images = sample_support_images(
                    args.dataset, k, pretrained_model.val_preprocess, args
                ).cuda()

            results = evaluate_multimodal(
                hypernetwork, pretrained_model, task_vectors_list,
                text_descriptions, support_images=support_images, args=args,
            )
            print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
            all_results["sweep"][f"{k}-shot"] = results

        # Print summary table
        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  {'Shots':<8} {'Top-1':>8}")
        print(f"  {'-'*8} {'-'*8}")
        for k in shot_counts:
            acc = all_results["sweep"][f"{k}-shot"]["top1"]
            print(f"  {k:<8} {acc:>7.2f}%")
        print("=" * 80)

    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    # --- Save results ---
    output_dir = os.path.join(
        args.save, "multimodal_adapted", args.dataset
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{eval_mode}_results.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
