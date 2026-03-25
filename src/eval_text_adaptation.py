"""Evaluate text-based adaptation approaches (synthetic images and hypernetwork).

This script evaluates models adapted using text-only methods on target datasets.

Usage:
    # Evaluate hypernetwork zero-shot
    python src/eval_text_adaptation.py \
        --model ViT-B-32 \
        --dataset Flowers102 \
        --approach hypernetwork \
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
        --text-source manual

    # Evaluate with synthetic task vector
    python src/eval_text_adaptation.py \
        --model ViT-B-32 \
        --dataset Flowers102 \
        --approach synthetic \
        --synthetic-backend stable_diffusion
"""

import os
import json
import torch
from args import parse_arguments, get_checkpoint_dir
from modeling import ImageEncoder, ImageClassifier
from task_vectors import NonLinearTaskVector
from hypernetworks.text_to_coef import TextToCoefHypernetwork
from composition import WeightedImageEncoder, TextConditionedWeightedImageEncoder
from datasets.registry import get_dataset
from heads import get_classification_head
from utils import load_text_descriptions
from eval import eval_single_dataset
from learn_few_shots import load_task_vectors


def evaluate_hypernetwork_approach(args):
    """Evaluate hypernetwork-based text adaptation.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary of results
    """
    print("=" * 80)
    print("Evaluating Hypernetwork Approach (Zero-Shot)")
    print("=" * 80)

    # Load hypernetwork
    print(f"Loading hypernetwork from: {args.hypernetwork_checkpoint}")
    hypernetwork = TextToCoefHypernetwork.load(
        args.hypernetwork_checkpoint,
        device="cuda"
    )
    hypernetwork.eval()

    # Load pretrained model
    print("Loading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # Load task vectors (all except target)
    print("Loading task vectors...")
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    ]

    task_vectors = []
    for dataset in pool:
        if dataset == args.dataset:
            continue  # Skip target dataset

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

        if os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint):
            tv = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            task_vectors.append(tv)

    print(f"Loaded {len(task_vectors)} task vectors")

    # Load text descriptions for target dataset
    print(f"Loading text descriptions for {args.dataset}...")
    text_descriptions = load_text_descriptions(args.dataset, args)

    # Create text-conditioned encoder
    print("Predicting coefficients from text...")
    weighted_encoder = TextConditionedWeightedImageEncoder(
        model=pretrained_model,
        task_vectors=task_vectors,
        hypernetwork=hypernetwork,
        text_descriptions=text_descriptions,
        text_aggregate=args.text_aggregate,
        blockwise=args.blockwise_coef
    )

    # Load classification head
    classification_head = get_classification_head(args, args.dataset)

    # Create classifier
    image_classifier = ImageClassifier(weighted_encoder, classification_head)
    image_classifier = image_classifier.cuda()

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = get_dataset(
        args.dataset,
        weighted_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    # Evaluate
    print("\nEvaluating...")
    results = eval_single_dataset(
        weighted_encoder,
        args.dataset,
        args,
        model=image_classifier
    )

    print("\n" + "=" * 80)
    print("Results:")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    if 'top5' in results:
        print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print("=" * 80)

    return results


def evaluate_synthetic_approach(args):
    """Evaluate synthetic task vector approach.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary of results
    """
    print("=" * 80)
    print("Evaluating Synthetic Task Vector Approach")
    print("=" * 80)

    # Load pretrained model
    print("Loading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # Load task vectors including synthetic
    print("Loading task vectors...")
    pool = [
        "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
        "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101", "Caltech101", "Caltech256",
        "FGVCAircraft", "Flowers102", "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
    ]

    task_vectors = []
    for dataset in pool:
        if dataset == args.dataset:
            continue

        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"

        if os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint):
            tv = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            task_vectors.append(tv)

    # Add synthetic task vector
    synthetic_pretrained = f"{args.save}/{args.dataset}Val/zeroshot.pt"
    synthetic_finetuned = f"{args.save}/{args.dataset}Val/synthetic_{args.synthetic_backend}_finetuned.pt"

    if os.path.exists(synthetic_finetuned):
        print(f"Loading synthetic task vector from {args.synthetic_backend}...")
        synthetic_tv = NonLinearTaskVector(synthetic_pretrained, synthetic_finetuned)
        task_vectors.append(synthetic_tv)
    else:
        print(f"Warning: Synthetic checkpoint not found at {synthetic_finetuned}")
        print("Generate synthetic images and fine-tune first")
        return {}

    print(f"Loaded {len(task_vectors)} task vectors (including synthetic)")

    # Create weighted encoder
    weighted_encoder = WeightedImageEncoder(
        model=pretrained_model,
        task_vectors=task_vectors,
        blockwise=args.blockwise_coef
    )

    # Load classification head
    classification_head = get_classification_head(args, args.dataset)

    # Create classifier
    image_classifier = ImageClassifier(weighted_encoder, classification_head)
    image_classifier = image_classifier.cuda()

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = get_dataset(
        args.dataset,
        weighted_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    # Evaluate
    print("\nEvaluating...")
    results = eval_single_dataset(
        weighted_encoder,
        args.dataset,
        args,
        model=image_classifier
    )

    print("\n" + "=" * 80)
    print("Results:")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    if 'top5' in results:
        print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print("=" * 80)

    return results


def evaluate_synthetic_pool_approach(args):
    """Evaluate hypernetwork on a fully synthetic task vector pool.

    Unlike the existing 'synthetic' mode which adds a synthetic target TV to the
    real pool, this mode replaces the entire pool with synthetic task vectors.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary of results
    """
    print("=" * 80)
    print("Evaluating Synthetic Pool + Hypernetwork Approach")
    print("=" * 80)

    if args.hypernetwork_checkpoint is None:
        raise ValueError("--hypernetwork-checkpoint required for synthetic_pool approach")

    # Load hypernetwork
    print(f"Loading hypernetwork from: {args.hypernetwork_checkpoint}")
    hypernetwork = TextToCoefHypernetwork.load(args.hypernetwork_checkpoint, device="cuda")
    hypernetwork.eval()

    # Load pretrained model
    print("Loading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # Load synthetic task vectors (all except target)
    backend = getattr(args, 'synthetic_backend', args.t2i_backend)
    print(f"Loading synthetic task vectors (backend={backend})...")
    all_synthetic_tvs = load_task_vectors(args, source="synthetic", backend=backend)

    task_vectors = [v for k, v in all_synthetic_tvs.items() if k != args.dataset]
    print(f"Loaded {len(task_vectors)} synthetic task vectors (excluding {args.dataset})")

    if len(task_vectors) == 0:
        print("Warning: No synthetic task vectors available")
        return {}

    # Load text descriptions for target dataset
    print(f"Loading text descriptions for {args.dataset}...")
    text_descriptions = load_text_descriptions(args.dataset, args)

    # Create text-conditioned encoder with synthetic pool
    print("Predicting coefficients from text (synthetic pool)...")
    weighted_encoder = TextConditionedWeightedImageEncoder(
        model=pretrained_model,
        task_vectors=task_vectors,
        hypernetwork=hypernetwork,
        text_descriptions=text_descriptions,
        text_aggregate=args.text_aggregate,
        blockwise=args.blockwise_coef,
    )

    # Load classification head
    classification_head = get_classification_head(args, args.dataset)

    # Create classifier
    image_classifier = ImageClassifier(weighted_encoder, classification_head)
    image_classifier = image_classifier.cuda()

    # Evaluate on real test set
    print("\nEvaluating on real test set...")
    results = eval_single_dataset(
        weighted_encoder,
        args.dataset,
        args,
        model=image_classifier,
    )

    print("\n" + "=" * 80)
    print("Results:")
    print(f"  Top-1 Accuracy: {results['top1']:.2f}%")
    if 'top5' in results:
        print(f"  Top-5 Accuracy: {results['top5']:.2f}%")
    print("=" * 80)

    return results


def main():
    args = parse_arguments()

    # Set save directory
    args.save = get_checkpoint_dir(args)

    # Evaluate based on approach
    if args.text_adaptation_mode == "hypernetwork":
        if args.hypernetwork_checkpoint is None:
            raise ValueError("--hypernetwork-checkpoint required for hypernetwork approach")
        results = evaluate_hypernetwork_approach(args)
    elif args.text_adaptation_mode == "synthetic":
        results = evaluate_synthetic_approach(args)
    elif args.text_adaptation_mode == "synthetic_pool":
        results = evaluate_synthetic_pool_approach(args)
    elif args.text_adaptation_mode == "both":
        print("Evaluating both approaches...\n")
        hyper_results = evaluate_hypernetwork_approach(args)
        print("\n")
        synth_results = evaluate_synthetic_approach(args)

        # Compare
        print("\n" + "=" * 80)
        print("Comparison:")
        print(f"  Hypernetwork: {hyper_results.get('top1', 0):.2f}%")
        print(f"  Synthetic: {synth_results.get('top1', 0):.2f}%")
        print("=" * 80)

        results = {
            'hypernetwork': hyper_results,
            'synthetic': synth_results
        }
    else:
        raise ValueError(f"Unknown text adaptation mode: {args.text_adaptation_mode}")

    # Save results
    output_dir = os.path.join(args.save, "text_adapted", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.text_adaptation_mode}_results.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
