"""Benchmark synthetic vs real task vectors across multiple conditions.

Evaluates 5 conditions on each target dataset:
1. Real pool + learned coefficients
2. Synthetic pool + learned coefficients
3. Mixed pool (real + synthetic target TV) + learned coefficients
4. Real pool + hypernetwork coefficients
5. Synthetic pool + hypernetwork coefficients

All conditions evaluate on the real test set.

Usage:
    python src/benchmark_synthetic_vs_real.py \
        --model ViT-B-32 \
        --blockwise-coef \
        --t2i-backend stable_diffusion \
        --hypernetwork-checkpoint checkpoints/ViT-B-32/hypernetworks/text_to_coef/meta_trained.pt \
        --target-datasets CIFAR10,EuroSAT,Flowers102 \
        --subsample 4
"""

import os
import json
import time
import torch

from src.args import parse_arguments
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import NonLinearTaskVector
from src.composition import WeightedImageEncoder, TextConditionedWeightedImageEncoder
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.learn_few_shots import load_task_vectors, train, DATASET_POOL
from src.utils import load_text_descriptions


def evaluate_with_hypernetwork(args, task_vectors_list, target_dataset):
    """Evaluate using hypernetwork-predicted coefficients.

    Args:
        args: Parsed arguments (must have hypernetwork_checkpoint set).
        task_vectors_list: List of task vectors (already excluding target).
        target_dataset: Target dataset name (without "Val" suffix).

    Returns:
        Dictionary with evaluation metrics, or None on failure.
    """
    from src.hypernetworks.text_to_coef import TextToCoefHypernetwork

    if args.hypernetwork_checkpoint is None or not os.path.exists(args.hypernetwork_checkpoint):
        print(f"  Hypernetwork checkpoint not found: {args.hypernetwork_checkpoint}")
        return None

    hypernetwork = TextToCoefHypernetwork.load(args.hypernetwork_checkpoint, device="cuda")
    hypernetwork.eval()

    pretrained_model = ImageEncoder(args)

    text_descriptions = load_text_descriptions(target_dataset, args)

    weighted_encoder = TextConditionedWeightedImageEncoder(
        model=pretrained_model,
        task_vectors=task_vectors_list,
        hypernetwork=hypernetwork,
        text_descriptions=text_descriptions,
        text_aggregate=args.text_aggregate,
        blockwise=args.blockwise_coef,
    )

    classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(weighted_encoder, classification_head)
    model = model.cuda()

    results = eval_single_dataset(weighted_encoder, target_dataset, args)
    return results


def evaluate_with_learned_coefs(args, task_vectors_dict, target_dataset, comp_acc=None):
    """Evaluate using learned coefficients (via few-shot training).

    Delegates to the existing train() function from learn_few_shots.py.

    Args:
        args: Parsed arguments.
        task_vectors_dict: Dictionary mapping dataset name to task vector.
        target_dataset: Target dataset name (without "Val" suffix).
        comp_acc: Optional accumulator dict for results.

    Returns:
        Dictionary with evaluation metrics.
    """
    if comp_acc is None:
        comp_acc = {}

    # Set up args expected by train()
    args.target_dataset = target_dataset + "Val"
    args.rank = 0

    # Load or init zero-shot accuracy
    zs_acc_path = os.path.join(f"{args.save}/{target_dataset}Val/", "zeroshot_accuracies.json")
    if os.path.isfile(zs_acc_path):
        with open(zs_acc_path, 'r') as f:
            args.zs_acc = json.load(f)
        comp_acc[f"{target_dataset}Val_zeroshot"] = args.zs_acc.get(f"{target_dataset}Val", 0.0)
    else:
        if not hasattr(args, 'zs_acc'):
            args.zs_acc = {}

    comp_acc = train(task_vectors_dict, args, comp_acc)

    # Extract the result
    top1 = comp_acc.get(target_dataset, comp_acc.get(f"{target_dataset}Val", 0.0))
    return {"top1": top1}


def run_benchmark(args):
    """Run the full 5-condition benchmark.

    Args:
        args: Parsed arguments.

    Returns:
        Dictionary of results.
    """
    target_datasets = args.target_datasets
    if target_datasets is None:
        target_datasets = DATASET_POOL

    backend = args.t2i_backend

    print("=" * 100)
    print("Benchmark: Synthetic vs Real Task Vectors")
    print("=" * 100)
    print(f"Model: {args.model}")
    print(f"T2I Backend: {backend}")
    print(f"Target datasets: {target_datasets}")
    print(f"Subsample: {args.subsample}")
    print(f"Blockwise coef: {args.blockwise_coef}")
    print("=" * 100)

    # Load task vector pools
    print("\nLoading real task vectors...")
    real_tvs = load_task_vectors(args, source="real")
    print(f"  Loaded {len(real_tvs)} real task vectors")

    print(f"\nLoading synthetic task vectors (backend={backend})...")
    synthetic_tvs = load_task_vectors(args, source="synthetic", backend=backend)
    print(f"  Loaded {len(synthetic_tvs)} synthetic task vectors")

    results = {}
    incomplete_pools = {}

    for target_dataset in target_datasets:
        print("\n" + "=" * 80)
        print(f"Target: {target_dataset}")
        print("=" * 80)

        dataset_results = {}
        pool_info = {
            "real_available": target_dataset in real_tvs or any(
                d != target_dataset and d in real_tvs for d in DATASET_POOL
            ),
            "synthetic_available": target_dataset in synthetic_tvs or any(
                d != target_dataset and d in synthetic_tvs for d in DATASET_POOL
            ),
        }

        # Real TVs excluding target
        real_pool = {k: v for k, v in real_tvs.items() if k != target_dataset}
        synthetic_pool = {k: v for k, v in synthetic_tvs.items() if k != target_dataset}

        # Track which datasets are missing from each pool
        missing_real = [d for d in DATASET_POOL if d != target_dataset and d not in real_pool]
        missing_synthetic = [d for d in DATASET_POOL if d != target_dataset and d not in synthetic_pool]
        if missing_real or missing_synthetic:
            incomplete_pools[target_dataset] = {
                "missing_real": missing_real,
                "missing_synthetic": missing_synthetic,
            }

        # Condition 1: Real pool + learned coefficients
        print(f"\n  [1/5] Real pool + learned coefficients ({len(real_pool)} TVs)")
        try:
            args.epochs = args.target_epochs.get(target_dataset, 10)
            dataset_results["real_pool_learned_coef"] = evaluate_with_learned_coefs(
                args, real_pool, target_dataset
            )
            print(f"    Top-1: {dataset_results['real_pool_learned_coef']['top1']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
            dataset_results["real_pool_learned_coef"] = {"error": str(e)}

        # Condition 2: Synthetic pool + learned coefficients
        if len(synthetic_pool) > 0:
            print(f"\n  [2/5] Synthetic pool + learned coefficients ({len(synthetic_pool)} TVs)")
            try:
                args.epochs = args.target_epochs.get(target_dataset, 10)
                dataset_results["synthetic_pool_learned_coef"] = evaluate_with_learned_coefs(
                    args, synthetic_pool, target_dataset
                )
                print(f"    Top-1: {dataset_results['synthetic_pool_learned_coef']['top1']:.4f}")
            except Exception as e:
                print(f"    Failed: {e}")
                dataset_results["synthetic_pool_learned_coef"] = {"error": str(e)}
        else:
            print("\n  [2/5] Synthetic pool + learned coefficients: SKIPPED (no synthetic TVs)")
            dataset_results["synthetic_pool_learned_coef"] = {"error": "no synthetic task vectors"}

        # Condition 3: Mixed pool (real excluding target + synthetic target TV) + learned coefficients
        synthetic_target_path = f"{args.save}/{target_dataset}Val/synthetic_{backend}_finetuned.pt"
        pretrained_target_path = f"{args.save}/{target_dataset}Val/zeroshot.pt"
        if os.path.exists(synthetic_target_path) and os.path.exists(pretrained_target_path):
            print(f"\n  [3/5] Mixed pool + learned coefficients ({len(real_pool)} real + 1 synthetic target)")
            try:
                mixed_pool = dict(real_pool)
                mixed_pool[f"{target_dataset}_synthetic"] = NonLinearTaskVector(
                    pretrained_target_path, synthetic_target_path
                )
                args.epochs = args.target_epochs.get(target_dataset, 10)
                dataset_results["mixed_pool_learned_coef"] = evaluate_with_learned_coefs(
                    args, mixed_pool, target_dataset
                )
                print(f"    Top-1: {dataset_results['mixed_pool_learned_coef']['top1']:.4f}")
            except Exception as e:
                print(f"    Failed: {e}")
                dataset_results["mixed_pool_learned_coef"] = {"error": str(e)}
        else:
            print("\n  [3/5] Mixed pool + learned coefficients: SKIPPED (no synthetic target TV)")
            dataset_results["mixed_pool_learned_coef"] = {"error": "no synthetic target checkpoint"}

        # Condition 4: Real pool + hypernetwork coefficients
        if args.hypernetwork_checkpoint:
            print(f"\n  [4/5] Real pool + hypernetwork coefficients ({len(real_pool)} TVs)")
            try:
                real_tv_list = list(real_pool.values())
                dataset_results["real_pool_hypernetwork"] = evaluate_with_hypernetwork(
                    args, real_tv_list, target_dataset
                )
                if dataset_results["real_pool_hypernetwork"]:
                    print(f"    Top-1: {dataset_results['real_pool_hypernetwork']['top1']:.4f}")
                else:
                    dataset_results["real_pool_hypernetwork"] = {"error": "hypernetwork evaluation returned None"}
            except Exception as e:
                print(f"    Failed: {e}")
                dataset_results["real_pool_hypernetwork"] = {"error": str(e)}
        else:
            print("\n  [4/5] Real pool + hypernetwork: SKIPPED (no --hypernetwork-checkpoint)")
            dataset_results["real_pool_hypernetwork"] = {"error": "no hypernetwork checkpoint"}

        # Condition 5: Synthetic pool + hypernetwork coefficients
        if args.hypernetwork_checkpoint and len(synthetic_pool) > 0:
            print(f"\n  [5/5] Synthetic pool + hypernetwork coefficients ({len(synthetic_pool)} TVs)")
            try:
                synthetic_tv_list = list(synthetic_pool.values())
                dataset_results["synthetic_pool_hypernetwork"] = evaluate_with_hypernetwork(
                    args, synthetic_tv_list, target_dataset
                )
                if dataset_results["synthetic_pool_hypernetwork"]:
                    print(f"    Top-1: {dataset_results['synthetic_pool_hypernetwork']['top1']:.4f}")
                else:
                    dataset_results["synthetic_pool_hypernetwork"] = {"error": "hypernetwork evaluation returned None"}
            except Exception as e:
                print(f"    Failed: {e}")
                dataset_results["synthetic_pool_hypernetwork"] = {"error": str(e)}
        else:
            reason = "no hypernetwork checkpoint" if not args.hypernetwork_checkpoint else "no synthetic task vectors"
            print(f"\n  [5/5] Synthetic pool + hypernetwork: SKIPPED ({reason})")
            dataset_results["synthetic_pool_hypernetwork"] = {"error": reason}

        results[target_dataset] = dataset_results

    return results, incomplete_pools


def main():
    args = parse_arguments()

    # Default to all datasets if none specified
    if args.target_datasets is None:
        args.target_datasets = DATASET_POOL

    # Set up save directory
    if args.save is None:
        args.save = f"checkpoints/{args.model}"

    # Epoch map for few-shot training per dataset
    args.target_epochs = {
        "Cars": 10, "DTD": 10, "EuroSAT": 10, "GTSRB": 10, "MNIST": 10,
        "RESISC45": 10, "SUN397": 10, "SVHN": 10, "CIFAR10": 10, "CIFAR100": 10,
        "ImageNet": 10, "STL10": 10, "Food101": 10, "Caltech101": 10, "Caltech256": 10,
        "FGVCAircraft": 10, "Flowers102": 10, "OxfordIIITPet": 10, "CUB200": 10,
        "PascalVOC": 10, "Country211": 10, "UCF101": 10,
    }

    # Training configuration
    args.lr = 1e-1
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.num_workers = 4

    # Set up log directory
    args.logdir = os.path.join(args.logdir, args.model, "benchmark_synthetic_vs_real")
    os.makedirs(args.logdir, exist_ok=True)
    args.head_path = os.path.join(args.logdir, "benchmark_coefs.pt")
    args.log_path = os.path.join(args.logdir, "benchmark_log.json")

    if args.seed is None:
        args.seed = 1

    print("\nRunning benchmark...")
    start_time = time.time()
    results, incomplete_pools = run_benchmark(args)
    elapsed = time.time() - start_time

    # Build output
    output = {
        "model": args.model,
        "synthetic_backend": args.t2i_backend,
        "subsample": args.subsample,
        "blockwise_coef": args.blockwise_coef,
        "elapsed_seconds": elapsed,
        "results": results,
    }

    if incomplete_pools:
        output["incomplete_pools"] = incomplete_pools

    # Save results
    output_path = os.path.join(args.save, "benchmark_synthetic_vs_real.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    conditions = [
        "real_pool_learned_coef",
        "synthetic_pool_learned_coef",
        "mixed_pool_learned_coef",
        "real_pool_hypernetwork",
        "synthetic_pool_hypernetwork",
    ]

    header = f"{'Dataset':<20}" + "".join(f"{'Cond ' + str(i+1):<15}" for i in range(5))
    print(header)
    print("-" * len(header))

    for dataset, dataset_results in results.items():
        row = f"{dataset:<20}"
        for cond in conditions:
            val = dataset_results.get(cond, {})
            if "error" in val:
                row += f"{'N/A':<15}"
            else:
                row += f"{val.get('top1', 0.0)*100:<15.2f}"
        print(row)

    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
