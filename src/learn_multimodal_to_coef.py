"""Meta-train a multi-modal hypernetwork to predict aTLAS coefficients.

This script implements episode-based meta-training for the
MultiModalHypernetwork, which predicts aTLAS coefficients from both text
descriptions and a small set of support images.  The approach follows the
standard meta-learning paradigm:

    Outer loop (meta-epochs):
        Inner loop (episodes):
            1. Sample a task (dataset) from the meta-train set.
            2. Sample text descriptions and K-shot support images for that task.
            3. Predict coefficients using the hypernetwork.
            4. Compose the pretrained model with task vectors weighted by
               the predicted coefficients.
            5. Evaluate the composed model on a validation batch.
            6. Backpropagate the loss through the coefficient predictions
               to update the hypernetwork.

The training objective is cross-entropy on the composed model's predictions.
Gradient flow is maintained from the loss, through the composed model's
forward pass, through the predicted coefficients, and back into the
hypernetwork's parameters.

Usage:
    python src/learn_multimodal_to_coef.py \\
        --model ViT-B-32 \\
        --meta-train-datasets CIFAR10,EuroSAT,MNIST,DTD,GTSRB \\
        --meta-val-datasets Cars,SUN397 \\
        --hypernetwork-arch small \\
        --fusion-mode concat \\
        --num-shots 4 \\
        --meta-epochs 100 \\
        --episodes-per-epoch 20 \\
        --blockwise-coef

See ``src/args.py`` for the full list of command-line arguments, including
multi-modal specific options (--fusion-mode, --num-shots, --image-pooling,
--text-input-mode, --variable-shots).
"""

import os
import json
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from args import parse_arguments, get_checkpoint_dir
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder
from src.task_vectors import NonLinearTaskVector
from src.hypernetworks.multimodal_to_coef import create_multimodal_hypernetwork_from_args
from src.composition import TextConditionedWeightedImageEncoder
from src.heads import get_classification_head
from src.utils import load_text_descriptions
from src.learn_few_shots import load_task_vectors
from src.meta_learning.multimodal_sampler import MultiModalEpisodeSampler


class EpisodeCache:
    """Cache for expensive objects that are reused across episodes.

    Avoids re-creating WeightedImageEncoder, classification heads, and
    dataloaders on every episode — the dominant cost in naive meta-training.
    """

    def __init__(self):
        self._weighted_encoders = {}  # keyed by frozenset of excluded datasets
        self._classification_heads = {}
        self._dataloaders = {}

    def get_weighted_encoder(self, pretrained_model, task_vectors, exclude_name, blockwise):
        """Get or create a cached WeightedImageEncoder."""
        key = exclude_name
        if key not in self._weighted_encoders:
            relevant_tvs = [tv for name, tv in task_vectors.items()
                            if name != exclude_name]
            encoder = TextConditionedWeightedImageEncoder(
                model=pretrained_model,
                task_vectors=relevant_tvs,
                hypernetwork=None,
                blockwise=blockwise,
            )
            encoder = encoder.cuda()
            self._weighted_encoders[key] = encoder
        return self._weighted_encoders[key]

    def get_classification_head(self, args, dataset_name):
        """Get or create a cached classification head."""
        key = dataset_name
        if key not in self._classification_heads:
            head = get_classification_head(args, dataset_name)
            head = head.cuda()
            self._classification_heads[key] = head
        return self._classification_heads[key]

    def get_dataloader(self, dataset_name, preprocess, args):
        """Get or create a cached dataloader."""
        key = dataset_name
        if key not in self._dataloaders:
            dataset = get_dataset(
                dataset_name,
                preprocess,
                location=args.data_location,
                batch_size=args.meta_batch_size,
            )
            raw = dataset.test_dataset if hasattr(dataset, "test_dataset") else dataset
            loader = torch.utils.data.DataLoader(
                raw,
                batch_size=args.meta_batch_size,
                shuffle=True,
                num_workers=2,
            )
            self._dataloaders[key] = loader
        return self._dataloaders[key]


def meta_train_episode(
    hypernetwork,
    task_vectors,
    dataset_name,
    pretrained_model,
    sampler,
    args,
    cache=None,
):
    """Execute a single meta-training episode.

    This is the inner-loop step: for one sampled task, predict coefficients
    using the hypernetwork, compose the model, and compute a loss on a
    validation batch.

    Args:
        hypernetwork: MultiModalHypernetwork instance (trainable).
        task_vectors: Dict mapping dataset names to NonLinearTaskVector.
        dataset_name: The target dataset for this episode.
        pretrained_model: Pretrained CLIP image encoder.
        sampler: MultiModalEpisodeSampler for sampling episodes.
        args: Parsed CLI arguments.
        cache: Optional EpisodeCache for reusing expensive objects.

    Returns:
        loss: Differentiable loss tensor (for backpropagation).
        accuracy: Validation accuracy on this episode (float, detached).
    """
    if cache is None:
        cache = EpisodeCache()

    # --- Sample episode data ---
    _, text_descriptions, support_images, support_labels = sampler.sample_episode(
        dataset_name=dataset_name
    )

    # Move support images to GPU
    support_images = support_images.cuda()  # [num_classes, K, C, H, W]

    # --- Predict coefficients via hypernetwork ---
    if hypernetwork.text_input_mode == "dataset":
        all_descriptions = []
        for descs in text_descriptions.values():
            all_descriptions.extend(descs)
        # Use all support images as a single set (keep class structure)
        # rather than averaging pixel values across classes
        predicted_coef = hypernetwork.predict_for_dataset(
            text_descriptions,
            aggregate=getattr(args, "text_aggregate", "mean"),
            support_images=support_images,  # [num_classes, K, C, H, W]
        )
    else:
        # Per-class mode
        predicted_coef = hypernetwork.predict_for_dataset(
            text_descriptions,
            aggregate=getattr(args, "text_aggregate", "mean"),
            support_images=support_images,
        )

    # --- Compose model with predicted coefficients (cached) ---
    weighted_encoder = cache.get_weighted_encoder(
        pretrained_model, task_vectors, dataset_name, args.blockwise_coef
    )

    # --- Evaluate on a validation batch (cached head + dataloader) ---
    classification_head = cache.get_classification_head(args, dataset_name + "Val")
    dataloader = cache.get_dataloader(
        dataset_name + "Val", weighted_encoder.val_preprocess, args
    )

    batch = next(iter(dataloader))
    images = batch[0].cuda() if isinstance(batch, (list, tuple)) else batch["images"].cuda()
    labels = batch[1].cuda() if isinstance(batch, (list, tuple)) else batch["labels"].cuda()

    # Forward through composed model — gradient flows through predicted_coef
    coef_for_forward = predicted_coef.squeeze(0)
    features = weighted_encoder(images, coef=coef_for_forward)
    logits = classification_head(features)

    loss = nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item() * 100

    return loss, accuracy


def meta_train(args):
    """Main meta-training loop for the multi-modal hypernetwork.

    Args:
        args: Parsed CLI arguments.
    """
    save_dir = os.path.join(args.save, "hypernetworks", "multimodal_to_coef")
    os.makedirs(save_dir, exist_ok=True)

    # --- Print configuration ---
    print("=" * 80)
    print("Meta-Training Multi-Modal Hypernetwork")
    print("=" * 80)
    print(f"  Model:              {args.model}")
    print(f"  Meta-train sets:    {args.meta_train_datasets}")
    print(f"  Meta-val sets:      {args.meta_val_datasets}")
    print(f"  Architecture:       {args.hypernetwork_arch}")
    print(f"  Fusion mode:        {getattr(args, 'fusion_mode', 'concat')}")
    print(f"  Image pooling:      {getattr(args, 'image_pooling', 'mean')}")
    print(f"  Text input mode:    {getattr(args, 'text_input_mode', 'dataset')}")
    print(f"  Num shots:          {getattr(args, 'num_shots', 4)}")
    print(f"  Variable shots:     {getattr(args, 'variable_shots', False)}")
    print(f"  Meta LR:            {args.meta_lr}")
    print(f"  Meta epochs:        {args.meta_epochs}")
    print(f"  Episodes/epoch:     {args.episodes_per_epoch}")
    print(f"  Blockwise coef:     {args.blockwise_coef}")
    print("=" * 80)

    # --- Load pretrained model ---
    print("\nLoading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # --- Load task vectors ---
    source = getattr(args, "task_vector_source", "real")
    backend = getattr(args, "t2i_backend", "stable_diffusion")
    print(f"\nLoading task vectors (source={source})...")
    all_task_vectors = load_task_vectors(args, source=source, backend=backend)

    pool = args.meta_train_datasets + args.meta_val_datasets
    task_vectors = {k: v for k, v in all_task_vectors.items() if k in pool}
    for dataset in pool:
        status = "Loaded" if dataset in task_vectors else "Missing (skipped)"
        print(f"  {dataset}: {status}")

    # --- Determine number of parameter blocks ---
    num_blocks = len(pretrained_model.model.state_dict())
    print(f"\nNumber of parameter blocks: {num_blocks}")

    # --- Create hypernetwork ---
    print(f"\nCreating multi-modal hypernetwork (arch: {args.hypernetwork_arch})...")
    hypernetwork = create_multimodal_hypernetwork_from_args(args, num_blocks)
    hypernetwork = hypernetwork.cuda()

    total_params = sum(p.numel() for p in hypernetwork.parameters())
    trainable_params = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # --- Create episode sampler ---
    num_shots = getattr(args, "num_shots", 4)
    variable_shots = getattr(args, "variable_shots", False)
    sampler = MultiModalEpisodeSampler(
        datasets=args.meta_train_datasets,
        num_shots=num_shots,
        args=args,
        preprocess=pretrained_model.train_preprocess,
        variable_shots=variable_shots,
    )

    # Also create a validation sampler
    val_sampler = MultiModalEpisodeSampler(
        datasets=args.meta_val_datasets,
        num_shots=num_shots,
        args=args,
        preprocess=pretrained_model.val_preprocess,
        variable_shots=False,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        hypernetwork.parameters(),
        lr=args.meta_lr,
        weight_decay=0.01,
    )

    # --- Learning rate scheduler (cosine annealing) ---
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.meta_epochs, eta_min=args.meta_lr * 0.01
    )

    # --- Meta-training loop ---
    print("\n" + "=" * 80)
    print("Starting Meta-Training")
    print("=" * 80)

    # --- Episode caches (avoid re-creating expensive objects per episode) ---
    train_cache = EpisodeCache()
    val_cache = EpisodeCache()

    best_val_acc = 0.0
    results_history = {
        "config": {
            "model": args.model,
            "fusion_mode": getattr(args, "fusion_mode", "concat"),
            "image_pooling": getattr(args, "image_pooling", "mean"),
            "text_input_mode": getattr(args, "text_input_mode", "dataset"),
            "num_shots": num_shots,
            "variable_shots": variable_shots,
            "hypernetwork_arch": args.hypernetwork_arch,
            "meta_lr": args.meta_lr,
            "meta_train_datasets": args.meta_train_datasets,
            "meta_val_datasets": args.meta_val_datasets,
        },
        "train_losses": [],
        "val_accuracies": [],
    }

    for epoch in range(args.meta_epochs):
        hypernetwork.train()
        epoch_losses = []
        epoch_accs = []

        print(f"\nEpoch {epoch + 1}/{args.meta_epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        print("-" * 40)

        for episode in tqdm(range(args.episodes_per_epoch), desc="Episodes"):
            dataset_name = random.choice(args.meta_train_datasets)

            try:
                loss, acc = meta_train_episode(
                    hypernetwork, task_vectors, dataset_name,
                    pretrained_model, sampler, args,
                    cache=train_cache,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_accs.append(acc)

            except Exception as e:
                print(f"  Error in episode {episode} ({dataset_name}): {e}")
                continue

        scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        avg_acc = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0.0
        results_history["train_losses"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": avg_acc,
        })
        print(f"  Train loss: {avg_loss:.4f}  |  Train acc: {avg_acc:.1f}%")

        # --- Validation (every 5 epochs) ---
        if (epoch + 1) % 5 == 0:
            hypernetwork.eval()
            val_accs = []

            print("\n  Validation:")
            for val_dataset in args.meta_val_datasets:
                try:
                    with torch.no_grad():
                        _, acc = meta_train_episode(
                            hypernetwork, task_vectors, val_dataset,
                            pretrained_model, val_sampler, args,
                            cache=val_cache,
                        )
                    val_accs.append(acc)
                    print(f"    {val_dataset}: {acc:.2f}%")
                except Exception as e:
                    print(f"    {val_dataset}: Error - {e}")

            avg_val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
            print(f"  Average validation acc: {avg_val_acc:.2f}%")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                save_path = os.path.join(save_dir, "meta_trained.pt")
                hypernetwork.save(save_path)
                print(f"  ** New best model saved (val_acc: {best_val_acc:.2f}%) **")

            results_history["val_accuracies"].append({
                "epoch": epoch + 1,
                "accuracy": avg_val_acc,
                "per_dataset": {
                    ds: acc for ds, acc in zip(args.meta_val_datasets, val_accs)
                },
            })

    # --- Save final results ---
    results_path = os.path.join(save_dir, "meta_results.json")
    with open(results_path, "w") as f:
        json.dump(results_history, f, indent=2)

    # Save final model (regardless of val accuracy)
    final_path = os.path.join(save_dir, "meta_trained_final.pt")
    hypernetwork.save(final_path)

    print("\n" + "=" * 80)
    print("Meta-Training Complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Results saved to: {results_path}")
    print(f"  Best model: {os.path.join(save_dir, 'meta_trained.pt')}")
    print(f"  Final model: {final_path}")
    print("=" * 80)


def main():
    args = parse_arguments()

    if args.meta_train_datasets is None:
        raise ValueError("--meta-train-datasets required for meta-training")
    if args.meta_val_datasets is None:
        raise ValueError("--meta-val-datasets required for meta-training")

    args.save = get_checkpoint_dir(args)

    meta_train(args)


if __name__ == "__main__":
    main()
