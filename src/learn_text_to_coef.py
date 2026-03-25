"""Meta-train hypernetwork to predict aTLAS coefficients from text descriptions.

This script implements episode-based meta-training where the hypernetwork learns
to predict good coefficients for new tasks by training on a diverse set of tasks.

Usage:
    python src/learn_text_to_coef.py \
        --model ViT-B-32 \
        --meta-train-datasets CIFAR10,EuroSAT,MNIST,DTD,GTSRB \
        --meta-val-datasets Cars,SUN397 \
        --text-source manual \
        --hypernetwork-arch small \
        --meta-epochs 100 \
        --episodes-per-epoch 20
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from args import parse_arguments, get_checkpoint_dir
from modeling import ImageEncoder
from task_vectors import NonLinearTaskVector
from hypernetworks.text_to_coef import create_hypernetwork_from_args
from composition import TextConditionedWeightedImageEncoder
from datasets.registry import get_dataset
from heads import get_classification_head
from utils import load_text_descriptions
from learn_few_shots import load_task_vectors


def meta_train_episode(
    hypernetwork,
    task_vectors,
    dataset_name,
    pretrained_model,
    args
):
    """Single meta-training episode on one task.

    Args:
        hypernetwork: TextToCoefHypernetwork instance
        task_vectors: Dictionary of task vectors
        dataset_name: Name of the target dataset
        pretrained_model: Pretrained image encoder
        args: Training arguments

    Returns:
        loss: Differentiable episode loss (tensor)
        accuracy: Validation accuracy (float)
    """
    # Load text descriptions for this dataset (use base name, not split name)
    text_descriptions = load_text_descriptions(dataset_name, args)

    # Predict coefficients from text (keep gradients flowing!)
    predicted_coef = hypernetwork.predict_for_dataset(
        text_descriptions,
        aggregate=args.text_aggregate
    )

    # Create weighted encoder with predicted coefficients
    # Use all task vectors except the target
    relevant_tvs = [tv for name, tv in task_vectors.items()
                   if name != dataset_name]

    weighted_encoder = TextConditionedWeightedImageEncoder(
        model=pretrained_model,
        task_vectors=relevant_tvs,
        hypernetwork=None,  # We pass coefficients directly to forward
        blockwise=args.blockwise_coef
    )

    # Move encoder to GPU (this moves dparams too via _apply)
    weighted_encoder = weighted_encoder.cuda()

    # Load classification head
    classification_head = get_classification_head(args, dataset_name + "Val")
    classification_head = classification_head.cuda()

    # Load validation dataset
    dataset = get_dataset(
        dataset_name + "Val",
        weighted_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.meta_batch_size
    )

    # Compute loss on a batch of validation data
    # Get one batch for efficiency in meta-training
    dataloader = torch.utils.data.DataLoader(
        dataset.test_dataset if hasattr(dataset, 'test_dataset') else dataset,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=2
    )

    batch = next(iter(dataloader))
    images = batch[0].cuda() if isinstance(batch, (list, tuple)) else batch['images'].cuda()
    labels = batch[1].cuda() if isinstance(batch, (list, tuple)) else batch['labels'].cuda()

    # Forward pass with predicted coefficients (gradient flows through predicted_coef)
    coef_for_forward = predicted_coef.squeeze(0)
    features = weighted_encoder(images, coef=coef_for_forward)
    logits = classification_head(features)

    # Compute cross-entropy loss (differentiable!)
    loss = nn.functional.cross_entropy(logits, labels)

    # Compute accuracy for logging (detached)
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item() * 100

    return loss, accuracy


def meta_train(args):
    """Main meta-training loop.

    Args:
        args: Parsed arguments
    """
    # Setup save directory
    save_dir = os.path.join(
        args.save,
        "hypernetworks",
        "text_to_coef"
    )
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("Meta-Training Text-to-Coefficient Hypernetwork")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Meta-train datasets: {args.meta_train_datasets}")
    print(f"Meta-val datasets: {args.meta_val_datasets}")
    print(f"Hypernetwork arch: {args.hypernetwork_arch}")
    print(f"Meta-learning rate: {args.meta_lr}")
    print(f"Meta-epochs: {args.meta_epochs}")
    print(f"Episodes per epoch: {args.episodes_per_epoch}")
    print("=" * 80)

    # Load pretrained model
    print("\nLoading pretrained model...")
    pretrained_model = ImageEncoder(args)
    pretrained_model = pretrained_model.cuda()

    # Load all task vectors
    source = getattr(args, 'task_vector_source', 'real')
    backend = getattr(args, 't2i_backend', 'stable_diffusion')
    print(f"\nLoading task vectors (source={source})...")
    all_task_vectors = load_task_vectors(args, source=source, backend=backend)

    # Filter to only meta-train + meta-val datasets
    pool = args.meta_train_datasets + args.meta_val_datasets
    task_vectors = {k: v for k, v in all_task_vectors.items() if k in pool}
    for dataset in pool:
        if dataset in task_vectors:
            print(f"  Loaded: {dataset}")
        else:
            print(f"  Warning: Missing checkpoints for {dataset}, skipping")

    # Determine number of blocks
    num_blocks = len(pretrained_model.model.state_dict())
    print(f"\nNumber of parameter blocks: {num_blocks}")

    # Create hypernetwork
    print(f"\nCreating hypernetwork (arch: {args.hypernetwork_arch})...")
    hypernetwork = create_hypernetwork_from_args(args, num_blocks)
    hypernetwork = hypernetwork.cuda()

    # Count parameters
    total_params = sum(p.numel() for p in hypernetwork.parameters())
    trainable_params = sum(p.numel() for p in hypernetwork.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        hypernetwork.parameters(),
        lr=args.meta_lr,
        weight_decay=0.01
    )

    # Meta-training loop
    print("\n" + "=" * 80)
    print("Starting Meta-Training")
    print("=" * 80)

    best_val_acc = 0.0
    results_history = {
        'train_losses': [],
        'val_accuracies': [],
    }

    for epoch in range(args.meta_epochs):
        # Training phase
        hypernetwork.train()
        epoch_losses = []

        print(f"\nEpoch {epoch+1}/{args.meta_epochs}")
        print("-" * 40)

        # Sample episodes
        for episode in tqdm(range(args.episodes_per_epoch), desc="Episodes"):
            # Randomly sample a task from meta-train set
            import random
            dataset_name = random.choice(args.meta_train_datasets)

            # Run episode
            try:
                loss, acc = meta_train_episode(
                    hypernetwork,
                    task_vectors,
                    dataset_name,
                    pretrained_model,
                    args
                )

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hypernetwork.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            except Exception as e:
                print(f"  Error in episode {episode} ({dataset_name}): {e}")
                continue

        # Validation phase
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            hypernetwork.eval()
            val_accs = []

            print("\nValidation:")
            for val_dataset in args.meta_val_datasets:
                try:
                    with torch.no_grad():
                        _, acc = meta_train_episode(
                            hypernetwork,
                            task_vectors,
                            val_dataset,
                            pretrained_model,
                            args
                        )
                    val_accs.append(acc)
                    print(f"  {val_dataset}: {acc:.2f}%")
                except Exception as e:
                    print(f"  {val_dataset}: Error - {e}")

            avg_val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
            print(f"  Average validation accuracy: {avg_val_acc:.2f}%")

            # Save best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                save_path = os.path.join(save_dir, "meta_trained.pt")
                hypernetwork.save(save_path)
                print(f"  Saved best model (val_acc: {best_val_acc:.2f}%)")

            results_history['val_accuracies'].append({
                'epoch': epoch + 1,
                'accuracy': avg_val_acc
            })

        # Log epoch results
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        results_history['train_losses'].append({
            'epoch': epoch + 1,
            'loss': avg_loss
        })
        print(f"Average loss: {avg_loss:.4f}")

    # Save final results
    results_path = os.path.join(save_dir, "meta_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_history, f, indent=2)

    print("\n" + "=" * 80)
    print("Meta-Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Saved to: {save_dir}")
    print("=" * 80)


def main():
    args = parse_arguments()

    # Validate arguments
    if args.meta_train_datasets is None:
        raise ValueError("--meta-train-datasets required for meta-training")
    if args.meta_val_datasets is None:
        raise ValueError("--meta-val-datasets required for meta-training")

    # Set save directory
    args.save = get_checkpoint_dir(args)

    # Run meta-training
    meta_train(args)


if __name__ == "__main__":
    main()
