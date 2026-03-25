"""Learn the coefficients on task vectors
under the few-shot setting for a dataset
and find the optimal combination.

Paul Albert <paul.albert@adelaide.edu.au>
Fred Zhang <frederic.zhang@adelaide.edu.au>

Australian Institute for Machine Learning
"""

import os
import copy
import time
import json
import traceback
import torch
import torchvision

from torch.cuda.amp import GradScaler
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import NonLinearTaskVector
from src.composition import WeightedImageEncoder

from src.utils import cosine_lr, get_n_shots, TIPWrapper, LPPWrapper, IndexWrapper, _RepeatSampler
from src.args import parse_arguments, get_checkpoint_dir
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp

DATASET_POOL = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "CIFAR10", "CIFAR100",
    "STL10", "Food101", "Caltech101", "Caltech256", "FGVCAircraft", "Flowers102",
    "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101", "SUN397", "ImageNet"
]

# DATASET_POOL = [
#     "Cars", "DTD"
# ]


def load_task_vectors(args, source="real", backend="stable_diffusion"):
    """Load task vectors from checkpoints.

    Args:
        args: Parsed arguments (must have args.save set to checkpoint root).
        source: "real" for standard fine-tuned checkpoints, "synthetic" for
                checkpoints fine-tuned on T2I-generated images.
        backend: T2I backend name (only used when source="synthetic").

    Returns:
        Dictionary mapping dataset name to NonLinearTaskVector.
    """
    task_vectors = {}
    for dataset in DATASET_POOL:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        if source == "real" or source == "mixed":
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        else:
            finetuned_checkpoint = f"{args.save}/{dataset}Val/synthetic_{backend}_finetuned.pt"
        if os.path.exists(pretrained_checkpoint) and os.path.exists(finetuned_checkpoint):
            task_vectors[dataset] = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        else:
            print(f"Warning: Missing checkpoint for {dataset} (source={source}), skipping")
    return task_vectors

def _load_dataset_for_training(args, preprocess_fn, target_dataset, orig_dataset):
    """Load the training dataset based on the task vector source."""
    source = getattr(args, 'task_vector_source', 'real')
    if source == 'synthetic':
        from src.datasets.synthetic import SyntheticDatasetWrapper
        dataset = SyntheticDatasetWrapper(
            preprocess=preprocess_fn,
            location=args.synthetic_data_location,
            dataset_name=orig_dataset,
            t2i_backend=args.t2i_backend,
            batch_size=args.batch_size,
            num_workers=8,
        )
    elif source == 'mixed':
        from src.datasets.synthetic import SyntheticDatasetWrapper, MixedDatasetWrapper
        real_dataset = get_dataset(
            target_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=8,
        )
        synthetic_dataset = SyntheticDatasetWrapper(
            preprocess=preprocess_fn,
            location=args.synthetic_data_location,
            dataset_name=orig_dataset,
            t2i_backend=args.t2i_backend,
            batch_size=args.batch_size,
            num_workers=8,
        )
        dataset = MixedDatasetWrapper(real_dataset, synthetic_dataset, batch_size=args.batch_size, seed=args.seed)
    else:
        dataset = get_dataset(
            target_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=8,
        )
    return dataset

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def _failures_path(args):
    """Path to the failures log file, next to the results log."""
    return os.path.join(args.logdir, "failed_evaluations.json")


def _load_failures(args):
    """Load the set of failed (dataset, shot_key) pairs."""
    path = _failures_path(args)
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def _save_failures(args, failures):
    """Save the failures dict. Remove the file entirely if empty."""
    path = _failures_path(args)
    if not failures:
        if os.path.isfile(path):
            os.remove(path)
        return
    with open(path, 'w') as f:
        json.dump(failures, f, indent=4)


def _record_failure(args, dataset, shot_key, error_msg):
    """Record a single failure."""
    failures = _load_failures(args)
    failures[f"{dataset}/{shot_key}"] = {
        "dataset": dataset,
        "shot_key": shot_key,
        "error": error_msg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_failures(args, failures)


def _clear_failure(args, dataset, shot_key):
    """Remove a failure entry after successful retry."""
    failures = _load_failures(args)
    key = f"{dataset}/{shot_key}"
    if key in failures:
        del failures[key]
        _save_failures(args, failures)


def main(rank, args):
    # Load the individual task vectors.
    task_vectors = load_task_vectors(args, source="real")

    args.rank = rank

    # Normalize subsample to a list of shot values.
    subsample_values = args.subsample if isinstance(args.subsample, list) else [args.subsample]
 
    # Cache a base ImageEncoder loaded from OpenCLIP once.
    # Each training run will deepcopy this instead of reloading from disk.
    base_image_encoder = ImageEncoder(args)

    # Build the training preprocess transform (depends only on model, not dataset).
    preprocess_fn = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(
            size=224, scale=(0.5, 1),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
    ] + base_image_encoder.train_preprocess.transforms[-3:])

    # If --retry-failed, load the failures log and restrict to those pairs.
    retry_only = getattr(args, 'retry_failed', False)
    if retry_only:
        pending_failures = _load_failures(args)
        if not pending_failures:
            print("No failed evaluations to retry. Exiting.")
            return
        retry_set = {(v["dataset"], v["shot_key"]) for v in pending_failures.values()}
        print(f"Retrying {len(retry_set)} previously failed evaluation(s):")
        for ds, sk in sorted(retry_set):
            print(f"  - {ds} / {sk}")
    else:
        retry_set = None

    n_succeeded, n_failed = 0, 0

    for dataset, epochs in args.target_datasets.items():
        args.target_dataset = dataset + "Val"
        args.epochs = epochs
        target_dataset = args.target_dataset
        orig_dataset = dataset

        # Check early if any subsample value for this dataset is needed.
        if retry_set is not None:
            needed = any((dataset, f"{s}_shot") in retry_set for s in subsample_values)
            if not needed:
                continue

        # --- Per-dataset caching: load these expensive resources once ---
        try:
            # Cache the classification head for this dataset.
            cached_classification_head = get_classification_head(args, target_dataset)

            # Cache the training dataset for this target.
            cached_dataset = _load_dataset_for_training(
                args, preprocess_fn, target_dataset, orig_dataset,
            )
        except Exception as e:
            # If per-dataset setup fails, record failure for all shots and continue.
            error_msg = f"Dataset setup failed: {e}\n{traceback.format_exc()}"
            print(f"\n{'!'*80}")
            print(f"FAILED setup for {dataset}: {e}")
            print(f"{'!'*80}\n")
            for subsample in subsample_values:
                shot_key = f"{subsample}_shot"
                if retry_set is None or (dataset, shot_key) in retry_set:
                    _record_failure(args, dataset, shot_key, error_msg)
                    n_failed += 1
            continue

        # Load zero-shot accuracy once per dataset (if file exists).
        zs_acc_path = os.path.join(f"{args.save}/{dataset}Val/", "zeroshot_accuracies.json")
        if os.path.isfile(zs_acc_path):
            with open(zs_acc_path, 'r') as f:
                args.zs_acc = json.load(f)
        else:
            if not hasattr(args, 'zs_acc'):
                args.zs_acc = {}

        cached_resources = {
            'image_encoder': base_image_encoder,
            'classification_head': cached_classification_head,
            'dataset': cached_dataset,
        }

        # --- Inner loop: iterate over shot values for this dataset ---
        for subsample in subsample_values:
            shot_key = f"{subsample}_shot"

            # Skip if not in retry set.
            if retry_set is not None and (dataset, shot_key) not in retry_set:
                continue

            args.subsample = subsample
            data_amount = f"{subsample} shots"
            args.head_path = os.path.join(args.logdir, f"learned_composition_{shot_key}.pt")

            # Load accumulated results for this shot setting.
            if os.path.exists(args.log_path):
                with open(args.log_path, 'r') as f:
                    all_results = json.load(f)
            else:
                all_results = {}

            comp_acc = all_results.get(shot_key, {})

            # Carry over zero-shot accuracy if already known.
            if os.path.isfile(zs_acc_path):
                comp_acc[f"{dataset}Val_zeroshot"] = args.zs_acc[f"{dataset}Val"]

            print("=" * 100)
            print(f"Learning task vector coefficients on {dataset} with {args.model} - {data_amount}")
            print("=" * 100)

            try:
                comp_acc = train(task_vectors, args, comp_acc, cached_resources=cached_resources)
                all_results[shot_key] = comp_acc
                n_succeeded += 1
                # Clear from failures log if this was a retry.
                _clear_failure(args, dataset, shot_key)
            except Exception as e:
                error_msg = f"{e}\n{traceback.format_exc()}"
                print(f"\n{'!'*80}")
                print(f"FAILED {dataset} / {shot_key}: {e}")
                print(f"Traceback:\n{traceback.format_exc()}")
                print(f"{'!'*80}\n")
                _record_failure(args, dataset, shot_key, error_msg)
                n_failed += 1
                continue

    # Print summary.
    print("\n" + "=" * 100)
    print(f"Run complete: {n_succeeded} succeeded, {n_failed} failed")
    if n_failed > 0:
        failures = _load_failures(args)
        print(f"Failed evaluations logged to: {_failures_path(args)}")
        print("Re-run with --retry-failed to retry only the failed evaluations.")
        for key, info in failures.items():
            print(f"  - {key}: {info['error'].splitlines()[0]}")
    print("=" * 100)
 
def train(task_vectors, args, comp_acc={}, cached_resources=None):
    """Train task vector coefficients for a single dataset and shot setting.
 
    Args:
        task_vectors: Dict mapping dataset names to NonLinearTaskVector objects.
        args: Parsed arguments.
        comp_acc: Dict accumulating accuracy results for the current shot setting.
        cached_resources: Optional dict with pre-loaded resources to avoid redundant I/O:
            - 'image_encoder': Base ImageEncoder (will be deepcopied)
            - 'classification_head': Pre-loaded classification head
            - 'dataset': Pre-loaded dataset object
    """

    setup_ddp(args.rank, args.world_size, port=args.port)
    target_dataset = args.target_dataset

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    orig_dataset = target_dataset.replace("Val", "")
    # Remove the task vector for the target task
    task_vectors = [v for k, v in task_vectors.items() if orig_dataset != k]

    # Use cached ImageEncoder (deepcopy) or load fresh from disk.
    if cached_resources is not None:
        image_encoder = copy.deepcopy(cached_resources['image_encoder'])
    else:
        image_encoder = ImageEncoder(args)

    image_encoder = WeightedImageEncoder(
        image_encoder, task_vectors, blockwise=args.blockwise_coef, partition=args.partition,
    )

    # Use cached classification head or load fresh.
    if cached_resources is not None:
        classification_head = cached_resources['classification_head']
    else:
        classification_head = get_classification_head(args, target_dataset)
    model = ImageClassifier(image_encoder, classification_head)

    model.freeze_head()
    model = model.cuda()

    # Use cached dataset or load fresh.
    if cached_resources is not None:
        dataset = cached_resources['dataset']
    else:
        # TIP's more aggressive random crop with horizontal flip
        preprocess_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                size=224, scale=(0.5, 1),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ), torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ] + model.train_preprocess.transforms[-3:])
 
        dataset = _load_dataset_for_training(args, preprocess_fn, target_dataset, orig_dataset)

    # Validate subsample: this script currently only supports an integer number of shots.
    if not isinstance(args.subsample, int):
        raise TypeError(
            f"--subsample must be an integer number of shots for learn_few_shots; "
            f"got value {args.subsample!r} of type {type(args.subsample).__name__}. "
            "Percentage subsampling with float values is not supported in this script."
        )
    n_shots = args.subsample

    if os.path.isfile(f"{args.save}/{target_dataset}/{n_shots}_shots_{args.seed}.pt") and args.seed == 1:
        to_keep = torch.load(f"{args.save}/{target_dataset}/{n_shots}_shots_{args.seed}.pt", weights_only=False)
    else:
        to_keep = get_n_shots(dataset.train_dataset, n_shots, classification_head.out_features, args)
        torch.save(to_keep, f"{args.save}/{target_dataset}/{n_shots}_shots_{args.seed}.pt")

    r = len(to_keep) / args.batch_size
    if r < 10:
        over_sampling = 10/r
        over_sampling = int(over_sampling) + 1
        print(f"Oversampling {over_sampling} times")
        to_keep = torch.cat([to_keep] * over_sampling)
        
    index_dataset = IndexWrapper(dataset.train_dataset)
    sampler = torch.utils.data.SubsetRandomSampler(to_keep)        
    data_loader = torch.utils.data.DataLoader(index_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    
    # Distribute the data and model across the GPUs.
    ddp_loader = distribute_loader(data_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )

    num_batches = len(ddp_loader)
    # Printing loss between four and ten times an epoch
    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # Do not use warm up
    scheduler = cosine_lr(
        optimizer, args.lr, 0,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    scaler = GradScaler()
    if is_main_process():
        if f"{target_dataset}_zeroshot" not in comp_acc.keys():
            comp_acc[f"{target_dataset}_zeroshot"] = eval_single_dataset(image_encoder, target_dataset.replace('Val',''), args)["top1"]
            with open(os.path.join(f"{args.save}/{target_dataset}/", "zeroshot_accuracies.json"), 'w') as f:
                json.dump({f"{target_dataset}": comp_acc[f"{target_dataset}_zeroshot"]}, f, indent=4)
            args.zs_acc[f"{target_dataset}"] = comp_acc[f"{target_dataset}_zeroshot"]
            
        print(f"=> Zero-shot accuracy on {target_dataset}:\t{100*args.zs_acc[target_dataset]:.2f}%.")
        
    best_coef = ddp_model.module.image_encoder.coef.data.clone()
    best_acc = args.zs_acc[target_dataset]
    best_trained_acc = 0.0
    best_trained_coef = ddp_model.module.image_encoder.coef.data.clone()
    epoch_accs = []
    for epoch in range(args.epochs):
        for i, batch in enumerate(ddp_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            data_time = time.time() - start_time

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs)
                labels = batch["labels"].cuda()
                loss = loss_fn(logits, labels)
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * (i + 1) / len(ddp_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{num_batches}]\t"           # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                    flush=True,
                )
        
        # Evaluate after each epoch
        if is_main_process():
            image_encoder = ddp_model.module.image_encoder
            coef = ddp_model.module.image_encoder.coef
            acc = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
            epoch_accs.append(acc)
            if acc > best_trained_acc:
                best_trained_acc = acc
                best_trained_coef = coef.data.clone()
            if acc > best_acc:
                best_acc = acc
                best_coef = coef.data.clone()

    if is_main_process():
        comp_acc[target_dataset] = best_acc
        comp_acc[f"{target_dataset}_trained"] = best_trained_acc
        comp_acc[f"{target_dataset}_epoch_accs"] = epoch_accs
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.module.image_encoder
        # Evaluate best trained coefficients on full test set
        image_encoder.coef = torch.nn.Parameter(best_trained_coef)
        comp_acc[f"{target_dataset}_trained"] = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
        # Evaluate overall best (may be zero-shot) on full test set
        image_encoder.coef = torch.nn.Parameter(best_coef)
        comp_acc[target_dataset] = eval_single_dataset(image_encoder, target_dataset, args)["top1"]
        # Save nested results keyed by shot number
        if os.path.exists(args.log_path):
            with open(args.log_path, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        shot_key = f"{args.subsample}_shot"
        all_results[shot_key] = comp_acc
        with open(args.log_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path, weights_only=False)
        else:
            heads = {}

        heads[target_dataset] = best_coef
        torch.save(heads, args.head_path)

    if args.adapter is not None:
        comp_acc = train_adapter(ddp_model, ddp_loader, args, comp_acc, which=args.adapter)
        
    cleanup_ddp()
    return comp_acc


def train_adapter(ddp_model, ddp_loader, args, comp_acc, which='lpp'):
    #Extracting features:
    all_features, all_labels, all_indexes, all_logits = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(ddp_loader):
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()

            logits, features = ddp_model(inputs, return_features=True)
            labels = batch["labels"]
            
            all_features.append(features.detach().cpu())
            all_labels.append(labels)
            all_indexes.append(batch["index"])
            all_logits.append(logits.detach().cpu())
            
    logits_cache = torch.cat(all_logits)
    features_cache = torch.cat(all_features)
    labels = torch.cat(all_labels)
    indexes = torch.cat(all_indexes)
    indexes_to_i = {indexes[i].item():i for i in range(len(indexes))}
    
    model = ddp_model.module
    if which == 'lpp':
        model = LPPWrapper(model, features_cache, labels, args.subsample)
        epochs = 300
        lr = model.lr_temp
    elif which == 'tip':
        model = TIPWrapper(model, features_cache, labels)
        lr = 1e-3
        epochs = 10
    else:
        raise NotImplementedError(f"Adapter {which} unknown")

    model = model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.rank],
        find_unused_parameters=False,
        output_device=args.rank,
    )
    
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    num_batches = len(ddp_loader)
    scheduler = cosine_lr(
        optimizer, lr, 0,
        epochs * num_batches // args.num_grad_accumulation,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss()
    print_every = 100
    ddp_loader._DataLoader__initialized = False
    ddp_loader.batch_sampler = _RepeatSampler(ddp_loader.batch_sampler, epochs)
    ddp_loader._DataLoader__initialized = True

    for i, batch in enumerate(ddp_loader):
        start_time = time.time()
        epoch = i // num_batches
        step = (
            i // args.num_grad_accumulation
            + epoch * num_batches // args.num_grad_accumulation
        )
        
        batch = maybe_dictionarize(batch)
        inputs = batch["images"].cuda()
        data_time = time.time() - start_time

        ids = [indexes_to_i[i.item()] for i in batch['index']]
        l_cache, f_cache = logits_cache[ids].to(inputs), features_cache[ids].to(inputs)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = ddp_model(inputs, l_cache, f_cache)
            labels = batch["labels"].to(logits.device)
            loss = loss_fn(logits, labels)
            loss = loss / args.num_grad_accumulation

        if (i + 1) % args.num_grad_accumulation == 0:
            scheduler(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        batch_time = time.time() - start_time

        if (
            step % print_every == 0
            and ((i + 1) % args.num_grad_accumulation == 0)
            and is_main_process()
        ):
            percent_complete = 100 * (i + 1) / len(ddp_loader)
            print(
                f"Train Epoch: {epoch} [{percent_complete:.0f}% {i + 1}/{len(ddp_loader)}]\t"           # noqa: E501
                f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",   # noqa: E501
                flush=True,
            )

    if is_main_process():
        #comp_acc[target_dataset+f"_{which}"] = best_acc
        target_dataset = args.target_dataset
        target_dataset = target_dataset.replace("Val", "")
        image_encoder = ddp_model.module.model.image_encoder        
        comp_acc[target_dataset+f"_{which}"] = eval_single_dataset(image_encoder, target_dataset, args, model=ddp_model)["top1"]
        # Save nested results keyed by shot number
        if os.path.exists(args.log_path):
            with open(args.log_path, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        shot_key = f"{args.subsample}_shot"
        all_results[shot_key] = comp_acc
        with open(args.log_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        if os.path.isfile(args.head_path):
            heads = torch.load(args.head_path, weights_only=False)
        else:
            heads = {}

        adapter_coefs = {k:v for k,v in ddp_model.module.state_dict().items() if v.requires_grad}
        heads[target_dataset] = adapter_coefs
        torch.save(heads, args.head_path)
        
    return comp_acc

if __name__ == "__main__":

    target_datasets = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 13,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SVHN": 4,
        "CIFAR10": 5,
        "CIFAR100": 6,
        "STL10": 4,
        "Food101": 15,
        "Caltech256": 8,
        "FGVCAircraft": 60,
        "Flowers102": 40,
        "OxfordIIITPet": 5,
        "CUB200": 20,
        "PascalVOC": 10,
        "Country211": 15,
        "UCF101": 20,
        "Caltech101":10,
        "SUN397": 14,
        "ImageNet": 10
    }

    args = parse_arguments()
    args.target_datasets = target_datasets
    # HACK: Some command line arguments are overwritten by defaults here.
    args.lr = 1e-1
    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10

    args.logdir += f"{args.model}"
    args.target_datasets = {k:10 for k,v in args.target_datasets.items()} #10 epochs for few-shots using ViTs.

    args.save = get_checkpoint_dir(args)
    if args.seed is not None:
        args.logdir += f"/{args.seed}"

    # log_path is shared across all shot settings (results are nested by shot_key).
    args.log_path = os.path.join(args.logdir, "learned_composition.json")
    # head_path is set per-shot inside main() since it depends on the subsample value.
    # For backward compatibility with single subsample, set a default here.
    subsample_values = args.subsample if isinstance(args.subsample, list) else [args.subsample]
    if len(subsample_values) == 1:
        shot_key = f"{subsample_values[0]}_shot"
        args.head_path = os.path.join(args.logdir, f"learned_composition_{shot_key}.pt")

    os.makedirs(args.logdir, exist_ok=True)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
