#!/usr/bin/env python3
"""
extract_imagenet_stream.py

Single-pass streaming extraction of ImageNet-1K from the ILSVRC2012 train tar.
Reads the outer tar once, and for each nested class tar, extracts its images
directly to ~/data/imagenet/train/<synset_id>/ WITHOUT writing the class tar
itself to disk. This keeps peak extra disk usage minimal.

Skips classes whose directories already exist and contain images (resume-safe).
"""
import argparse
import os
import sys
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tar", default=os.path.expanduser("~/Downloads/ILSVRC2012_img_train.tar"))
    parser.add_argument("--train-dir", default=os.path.expanduser("~/data/imagenet/train"))
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    train_tar_path = Path(args.train_tar)
    train_dir = Path(args.train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    if not train_tar_path.exists():
        print(f"ERROR: train tar not found at {train_tar_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening {train_tar_path} (this may take a moment to index)...")
    total_classes = 0
    skipped = 0
    extracted = 0
    failed = []

    with tarfile.open(train_tar_path, "r|") as outer:
        # Streaming mode (r|) — iterate members without seeking.
        for member in outer:
            if not member.isfile() or not member.name.endswith(".tar"):
                continue
            total_classes += 1
            synset = Path(member.name).stem  # e.g., n01440764
            class_dir = train_dir / synset

            # Skip if already extracted (has JPEGs)
            if class_dir.is_dir() and any(class_dir.glob("*.JPEG")):
                skipped += 1
                continue

            class_dir.mkdir(parents=True, exist_ok=True)

            # Extract the class tar as a file-like object (in-memory stream)
            try:
                class_tar_fileobj = outer.extractfile(member)
                if class_tar_fileobj is None:
                    failed.append(synset)
                    continue
                # Open the inner tar from the file object and extract all images
                with tarfile.open(fileobj=class_tar_fileobj, mode="r|") as inner:
                    for img_member in inner:
                        if img_member.isfile():
                            # Extract directly into class_dir
                            inner.extract(img_member, path=class_dir, set_attrs=False)
                extracted += 1
            except Exception as e:
                print(f"  FAILED {synset}: {e}", file=sys.stderr)
                failed.append(synset)
                continue

            if (extracted + skipped) % args.progress_every == 0:
                n_done = len(list(train_dir.glob("n*")))
                print(f"  Progress: {total_classes} seen, {skipped} skipped, {extracted} newly extracted "
                      f"({n_done} total dirs on disk)")

    print()
    print(f"Done. Seen: {total_classes}, Skipped: {skipped}, Newly extracted: {extracted}, Failed: {len(failed)}")
    if failed:
        print(f"  Failed synsets: {failed[:20]}{'...' if len(failed) > 20 else ''}")
        sys.exit(2)

    final_count = len(list(train_dir.glob("n*")))
    print(f"Final train dir count: {final_count}/1000")
    if final_count < 1000:
        print(f"WARNING: Only {final_count} class directories present")
        sys.exit(3)


if __name__ == "__main__":
    main()
