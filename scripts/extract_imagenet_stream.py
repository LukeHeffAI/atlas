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
import posixpath
import sys
import tarfile
from pathlib import Path


def _is_safe_member_name(name: str) -> bool:
    """Reject tar member names that would escape the extraction directory.

    Guards against absolute paths, parent-directory traversal (``..``), and
    drive letters / UNC-style prefixes — the standard ``tarfile`` extractor
    happily writes outside the destination otherwise (CVE-2007-4559).
    """
    if not name or name.startswith(("/", "\\")):
        return False
    # Normalise via posixpath to canonicalise ``./``, ``//``, and back-refs.
    normalised = posixpath.normpath(name)
    if normalised.startswith(("/", "..")) or normalised == "..":
        return False
    if any(part == ".." for part in normalised.split("/")):
        return False
    # Drive letters / UNC prefixes (rare in tar but cheap to reject)
    if len(normalised) >= 2 and normalised[1] == ":":
        return False
    return True


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
                # Open the inner tar from the file object and extract all images.
                # Validate each member name to avoid path-traversal extraction
                # outside ``class_dir`` if the archive is malformed/untrusted.
                with tarfile.open(fileobj=class_tar_fileobj, mode="r|") as inner:
                    for img_member in inner:
                        if not img_member.isfile():
                            continue
                        if not _is_safe_member_name(img_member.name):
                            print(f"  WARN {synset}: skipping unsafe member "
                                  f"{img_member.name!r}", file=sys.stderr)
                            continue
                        # Extract directly into class_dir
                        inner.extract(img_member, path=class_dir, set_attrs=False)
                extracted += 1
            except Exception as e:
                print(f"  FAILED {synset}: {e}", file=sys.stderr)
                failed.append(synset)
                continue

            if (extracted + skipped) % args.progress_every == 0:
                # Avoid re-walking the directory tree on every tick — on
                # network filesystems that scan can dominate runtime. The
                # running ``skipped + extracted`` counter is equivalent for
                # progress purposes (we already enforce no double-counting
                # via the ``any(class_dir.glob('*.JPEG'))`` skip check above).
                n_done = skipped + extracted
                print(f"  Progress: {total_classes} seen, {skipped} skipped, {extracted} newly extracted "
                      f"({n_done} total dirs processed)")

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
