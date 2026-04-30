"""Build LaTeX tables from aTLAS experiment outputs.

Mirrors the four headline tables / figures in the aTLAS paper
(Zhang et al., NeurIPS 2024):

  * Table 1 (task negation)  — Target ↓, Control ↑ averaged across 8 datasets.
  * Table 2 (task addition)  — Absolute & relative accuracy averaged across
    the same 8 datasets.
  * Few-shot table           — aTLAS accuracy at k ∈ {1, 2, 4, 8, 16} shots
    averaged across 22 datasets (Figure 5a in the paper rendered as a table).
  * Table 3 (test-time)      — Zero-shot vs aTLAS under each self-supervised
    objective available in the run (UFM = pseudo-labelling).

It reads the JSON artefacts emitted by ``learn_task_negation.py``,
``learn_task_addition.py``, ``learn_few_shots.py``, and ``learn_ufm.py``,
and prints LaTeX directly to stdout (or writes it to ``--output``).

Usage:
  python scripts/build_paper_tables.py --table all \
      --models ViT-B-32 ViT-B-16 ViT-L-14 \
      --backends clip openclip

  python scripts/build_paper_tables.py --table negation \
      --checkpoint-root checkpoints_clip --models ViT-B-32 \
      --output tables/negation.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


# The paper's canonical 8-dataset task-arithmetic suite (Table 1, Table 2).
TASK_ARITHMETIC_DATASETS: tuple[str, ...] = (
    "Cars", "DTD", "EuroSAT", "GTSRB",
    "MNIST", "RESISC45", "SUN397", "SVHN",
)

# The paper's 22-dataset suite (Figure 5, Table 3, Table 4).
FEW_SHOT_DATASETS: tuple[str, ...] = (
    *TASK_ARITHMETIC_DATASETS,
    "CIFAR10", "CIFAR100", "ImageNet", "STL10", "Food101",
    "Caltech101", "Caltech256", "FGVCAircraft", "Flowers102",
    "OxfordIIITPet", "CUB200", "PascalVOC", "Country211", "UCF101",
)

FEW_SHOT_K_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Run:
    """One (model, backend) checkpoint/result root combination."""
    model: str
    backend: str
    checkpoint_root: str  # e.g. checkpoints_clip
    logdir: str           # e.g. results_clip

    @property
    def model_ckpt_dir(self) -> str:
        return os.path.join(self.checkpoint_root, self.model)

    @property
    def model_result_dir(self) -> str:
        return os.path.join(self.logdir, self.model)


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_zs_acc(run: Run) -> dict:
    """Top-level zero-shot accuracy map for the run (all datasets)."""
    return _load_json(os.path.join(run.model_ckpt_dir, "zeroshot_accuracies.json")) or {}


def load_ft_acc(run: Run) -> dict:
    return _load_json(os.path.join(run.model_ckpt_dir, "ft_accuracies.json")) or {}


def load_negations(run: Run) -> dict | None:
    return _load_json(os.path.join(run.model_ckpt_dir, "learned_negations.json"))


def load_additions(run: Run) -> dict | None:
    return _load_json(os.path.join(run.model_ckpt_dir, "learned_additions.json"))


def load_few_shots(run: Run) -> dict | None:
    """Keys like ``"1_shot"`` → {dataset: acc, ...}. Written by learn_few_shots.py."""
    return _load_json(os.path.join(run.model_result_dir, "learned_composition.json"))


def load_ufm(run: Run) -> dict | None:
    """UFM / test-time adaptation results. Written by learn_ufm.py under
    the logdir passed by run_full_experiment_suite.sh (typically a
    ``test_time`` subdirectory)."""
    for candidate in (
        os.path.join(run.model_result_dir, "test_time", "learned_composition.json"),
        os.path.join(run.model_result_dir, "ufm", "learned_composition.json"),
        os.path.join(run.model_result_dir, "learned_composition_ufm.json"),
    ):
        data = _load_json(candidate)
        if data is not None:
            return data
    return None


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def _mean(values: Iterable[float]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return sum(vs) / len(vs)


def _fmt(value: float | None, places: int = 2) -> str:
    return "--" if value is None else f"{100 * value:.{places}f}"


def _latex_backend_label(backend: str) -> str:
    return {"clip": r"\textsc{hf}", "openclip": r"\textsc{open}"}.get(backend, backend)


# --------------------------------------------------------------------------- #
# Table 1 — Task negation
# --------------------------------------------------------------------------- #

def _negation_averages(data: dict | None, zs_acc: dict) -> tuple[float | None, float | None, float | None]:
    """Return (pretrained target avg, post-negation target avg, post-negation control avg).

    Pre-trained target avg is taken from ``zeroshot_accuracies.json`` using the
    test-split key (``<dataset>``) so it is comparable to the post-negation
    numbers, which come from the ``test`` / ``test_control`` fields of
    ``learned_negations.json`` (also test split). ``zeroshot_accuracies.json``
    is written by ``eval_single_task.py`` and contains both ``<dataset>`` (test)
    and ``<dataset>Val`` (val) keys; we deliberately use the former here.
    """
    if data is None:
        return None, None, None
    pre, tgt_post, ctr_post = [], [], []
    for ds in TASK_ARITHMETIC_DATASETS:
        entry = data.get(ds)
        if not entry:
            continue
        pre.append(zs_acc.get(ds))
        tgt_post.append(entry.get("test"))
        ctr_post.append(entry.get("test_control"))
    return _mean(pre), _mean(tgt_post), _mean(ctr_post)


def build_negation_table(runs: Sequence[Run]) -> str:
    """Table 1 in the paper: target (↓) and control (↑) averaged across 8 datasets."""
    cols = "l" + "rr" * len(runs)
    header_top = " & ".join(
        [""] + [fr"\multicolumn{{2}}{{c}}{{{run.model} ({_latex_backend_label(run.backend)})}}"
                for run in runs]
    )
    header_sub = " & ".join(
        ["Method"] + [r"Target ($\downarrow$) & Control ($\uparrow$)"] * len(runs)
    )
    cmidrules = " ".join(
        fr"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs))
    )

    pre_row, atlas_row = ["Pre-trained $f(\\mathbf{x};\\boldsymbol{\\theta}_0)$"], [r"aTLAS (ours)"]
    for run in runs:
        zs = load_zs_acc(run)
        pre, tgt, ctr = _negation_averages(load_negations(run), zs)
        # Use the test-split key for the control baseline too, to match the
        # split used by the post-negation control numbers.
        pretrained_ctr = zs.get("ImageNet")
        pre_row.extend([_fmt(pre), _fmt(pretrained_ctr)])
        atlas_row.extend([_fmt(tgt), _fmt(ctr)])

    body = "\n".join([
        " & ".join(pre_row) + r" \\",
        r"\midrule",
        " & ".join(atlas_row) + r" \\",
    ])

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Task negation performance averaged across the eight target "
        r"datasets (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN). "
        r"Control dataset is ImageNet. Numbers are top-1 accuracy (\%).}" "\n"
        r"\label{tab:negation}" "\n"
        fr"\begin{{tabular}}{{{cols}}}" "\n"
        r"\toprule" "\n"
        fr"{header_top} \\" "\n"
        fr"{cmidrules}" "\n"
        fr"{header_sub} \\" "\n"
        r"\midrule" "\n"
        fr"{body}" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# --------------------------------------------------------------------------- #
# Table 2 — Task addition
# --------------------------------------------------------------------------- #

def _addition_averages(data: dict | None) -> tuple[float | None, float | None]:
    """(avg abs accuracy, avg normalised-to-ft accuracy) from the test block."""
    if data is None:
        return None, None
    test = data.get("test", {})
    return test.get("avg_top1"), test.get("avg_normalised_top1")


def build_addition_table(runs: Sequence[Run]) -> str:
    """Table 2 in the paper: absolute and relative accuracy averaged across 8 datasets."""
    cols = "l" + "rr" * len(runs)
    header_top = " & ".join(
        [""] + [fr"\multicolumn{{2}}{{c}}{{{run.model} ({_latex_backend_label(run.backend)})}}"
                for run in runs]
    )
    header_sub = " & ".join(
        ["Method"] + [r"Abs.\ ($\uparrow$) & Rel.\ ($\uparrow$)"] * len(runs)
    )
    cmidrules = " ".join(
        fr"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs))
    )

    pre_row = [r"Pre-trained $f(\mathbf{x};\boldsymbol{\theta}_0)$"]
    atlas_row = [r"aTLAS (ours)"]
    for run in runs:
        zs = load_zs_acc(run)
        # Pre-trained baseline must use the test-split key (``<dataset>``)
        # to match the test-split metrics in ``learned_additions.json``'s
        # ``test`` block. ``zeroshot_accuracies.json`` carries both keys.
        pre = _mean(zs.get(ds) for ds in TASK_ARITHMETIC_DATASETS)
        abs_acc, rel_acc = _addition_averages(load_additions(run))
        pre_row.extend([_fmt(pre), "--"])
        atlas_row.extend([_fmt(abs_acc), _fmt(rel_acc)])

    body = "\n".join([
        " & ".join(pre_row) + r" \\",
        r"\midrule",
        " & ".join(atlas_row) + r" \\",
    ])

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Task addition performance averaged across the eight datasets. "
        r"Abs.\ is absolute top-1 accuracy (\%); Rel.\ is accuracy normalised by "
        r"the fine-tuned single-task model.}" "\n"
        r"\label{tab:addition}" "\n"
        fr"\begin{{tabular}}{{{cols}}}" "\n"
        r"\toprule" "\n"
        fr"{header_top} \\" "\n"
        fr"{cmidrules}" "\n"
        fr"{header_sub} \\" "\n"
        r"\midrule" "\n"
        fr"{body}" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# --------------------------------------------------------------------------- #
# Few-shot table
# --------------------------------------------------------------------------- #

def _fewshot_average(data: dict | None, k: int) -> float | None:
    """Average few-shot test accuracy across the 22 datasets for k-shot.

    ``learn_few_shots.py`` writes the adapted *test* accuracy under
    ``{dataset}Val_trained`` (the target dataset key used during training has
    a ``Val`` suffix; the trained metric is evaluated on the test split).
    It also keeps ``{dataset}Val`` as the validation metric used for model
    selection. For paper tables we want the test number, so we prefer
    ``_trained`` and fall back to the bare ``{dataset}`` (older outputs)
    or ``{dataset}Val`` (oldest outputs) for backward compatibility. Missing
    datasets are skipped rather than counted as zero.
    """
    if data is None:
        return None
    bucket = data.get(f"{k}_shot")
    if not bucket:
        return None
    accs = [
        bucket.get(f"{ds}Val_trained",
                   bucket.get(f"{ds}_trained",
                              bucket.get(ds, bucket.get(f"{ds}Val"))))
        for ds in FEW_SHOT_DATASETS
    ]
    return _mean(accs)


def build_fewshot_table(runs: Sequence[Run]) -> str:
    """Few-shot recognition accuracy averaged across 22 datasets.

    The paper reports this as Figure 5a; we lay it out as a table indexed by
    shot count k for consistency with the rest of the write-up."""
    cols = "l" + "r" * len(FEW_SHOT_K_VALUES)
    head_cols = [fr"$k={k}$" for k in FEW_SHOT_K_VALUES]
    header = " & ".join(["Model / backend"] + head_cols)

    rows = []
    for run in runs:
        data = load_few_shots(run)
        vals = [_fmt(_fewshot_average(data, k)) for k in FEW_SHOT_K_VALUES]
        rows.append(" & ".join([f"{run.model} ({_latex_backend_label(run.backend)})"] + vals) + r" \\")

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Few-shot recognition accuracy (\%) of aTLAS averaged over the "
        r"22 datasets of Zhang et al.\ \cite{zhang2024atlas}, for each shot count "
        r"$k$. Missing entries indicate experiments that have not been run for "
        r"that configuration.}" "\n"
        r"\label{tab:fewshot}" "\n"
        fr"\begin{{tabular}}{{{cols}}}" "\n"
        r"\toprule" "\n"
        fr"{header} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# --------------------------------------------------------------------------- #
# Table 3 — Test-time adaptation (UFM / pseudo-labelling)
# --------------------------------------------------------------------------- #

def _ufm_averages(data: dict | None, zs_acc: dict) -> tuple[float | None, float | None]:
    """Return (zero-shot avg, aTLAS-UFM avg) across the 22 datasets.

    ``learn_ufm.py`` logs both ``{dataset}Val`` (validation metric used during
    coefficient search) and ``{dataset}`` (test split, written by
    ``eval_single_dataset``). The paper reports the test number, so we average
    the bare ``{dataset}`` keys here for consistency. ``zeroshot_accuracies.json``
    likewise stores both — we use ``{dataset}`` for the same reason.
    """
    if data is None:
        return None, None
    zs = _mean(zs_acc.get(ds) for ds in FEW_SHOT_DATASETS)
    adapted = _mean(data.get(ds) for ds in FEW_SHOT_DATASETS)
    return zs, adapted


def build_tta_table(runs: Sequence[Run]) -> str:
    """Table 3 in the paper, restricted to the UFM objective we currently run."""
    cols = "l" + "rr" * len(runs)
    header_top = " & ".join(
        [""] + [fr"\multicolumn{{2}}{{c}}{{{run.model} ({_latex_backend_label(run.backend)})}}"
                for run in runs]
    )
    header_sub = " & ".join(["Method"] + [r"Zero-shot & aTLAS (UFM)"] * len(runs))
    cmidrules = " ".join(
        fr"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs))
    )

    row = [r"22-dataset avg."]
    for run in runs:
        zs, adapted = _ufm_averages(load_ufm(run), load_zs_acc(run))
        row.extend([_fmt(zs), _fmt(adapted)])

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Test-time adaptation accuracy (\%) averaged across the 22 "
        r"datasets, under the UFM (pseudo-labelling) self-supervised objective.}" "\n"
        r"\label{tab:tta}" "\n"
        fr"\begin{{tabular}}{{{cols}}}" "\n"
        r"\toprule" "\n"
        fr"{header_top} \\" "\n"
        fr"{cmidrules}" "\n"
        fr"{header_sub} \\" "\n"
        r"\midrule" "\n"
        + " & ".join(row) + r" \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# --------------------------------------------------------------------------- #
# Per-dataset detail table (Appendix-style)
# --------------------------------------------------------------------------- #

def build_negation_detail_table(runs: Sequence[Run]) -> str:
    """One row per target dataset; columns = post-negation target/control for each run."""
    cols = "l" + "rr" * len(runs)
    header_top = " & ".join(
        [""] + [fr"\multicolumn{{2}}{{c}}{{{run.model} ({_latex_backend_label(run.backend)})}}"
                for run in runs]
    )
    header_sub = " & ".join(
        ["Target dataset"] + [r"Target ($\downarrow$) & Control ($\uparrow$)"] * len(runs)
    )
    cmidrules = " ".join(
        fr"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(runs))
    )
    rows: list[str] = []
    per_run = [load_negations(run) or {} for run in runs]
    for ds in TASK_ARITHMETIC_DATASETS:
        cells = [ds]
        for data in per_run:
            entry = data.get(ds) or {}
            cells.extend([_fmt(entry.get("test")), _fmt(entry.get("test_control"))])
        rows.append(" & ".join(cells) + r" \\")

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Per-dataset task negation accuracy (\%). Lower target accuracy "
        r"is better; control accuracy should stay close to the pre-trained model's.}" "\n"
        r"\label{tab:negation-detail}" "\n"
        fr"\begin{{tabular}}{{{cols}}}" "\n"
        r"\toprule" "\n"
        fr"{header_top} \\" "\n"
        fr"{cmidrules}" "\n"
        fr"{header_sub} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

TABLE_BUILDERS = {
    "negation": build_negation_table,
    "negation-detail": build_negation_detail_table,
    "addition": build_addition_table,
    "fewshot": build_fewshot_table,
    "tta": build_tta_table,
}


def _default_logdir(backend: str) -> str:
    return {"clip": "results_clip", "openclip": "results_openclip"}.get(
        backend, f"results_{backend}"
    )


def _default_checkpoint_root(backend: str) -> str:
    return {"clip": "checkpoints_clip", "openclip": "checkpoints_openclip"}.get(
        backend, f"checkpoints_{backend}"
    )


def parse_runs(args: argparse.Namespace) -> list[Run]:
    runs: list[Run] = []
    for backend in args.backends:
        ckpt_root = args.checkpoint_root or _default_checkpoint_root(backend)
        logdir = args.logdir or _default_logdir(backend)
        for model in args.models:
            runs.append(Run(model=model, backend=backend,
                            checkpoint_root=ckpt_root, logdir=logdir))
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--table", required=True,
                   choices=list(TABLE_BUILDERS) + ["all"])
    p.add_argument("--models", nargs="+", default=["ViT-B-32"])
    p.add_argument("--backends", nargs="+", default=["clip", "openclip"])
    p.add_argument("--checkpoint-root",
                   help="Override checkpoint root (default: checkpoints_<backend>). "
                        "Only single-root runs supported through this flag.")
    p.add_argument("--logdir",
                   help="Override results log dir (default: results_<backend>).")
    p.add_argument("--output", help="Write LaTeX to this file instead of stdout.")
    args = p.parse_args(argv)

    runs = parse_runs(args)

    tables_to_build = list(TABLE_BUILDERS) if args.table == "all" else [args.table]
    chunks = [TABLE_BUILDERS[name](runs) for name in tables_to_build]
    output = "\n\n".join(chunks) + "\n"

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote {len(chunks)} table(s) to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
