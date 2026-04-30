# NeurIPS 2026 Submission Plan

> **Today: 2026-04-30.** **Abstract deadline: 2026-05-04 AOE.** **Full paper deadline: 2026-05-06 AOE.**
> Time remaining: **~6 days, including the deadline days themselves.**

## Honest opening note

This is a brutally tight window for an experimental campaign that has not yet
been run. The core aTLAS infrastructure is solid and the text/multi-modal
hypernetwork code is implemented, but no meta-training has been executed and
no headline results exist. To make the May 6 deadline you have two realistic
options:

1. **Sprint hard** at a *minimum-viable* set of experiments (one backbone, one
   primary contribution, one or two ablations) and accept a thinner paper.
2. **Defer** to a later venue (NeurIPS 2026 workshop track, ICLR 2027,
   CVPR 2027) and use the next 3-4 weeks to do this properly.

The rest of this document plans for option 1. Decision points where you can
abort to option 2 are flagged inline as **STOP-CHECK**. Pick the one
contribution you most believe in (recommendation: **multi-modal hypernetwork**,
since it's the more novel claim) and ruthlessly cut anything that doesn't
serve that headline.

Each task has a checkbox. Tick them off as you go.

---

## Pre-flight (today, before sprint starts)

- [ ] **Decide the headline claim in one sentence and write it on a sticky note.**
      Recommended: *"A single multi-modal hypernetwork predicts aTLAS coefficients
      from a text description and 0–16 support images, matching per-task aTLAS
      adaptation at a fraction of the inference cost on held-out datasets."*
      Everything below serves this sentence; if a task does not, drop it.
- [ ] **Confirm GPU budget.** Estimate: 1 ViT-B/32 meta-training run takes
      roughly 6–12 GPU-hours (100 epochs × 20 episodes/epoch × forward-backward
      through CLIP). With 3 seeds × text-only + multi-modal × 1 backbone,
      plan for 4–8 GPUs available continuously for 3 days.
- [ ] **Pull the official `neurips_2026.sty`** from
      <https://neurips.cc/Conferences/2026/CallForPapers> (requires login) and
      drop it into `paper/`. Update the `\usepackage` line in `paper/main.tex`
      (it currently uses the 2025 file as a placeholder).
- [ ] **Verify the paper compiles end-to-end** with `pdflatex && bibtex && pdflatex × 2`.
- [ ] **Create a results directory layout** so aggregation later is mechanical:
      `results/{model}/{method}/{dataset}/seed{N}.json`.

**STOP-CHECK 0:** If GPU access is shaky or you cannot get a meta-training run
to start by the end of Day 1, pivot to option 2.

---

## Day 0 — Today (Apr 30) — Setup and kick-off

Goal: long-running compute is going by tonight; the paper scaffold is built;
you wake up tomorrow with data starting to land.

### Compute (highest priority — start first)

- [ ] **Start ViT-B/32 multi-modal meta-training, seed 0.**
      ```bash
      python src/learn_multimodal_to_coef.py \
          --model ViT-B-32 \
          --meta-train-datasets CIFAR10,EuroSAT,DTD,GTSRB,SVHN,Food101 \
          --meta-val-datasets Caltech101,Flowers102 \
          --hypernetwork-arch medium \
          --fusion-mode concat \
          --num-shots 4 \
          --image-pooling mean \
          --text-input-mode dataset \
          --variable-shots \
          --meta-epochs 60 \
          --episodes-per-epoch 20 \
          --blockwise-coef \
          --seed 0
      ```
      Reduced from the recommended 100 to 60 epochs given the time budget.
      Watch the val curve at hour 1; if it isn't moving, kill and adjust LR.
- [ ] **Start ViT-B/32 text-only meta-training, seed 0**, in parallel on a
      different GPU (`learn_text_to_coef.py`). Same dataset split.
- [ ] **If you have ≥4 GPUs**, also kick off seeds 1 and 2 of the multi-modal
      run. If only 1–2 GPUs, multi-modal seed 0 takes priority and seeds 1/2
      run sequentially after.
- [ ] **Smoke-test the eval pipeline** while training runs: load any random
      checkpoint and run `eval_multimodal_adaptation.py --eval-mode sweep`
      against one held-out dataset. Catch I/O bugs *now*, not on Day 4.

### Code (in parallel with training)

- [ ] **Build `scripts/build_main_table.py`.** Walks
      `results/{model}/{method}/{dataset}/seed*.json`, joins on (method, K),
      computes mean ± std across seeds and across held-out datasets, emits
      both CSV and a LaTeX `\begin{tabular}` block. This script is on the
      critical path — without it you will lose hours hand-pasting numbers.
      Acceptance test: run it on synthetic JSONs in a `tmp/` dir; output
      should be a paste-ready LaTeX table.
- [ ] **Build `scripts/build_ablation_table.py`** (sibling of the above) that
      filters by a single ablation axis (e.g. `--axis fusion-mode`) and emits
      a small comparison table.
- [ ] **Generate text descriptions for held-out datasets.** Use either the
      manual loader for tasks where descriptions already exist, or run:
      ```bash
      python -m src.text_descriptions.generators \
          --dataset Flowers102 --llm gpt4o --num-descriptions 15
      # repeat for: Caltech101, Cars, RESISC45, SUN397
      ```
      Keep this on a separate machine — does not need GPU. Costs ~$2 per
      dataset.

### Paper

- [ ] **Open `paper/main.tex`** and write a tight one-paragraph introduction.
      Just stating the problem, the contribution, and the result you expect
      to land. Do not polish; it will get rewritten three times.
- [ ] **Pin the bibliography keys you'll cite** by adding stub `@inproceedings`
      entries to `paper/references.bib` with placeholder titles. Even if
      empty, this lets you `\cite{...}` while drafting and iterate later.

**STOP-CHECK 1 (end of Day 0):** Can you confirm at least one meta-training
run is making val-loss progress? If no, debug; if still no by Day 1 morning,
pivot to option 2.

---

## Day 1 — May 1 — Generate baselines, build evaluation harness

Goal: by tonight you have a first set of held-out numbers from at least one
run, and baselines started in parallel.

### Compute

- [ ] **Check overnight training.** If val-acc stalled, debug LR / batch size /
      gradient clipping. Common issue: hypernet output collapses to zero —
      check `text_only_proj` initialization.
- [ ] **Start baseline runs** that are needed for the main table:
  - [ ] CLIP zero-shot on all 5 held-out datasets (~minutes).
  - [ ] Per-task aTLAS at K ∈ {1, 2, 4, 8, 16} on all 5 held-out datasets
        via `learn_few_shots.py` (existing script, no new code).
  - [ ] Tip-Adapter and Tip-Adapter-F via `learn_few_shots.py --adapter tip`
        and a Tip-F variant.
  - [ ] LP++ via `--adapter lpp`.
  - [ ] CoOp baseline if a working implementation is reachable; otherwise
        cite published numbers and note in the paper. This is acceptable.
- [ ] **Once seed 0 multi-modal training finishes**, run
      `eval_multimodal_adaptation.py --eval-mode sweep` on every held-out
      dataset. Drop JSONs into the `results/` layout from Day 0.

### Paper

- [ ] **Write the related-work section.** Three subsections from the LaTeX
      backbone (task arithmetic, hypernetworks, CLIP few-shot). Aim for ~3/4
      page. Use placeholder citations from the seeded `.bib`.
- [ ] **Write the background section.** Mostly notation — half a page.
- [ ] **Sketch Figure 1** (the headline figure). Two options:
      (a) accuracy-vs-K line plot showing multi-modal beats text-only beats
          zero-shot;
      (b) schematic of the multi-modal hypernetwork (text branch + image
          branch + fusion + coefficient output).
      Recommendation: do (b) early since it gates how you describe the method,
      and add (a) as Figure 2 once results land. Sketch in PowerPoint /
      tikz / Excalidraw — anything that produces a PDF.

**STOP-CHECK 2 (end of Day 1):** Do you have at least one held-out accuracy
number from your trained hypernetwork that beats CLIP zero-shot? If yes,
proceed. If no — and the gap is large — there is a real risk the method does
not work as advertised. Inspect: is the predicted alpha collapsing? Are the
gradients flowing through `predicted_coef` correctly? If you cannot debug in
2-3 hours, pivot to option 2.

---

## Day 2 — May 2 — Ablations and methods write-up

Goal: ablations underway in parallel, methods section drafted, headline numbers
becoming credible.

### Compute

- [ ] **Run 2-3 ablations** in parallel on whatever GPUs are free. Pick the
      ablations that most differentiate your method:
  - [ ] **Fusion mode** (concat vs. add vs. attention) — one extra
        meta-training run per non-default mode. ~6 GPU-hours each.
  - [ ] **Variable-shots vs. fixed K=4** — a single comparison run.
  - [ ] **Hypernetwork size** (small vs. medium vs. large) — two extra runs.
  - **Tier these:** if compute is tight, do fusion mode only. The other two
        are nice-to-haves that go in the appendix even if completed.
- [ ] **If seeds 1 and 2 of the main run aren't done yet, start them now.**
      Multi-seed numbers are near-mandatory for the headline table; you can
      cut them only if you absolutely have to.
- [ ] **Finish baselines** (any not done from Day 1).

### Paper

- [ ] **Write the Method section in full.** Sections 4.1-4.4 from the
      backbone. ~1.5 pages. Use the same notation as the Background section.
- [ ] **Write the Experimental Setup subsection** (5.1). Tables of datasets,
      backbones, K values, baselines, seeds. Numbers go in later.
- [ ] **Render Figure 1 (architecture schematic)** to a final PDF and embed.
- [ ] **Run the table aggregator on whatever results exist.** It does not
      matter that the table is half-empty; you want to validate the build
      pipeline and catch column-mismatch bugs early.

---

## Day 3 — May 3 — Lock results, draft remaining sections

Goal: by tonight all main-table numbers and at least one ablation are
finalized; experiments section is fully written; abstract-and-title are
locked for tomorrow's abstract submission.

### Compute

- [ ] **Re-run the table aggregator** with all available results.
- [ ] **Generate Figure 2 (accuracy-vs-K plot)** from the aggregator output.
      A small `scripts/plot_accuracy_vs_k.py` helper.
- [ ] **Per-dataset breakdown table** for the appendix (rows = datasets,
      columns = methods × K subset). Generated by the same aggregator with
      a flag.
- [ ] **If multi-seed runs aren't done**, accept N=1 for the harder seeds and
      flag this honestly in the paper. Reviewers prefer "one seed, will add"
      over "no seeds at all."

### Paper

- [ ] **Write the Experiments section** (5.2-5.6) using real numbers from
      the aggregator. Cross-architecture and compute analysis subsections
      can be appendix-only if time is tight.
- [ ] **Write Discussion / Limitations.** Honesty here helps reviewers; don't
      oversell. Half a page.
- [ ] **Write the Conclusion.** Three sentences will do for the first pass.
- [ ] **Write the Abstract** based on the introduction and the locked
      headline numbers.
- [ ] **Lock the title.** No more changes after this.

**STOP-CHECK 3 (end of Day 3):** Does the paper read end-to-end as a coherent
8-9 page draft, even if rough? If no, you will likely not make the May 6
deadline at acceptable quality. Consider pivot to option 2.

---

## Day 4 — May 4 — Abstract submission day, polish

Goal: submit the abstract before AOE; spend the rest of the day on figures,
tables, and polish.

### Logistics

- [ ] **Submit the abstract on OpenReview** before 23:59 AOE. Keep it ~150-200
      words. Title and author list must match what you'll submit on May 6
      (title in particular cannot change after the abstract deadline at most
      venues — verify on the NeurIPS 2026 instructions page).
- [ ] **Lock the author list and ordering.** No more changes.

### Paper

- [ ] **Pass 1 polish on the introduction.** Do not let it stay rough.
- [ ] **Polish Method section** — make sure notation is consistent across
      Background and Method; reviewers will hammer on this.
- [ ] **Polish all tables** — units, captions, bolding the best, footnotes
      explaining any anomaly.
- [ ] **Polish Figure 1 and Figure 2** — readable axes, legible legend at
      print scale, color-blind-friendly palette.
- [ ] **Add a "Reproducibility" paragraph** in the appendix (or end of
      methods) listing repo URL placeholder, hyperparameters, seeds, hardware.
- [ ] **Begin the NeurIPS Paper Checklist.** This is mandatory and absence is
      grounds for desk reject. The official 2026 sty file ships with the
      checklist macros — answer every item honestly.

---

## Day 5 — May 5 — Final integration and review

Goal: paper reads cleanly end-to-end; checklist done; one external pair of
eyes has read it.

### Paper

- [ ] **Read the paper top-to-bottom out loud.** Catches at least 30%
      of the remaining typos and awkward sentences.
- [ ] **Fact-check every number in every table** against its source JSON.
      Off-by-one errors here are common and cost reviewer trust.
- [ ] **Check that every claim in the abstract and introduction is supported
      by a concrete experimental result later in the paper.** If not, either
      cut the claim or run the experiment.
- [ ] **Finish the NeurIPS Paper Checklist completely.**
- [ ] **Hand the draft to one colleague** for a fast read-through. 30 minutes
      of fresh eyes catches more than 3 hours of self-review.
- [ ] **Build the supplementary material PDF** (full per-dataset tables,
      additional ablations, failure cases, hyperparameters).
- [ ] **Verify page limit.** NeurIPS 2026 main content limit is 9 pages
      excluding references and the checklist. Trim aggressively if over.
- [ ] **Ensure the paper is anonymized** — the default `neurips_2026` option
      with no `[final]` does this for the title block, but check there are
      no self-identifying mentions in the text, code repo URLs, or figure
      captions.

---

## Day 6 — May 6 — Submission day

Goal: submitted, with at least 4 hours of slack before AOE. Do not aim for
the deadline minute.

- [ ] **Final pdflatex build** with no warnings (or only known-acceptable
      ones).
- [ ] **Verify the PDF visually** at 100% zoom, all pages.
- [ ] **Upload main paper PDF to OpenReview.**
- [ ] **Upload supplementary material zip** (extra figures, full tables,
      anonymized code if you choose).
- [ ] **Confirm the submission.** Screenshot the OpenReview submission page.
- [ ] **Then, and only then, breathe.**

---

## Risk register

Things most likely to derail this plan, with rough mitigations:

| Risk                                          | Mitigation                                             |
|-----------------------------------------------|--------------------------------------------------------|
| Meta-training does not converge               | LR sweep, smaller hypernetwork, sanity-check gradients |
| Multi-modal does not beat text-only           | Reframe paper around text-only + thorough ablation     |
| Compute insufficient                          | Drop seeds 1/2, drop ablations, ViT-B/32 only          |
| Per-task aTLAS baseline too strong to match   | Lean on the *efficiency* angle, not just accuracy      |
| Tip-F / LP++ baselines beat us at high K      | Honest in paper; pitch as "competitive at low K"       |
| Bug discovered late in eval pipeline          | The smoke-test on Day 0 is meant to prevent this       |

---

## What to drop, in order, if you fall behind

1. Cross-architecture generalization experiment (defer to extension).
2. Synthetic-data experiments (defer entirely).
3. Hypernetwork-size ablation (keep small only).
4. Text-source ablation (use manual descriptions only).
5. ViT-B/16 and ViT-L/14 (ViT-B/32 only).
6. Multi-seed runs (single seed, flag honestly).
7. Shot-pooling ablation.
8. Variable-K ablation.

If you reach #5 or below, seriously consider pivoting to option 2 (a later
venue) instead of submitting a thin paper.

---

## Post-submission (May 7+)

- [ ] **Tag the submitted commit** with `git tag neurips-2026-submission`.
- [ ] **Run the experiments you cut**, especially seeds and the
      cross-architecture transfer. These strengthen the rebuttal.
- [ ] **Draft an arXiv preprint** (use `[preprint]` option on the sty file).
      Post 1-2 weeks after submission.
- [ ] **Prepare for rebuttal** — reviews land in late July. Have your "if a
      reviewer asks X" answers cached.
