# NeurIPS 2026 paper draft

This directory holds the LaTeX source for the submission.

## Files

- `main.tex` — paper backbone (sections only; content as comments).
- `references.bib` — bib file (seed entries as TODOs).
- `neurips_2025.sty` — **placeholder** style file. The official `neurips_2026.sty`
  is gated behind authenticated NeurIPS download pages; structurally the 2026
  release is identical to 2025 every recent year. **Before submission**, drop
  the official `neurips_2026.sty` here and update the `\usepackage` line in
  `main.tex`.
- `neurips_2025.bst` — natbib-compatible bibliography style shipped alongside
  the sty file.

## Build

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission build options

Edit the `\usepackage` line in `main.tex`:

| Option            | When to use                              |
|-------------------|------------------------------------------|
| `neurips_2026`    | Submission (anonymized, default)         |
| `[preprint]`      | arXiv / public preprint                  |
| `[final]`         | Camera-ready (only after acceptance)     |
| `[nonatbib]`      | Avoid loading natbib (rarely needed)     |
