# NeurIPS 2026 paper draft

This directory holds the LaTeX source for the submission.

## Files

- `main.tex` — paper backbone (sections only; content as comments).
- `references.bib` — bib file (seed entries as TODOs).
- `neurips_2026.sty` — official NeurIPS 2026 style file.
- `checklist.tex` — mandatory NeurIPS Paper Checklist (answers stubbed as
  `\answerTODO`/`\justificationTODO`; `\input` from `main.tex`).
- `neurips_2026_reference.tex` — the official template, kept verbatim as a
  reference for option flags and example formatting. Not built.

## Build

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission build options

Edit the `\usepackage` line in `main.tex`. The default (no option) is
equivalent to `[main]` and produces an anonymized double-blind submission.

| Option flag                  | When to use                              |
|------------------------------|------------------------------------------|
| (default) / `[main]`         | Main Track submission                    |
| `[position]`                 | Position Paper Track                     |
| `[eandd]`                    | Evaluations & Datasets Track             |
| `[eandd, nonanonymous]`      | E&D Track, single-blind opt-in           |
| `[creativeai]`               | Creative AI Track                        |
| `[sglblindworkshop]`         | Workshop (single-blind)                  |
| `[dblblindworkshop]`         | Workshop (double-blind)                  |
| `[preprint]`                 | arXiv / public preprint                  |
| `[main, final]`              | Camera-ready (only after acceptance)     |
| `[nonatbib]`                 | Avoid loading natbib (rarely needed)     |

For workshop builds, also provide `\workshoptitle{...}` in the preamble.

## Page limit reminder

Main content limit is 9 pages. References, the appendix, and the checklist do
**not** count against this limit. Papers without the checklist are
desk-rejected.
