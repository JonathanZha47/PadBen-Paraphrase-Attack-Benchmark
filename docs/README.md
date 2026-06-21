# PADBen ICANN 2026 Papers

This directory contains the camera-ready proceedings paper and supplementary appendix for *PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks* (ICANN 2026).

## Files

| File | Description |
|------|-------------|
| [`padben_icann.pdf`](padben_icann.pdf) | Camera-ready main paper (12 pages, Springer proceedings) |
| [`padben_icann_appendix.pdf`](padben_icann_appendix.pdf) | Supplementary appendix PDF (Appendices A–F) |
| [`latex/`](latex/) | LaTeX source to rebuild both documents |

## Appendix structure

| Letter | Section |
|--------|---------|
| **A** | Overall Data Processing |
| **B** | Intrinsic Mechanisms of Paraphrase Attacks |
| **C** | Detailed Methodology |
| **D** | Detailed Task Data Setup |
| **E** | Detailed Evaluation Settings |
| **F** | Ethics Statements and Limitations |

## Rebuild

```bash
cd docs/latex
pdflatex padben_icann_appendix.tex
bibtex padben_icann_appendix
pdflatex padben_icann_appendix.tex
pdflatex padben_icann_appendix.tex
cp padben_icann_appendix.pdf ../padben_icann_appendix.pdf
```
