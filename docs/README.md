# PADBen Supplementary Appendix

This directory contains the supplementary appendix for the ICANN 2026 camera-ready paper *PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks*.

## Files

| File | Description |
|------|-------------|
| [`padben_icann_appendix.pdf`](padben_icann_appendix.pdf) | Compiled appendix PDF (Appendices A–F) |
| [`latex/`](latex/) | LaTeX source to rebuild the appendix |

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
