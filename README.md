# SES, Health Information Seeking, and Self-Treatment Attitude as Predictors of Self-Medication

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Nurudeen Showole, MD; Suhila Sawesi, PhD  
**Affiliation:** College of Computing, Grand Valley State University

Cross-sectional analysis of the 2021 NCME&PR survey examining how socioeconomic
status (SES), health information-seeking behaviour (HISB), and self-treatment
attitude predict daily over-the-counter (OTC) drug use and herbal supplement use.

This repository implements the **manuscript analysis**: descriptive statistics /
prevalence tables and multivariable binary logistic regression. Prior
hypothesis-driven mediation, ordinal, subgroup, multiple-imputation, and
calibration pipelines have been removed.

## Study aims (brief)

1. Describe SES, HISB, self-treatment attitude, and self-medication prevalence.
2. Estimate adjusted associations of SES tertile, HISB, and self-treatment attitude
   with OTC and herbal use via binary logistic regression.

## Analysis design

| Element | Specification |
| --- | --- |
| Outcomes | `otc_use`, `herbal_use` (any daily use from `NumOTC` / `NumHerbal`) |
| SES | Composite of education + household income (z-scored mean), tertiles Low / Middle / High (reference = Low) |
| HISB | Composite of Med7-Med9, DTCA items, and information-source count (**Self_Treat excluded**) |
| Self_Treat | Separate Likert predictor, entered as `self_treat_z` |
| Primary model | `logit(outcome ~ SES tertile + HISB + Self_Treat_z)` |

Crude SES-only, HISB-only, and Self_Treat-only models are also fit for context.

## Project structure

```
config/config.yaml      # paths and outcome definitions
data/raw/               # survey_raw.csv (not committed if large/PII)
docs/
  codebook.md
  methodology.md
results/                # analysis_dataset.csv, Excel, table CSVs
src/
  constants.py
  preprocess.py
  descriptives.py
  models.py
  tables.py
  export_excel.py
tests/
run_analysis.py
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python run_analysis.py
```

Outputs:

- `results/analysis_dataset.csv` — analysis-ready dataset
- `results/tables/*.csv` — individual tables
- `results/NCME_2021_Manuscript_Analysis_Results.xlsx` — combined workbook

## Primary adjusted results (manuscript)

In the adjusted model (`outcome ~ SES + HISB + Self_Treat_z`), HISB is the most
consistent correlate of both OTC and herbal use; high SES and self-treatment
attitude are more relevant for herbal use. Re-run `python run_analysis.py` to
reproduce Table 07 (`07_combined_or`) from the current data extract.

## Citation

See `CITATION.cff`.
