# Socioeconomic Status and Health Information Seeking as Determinants of Self-Medication Practices

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/reproducible-pipeline-success.svg)](#reproducing-the-analysis)

**Authors:** [Nurudeen Showole, MD](https://github.com/showolen-sudo/Socioeconomic-Status-and-Health-Information-Seeking-as-Determinants-of-Self-Medication-Practices/blob/main/docs/DR_NURUDEEN_SHOWOLE_Resume_GA.pdf) (1); Suhila Sawesi, PhD (1)

(1) Grand Valley State University

A reproducible, end-to-end research analysis pipeline investigating how **socioeconomic
status (SES)** and **health information-seeking behaviour (HISB)** predict
**self-medication practices** in an adult population. Self-medication is operationalized
as the daily use of **over-the-counter (OTC) drugs** and **herbal supplements**, analysed
as two separate outcomes.

> **Data note.** The repository ships with a fully reproducible *synthetic* dataset
> generator so the entire pipeline runs out of the box. Replace the synthetic data with
> your real survey data (same schema, see [`docs/codebook.md`](docs/codebook.md)) to run
> the study on observed data - no code changes required.

---

## Research questions

Each question is answered separately for the two self-medication outcomes: **OTC drug
use** and **herbal supplement use**.

1. **RQ1.** Is socioeconomic status associated with the likelihood of self-medication?
2. **RQ2.** Is health information-seeking behaviour (HISB) associated with self-medication?
3. **RQ3.** Do SES and HISB each remain significant predictors when modelled together?
4. **RQ4.** Is there an interaction between SES and HISB on self-medication?
5. **RQ5.** Do SES and HISB predict the *frequency* of self-medication (ordinal counts)?

## Hypotheses

- **H1.** Lower SES is associated with higher odds of self-medication.
- **H2.** Higher health information-seeking behaviour is associated with self-medication
  (direction tested empirically; literature is mixed).
- **H3.** SES and HISB remain independently associated when modelled jointly.

---

## Project structure

```
self-medication-ses-study/
+-- README.md                  # You are here
+-- LICENSE                    # MIT license
+-- CITATION.cff               # How to cite this work
+-- requirements.txt           # Python dependencies (pinned, minimal)
+-- pyproject.toml             # Tooling config (ruff, pytest)
+-- .gitignore
+-- config/
|   +-- config.yaml            # Single source of truth for paths & parameters
+-- data/
|   +-- raw/                   # Drop real raw survey data here (git-ignored)
|   +-- processed/             # Cleaned analysis-ready data (git-ignored)
+-- docs/
|   +-- codebook.md            # Variable definitions / data dictionary
|   +-- methodology.md         # Study design, measures, analysis plan
+-- src/
|   +-- config.py              # Loads config.yaml
|   +-- model_spec.py          # Shared predictor formulas + outcome list
|   +-- generate_synthetic_data.py
|   +-- data_preprocessing.py
|   +-- descriptive_analysis.py
|   +-- statistical_models.py  # Binary logistic regression
|   +-- ordinal_models.py      # Proportional-odds + Brant test + partial PO
|   +-- mediation_analysis.py  # SES -> HISB -> self-medication (bootstrap)
|   +-- subgroup_analysis.py   # Stratified ORs + effect-modification tests
|   +-- multiple_imputation.py # MICE vs complete-case (Rubin pooling)
|   +-- calibration.py         # ROC-AUC, Brier, calibration curve, HL test
|   +-- visualization.py
|   +-- run_pipeline.py        # Orchestrates the full analysis
+-- results/
|   +-- figures/               # Generated plots (git-ignored)
|   +-- tables/                # Generated tables (git-ignored)
+-- tests/
    +-- test_pipeline.py       # Smoke + sanity tests
```

---

## Quick start

### 1. Set up the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python -m src.run_pipeline
```

This will:

1. Generate a synthetic dataset -> `data/raw/survey_raw.csv`
2. Clean and build the SES + HISB composites -> `data/processed/analysis.csv`
3. Produce descriptive tables -> `results/tables/`
4. Fit logistic-regression models (per outcome) -> `results/tables/model_*__<outcome>.csv`
5. Fit ordinal model + Brant test + partial PO -> `results/tables/model_ordinal_*__<outcome>.csv`
6. Run the mediation analysis -> `results/tables/mediation_results__<outcome>.csv`
7. Run subgroup analysis -> `results/tables/subgroup_*__<outcome>.csv`
8. Run multiple imputation (MICE) -> `results/tables/mi_*__<outcome>.csv`
9. Compute calibration/discrimination -> `results/tables/discrimination_metrics__<outcome>.csv`
10. Render figures -> `results/figures/`

All per-outcome outputs are written twice - once for `otc_use` and once for `herbal_use`.

### 3. Run on your own data

Place your survey file at `data/raw/survey_raw.csv` matching the schema in
[`docs/codebook.md`](docs/codebook.md), then run:

```bash
python -m src.run_pipeline --no-generate
```

---

## Reproducing the analysis

The pipeline is deterministic: the random seed is fixed in `config/config.yaml`
(`seed: 42`). Re-running produces identical synthetic data, tables, and figures.

| Stage | Module | Output |
|-------|--------|--------|
| Data generation | `generate_synthetic_data.py` | `data/raw/survey_raw.csv` |
| Preprocessing | `data_preprocessing.py` | `data/processed/analysis.csv` |
| Descriptives | `descriptive_analysis.py` | `results/tables/descriptives_*.csv`, `bivariate_chi2.csv` |
| Binary modelling | `statistical_models.py` | `results/tables/model_*__<outcome>.csv` |
| Ordinal + Brant + partial PO | `ordinal_models.py` | `results/tables/model_ordinal_*__<outcome>.csv`, `model_partial_po*__<outcome>.csv` |
| Mediation analysis | `mediation_analysis.py` | `results/tables/mediation_results__<outcome>.csv` |
| Subgroup analysis | `subgroup_analysis.py` | `results/tables/subgroup_*__<outcome>.csv` |
| Multiple imputation | `multiple_imputation.py` | `results/tables/mi_*__<outcome>.csv` |
| Calibration & discrimination | `calibration.py` | `results/tables/discrimination_metrics__<outcome>.csv`, `calibration_curve__<outcome>.csv`, `hosmer_lemeshow__<outcome>.csv` |
| Figures | `visualization.py` and analysis modules | `results/figures/*.png` |

---

## Statistical methods

- **Descriptive statistics:** frequencies, means/SD, group comparisons.
- **Bivariate tests:** Pearson chi-square of each categorical predictor vs. each outcome.
- **SES index:** standardized composite of **education** and **household income**
  (z-score mean), binned into tertiles (Low / Middle / High).
- **HISB composite:** a single standardized score combining the active information-seeking
  items (Med7-9, the two DTCA items, and Self_Treat) with information-source breadth (count
  of `Info_*` sources). Entered as a per-unit predictor.
- **Outcomes:** self-medication is modelled as two separate outcomes - **OTC drug use**
  (`otc_use`) and **herbal supplement use** (`herbal_use`) - each derived from a daily-use
  count (`NumOTC`, `NumHerbal`).
- **Primary model:** multivariable **binary logistic regression** with each outcome
  (yes/no) regressed on SES + HISB; reports adjusted odds ratios (aOR) with 95% CIs.
- **Secondary model:** **proportional-odds ordinal logistic regression** on
  self-medication *frequency* (None < One < Two < Three+); reports proportional-odds
  ratios and predicted category probabilities by SES. The parallel-lines assumption is
  checked with a **Brant test**, and a **partial proportional-odds (generalized ordered
  logit)** model relaxes the constraint for any Brant-flagged terms (cutpoint-specific
  odds ratios).
- **Mediation:** SES -> HISB -> self-medication via the product-of-coefficients method
  with a **nonparametric bootstrap** (direct, indirect, and total effects).
- **Subgroup analysis:** SES/HISB odds ratios fitted within strata of binary subgroups,
  plus **likelihood-ratio tests for effect modification** (SES x subgroup interaction).
- **Missing data:** **multiple imputation (MICE)** with Rubin's-rules pooling, compared
  against complete-case analysis.
- **Model performance:** **discrimination** (ROC-AUC), **overall accuracy** (Brier, log
  loss), and **calibration** (reliability curve + Hosmer-Lemeshow), all cross-validated.
- **Sensitivity:** SES x HISB interaction term.

See [`docs/methodology.md`](docs/methodology.md) for the full analysis plan.

---

## Development

```bash
# Lint & format
ruff check src tests
ruff format src tests

# Run tests
pytest -q
```

---

## Roadmap (new analysis work)

- [x] Ordinal logistic regression for self-medication *frequency* (`ordinal_models.py`)
- [x] Brant test of the proportional-odds assumption (`ordinal_models.py`)
- [x] Partial proportional-odds (generalized ordered logit) model (`ordinal_models.py`)
- [x] Mediation analysis (SES -> HISB -> self-medication) (`mediation_analysis.py`)
- [x] Subgroup analysis + effect-modification tests (`subgroup_analysis.py`)
- [x] Multiple imputation for missing data (`multiple_imputation.py`)
- [x] Calibration / discrimination metrics (`calibration.py`)

---

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this repository, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).
