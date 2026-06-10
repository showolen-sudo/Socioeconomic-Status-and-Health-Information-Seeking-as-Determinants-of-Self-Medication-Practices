# Socioeconomic Status and Health Information Seeking as Determinants of Self-Medication Practices

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/reproducible-pipeline-success.svg)](#reproducing-the-analysis)

A reproducible, end-to-end research analysis pipeline investigating how **socioeconomic
status (SES)** and **health information-seeking behaviour (HISB)** predict
**self-medication practices** in an adult population.

> **Data note.** The repository ships with a fully reproducible *synthetic* dataset
> generator so the entire pipeline runs out of the box. Replace the synthetic data with
> your real survey data (same schema, see [`docs/codebook.md`](docs/codebook.md)) to run
> the study on observed data - no code changes required.

---

## Research questions

1. **RQ1.** Is socioeconomic status associated with the likelihood of self-medication?
2. **RQ2.** Is health information-seeking behaviour associated with self-medication?
3. **RQ3.** Do SES and HISB remain significant predictors after adjusting for the
   Self_Treat attitude (`self_treat_score`)?
4. **RQ4.** Is there an interaction between SES and HISB on self-medication?
5. **RQ5.** Do SES and HISB predict the *frequency* of self-medication (ordinal)?

## Hypotheses

- **H1.** Lower SES is associated with higher odds of self-medication.
- **H2.** Higher health information-seeking behaviour is associated with self-medication
  (direction tested empirically; literature is mixed).
- **H3.** Associations persist after covariate adjustment.

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
2. Clean and build the SES index -> `data/processed/analysis.csv`
3. Produce descriptive tables -> `results/tables/`
4. Fit logistic-regression models -> `results/tables/model_*.csv`
5. Fit ordinal model + Brant test + partial PO -> `results/tables/model_ordinal_*.csv`
6. Run the mediation analysis -> `results/tables/mediation_results.csv`
7. Run subgroup analysis -> `results/tables/subgroup_*.csv`
8. Run multiple imputation (MICE) -> `results/tables/mi_*.csv`
9. Compute calibration/discrimination -> `results/tables/discrimination_metrics.csv`
10. Render figures -> `results/figures/`

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
| Descriptives | `descriptive_analysis.py` | `results/tables/descriptives_*.csv` |
| Binary modelling | `statistical_models.py` | `results/tables/model_*.csv` |
| Ordinal + Brant + partial PO | `ordinal_models.py` | `results/tables/model_ordinal_*.csv`, `model_partial_po*.csv` |
| Mediation analysis | `mediation_analysis.py` | `results/tables/mediation_results.csv` |
| Subgroup analysis | `subgroup_analysis.py` | `results/tables/subgroup_*.csv` |
| Multiple imputation | `multiple_imputation.py` | `results/tables/mi_*.csv` |
| Calibration & discrimination | `calibration.py` | `results/tables/discrimination_metrics.csv`, `calibration_curve.csv`, `hosmer_lemeshow.csv` |
| Figures | `visualization.py` and analysis modules | `results/figures/*.png` |

---

## Statistical methods

- **Descriptive statistics:** frequencies, means/SD, group comparisons.
- **Bivariate tests:** Pearson chi-square (categorical), independent t-test / ANOVA.
- **SES index:** standardized composite of education, income, and occupation
  (z-score mean), binned into tertiles (Low / Middle / High).
- **Primary model:** multivariable **binary logistic regression** with
  self-medication (yes/no) as the outcome; reports adjusted odds ratios (aOR) with
  95% confidence intervals.
- **Secondary model:** **proportional-odds ordinal logistic regression** on
  self-medication *frequency* (Never < Rarely < Sometimes < Often); reports
  proportional-odds ratios and predicted category probabilities by SES. The
  parallel-lines assumption is checked with a **Brant test**, and a
  **partial proportional-odds (generalized ordered logit)** model relaxes the
  constraint for any Brant-flagged terms (cutpoint-specific odds ratios).
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
