# Methodology

## Study design

Cross-sectional analysis of adult respondents in the 2021 National Consumer Survey
on the Medication Experience and Patient-Reported Outcomes (NCME&PR). The analytic
goal is to quantify associations of **socioeconomic status (SES)**, **health
information-seeking behaviour (HISB)**, and **self-treatment attitude** with
**self-medication**, operationalized as daily use of over-the-counter (OTC) drugs
and herbal supplements.

This document describes the **manuscript analysis** implemented in `src/` and
orchestrated by `run_analysis.py` (descriptives + binary logistic regression).
Earlier hypothesis-driven mediation, ordinal, subgroup, MI, and calibration
modules are not part of this pipeline.

## Measures

### Outcomes — self-medication (OTC and herbal)

- **OTC drug use** (`otc_use`): 1 if any daily OTC use (`NumOTC` >= 1 or
  `NumOTC_Any`), else 0.
- **Herbal supplement use** (`herbal_use`): analogous from `NumHerbal` /
  `NumHerbal_Any`.
- Optional ordinal frequency bins (`otc_freq`, `herbal_freq`: None / One / Two /
  Three+) are derived for description only; primary models are binary logistic.

### Socioeconomic status (SES)

Education and household income ordinal codes are each z-scored (population SD) and
averaged to form `ses_score`. Sample tertiles create `ses_tertile` categories
Low / Middle / High. Models use Low as the reference category.

### Health information-seeking behaviour (HISB)

HISB is a composite of:

- Medication-information items `Med7`, `Med8`, `Med9`
- Direct-to-consumer advertising items `DTCA_Info`, `DTCA_Prescribe` (Yes=1, No=0)
- Information-source breadth (`info_source_count` / `Info_Sources_Count`)

Each component is z-scored and averaged to form `hisb_score`.

**Self_Treat is excluded from HISB** so that self-treatment attitude can be
modelled as its own predictor (`self_treat_score`, entered as `self_treat_z`).

### Self-treatment attitude

`Self_Treat` Likert responses are retained as `self_treat_score` and standardized
to `self_treat_z` for regression.

## Statistical analysis

1. **Descriptives** — means/SDs for continuous scores; prevalence of OTC and
   herbal use overall and by SES tertile, HISB median split, and Self_Treat level
   (`src/descriptives.py`).
2. **Binary logistic regression** (`src/models.py`) for each outcome:
   - Crude: SES only; Self_Treat only; HISB only
   - **Primary adjusted:** `outcome ~ SES tertile + HISB + Self_Treat_z`
3. Odds ratios with 95% confidence intervals; two-sided alpha = 0.05
   (`config/config.yaml`).

Presentation tables (combined OR table, long-format adjusted ORs, Self_Treat
crude vs adjusted) are built in `src/tables.py` and written by
`src/export_excel.py`.

## Software

Python 3.10+ with pandas, numpy, scipy, statsmodels, PyYAML, and openpyxl.
Run from the repository root:

```bash
pip install -r requirements.txt
python run_analysis.py
```

## Configuration and paths

`config/config.yaml` defines the raw data path, processed dataset path, tables
directory, Excel workbook path, and outcome labels. `run_analysis.py` resolves
all paths relative to the repository root.
