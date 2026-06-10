# Codebook / Data Dictionary

This codebook defines every variable in the analysis dataset. To run the study on
real data, provide `data/raw/survey_raw.csv` with the **raw** columns below.

> **Design note.** Demographic covariates (age, sex, residence, health insurance) are
> intentionally excluded from this study design. Models are adjusted for the
> **Self_Treat** attitude item instead (see below).

## Raw survey variables (`data/raw/survey_raw.csv`)

| Variable | Type | Values / Range | Description |
|----------|------|----------------|-------------|
| `respondent_id` | int | 1...N | Unique respondent identifier |
| `education` | str | `None`, `Primary`, `Secondary`, `Tertiary` | Highest education completed |
| `income_monthly` | float | >= 0 | Monthly household income (local currency units) |
| `occupation` | str | `Unemployed`, `Manual`, `Skilled`, `Professional` | Occupation category |
| `chronic_condition` | str | `Yes`, `No` | Self-reported chronic illness |
| `hisb_score` | int | 0-20 | Health information-seeking behaviour scale (higher = more seeking) |
| `internet_access` | str | `Yes`, `No` | Regular internet access |
| `self_treat` | str | `Strongly disagree`, `Disagree`, `Neutral`, `Agree`, `Strongly agree` | Self_Treat item: agreement with "I can usually self-treat with remedies that are available without a doctor's prescription." |
| `self_medication` | int | 0, 1 | Practised self-medication in the past 6 months (1 = yes) |
| `self_medication_freq` | str | `Never`, `Rarely`, `Sometimes`, `Often` | Frequency of self-medication |

## Derived variables (created in `data_preprocessing.py`)

| Variable | Type | Description |
|----------|------|-------------|
| `education_code` | int | Ordinal encoding of `education` (0-3) |
| `occupation_code` | int | Ordinal encoding of `occupation` (0-3) |
| `income_z` | float | Standardized (z-score) `income_monthly` |
| `education_z` | float | Standardized `education_code` |
| `occupation_z` | float | Standardized `occupation_code` |
| `ses_score` | float | SES composite = mean of the three z-scores |
| `ses_tertile` | str | `Low` / `Middle` / `High` (tertiles of `ses_score`) |
| `hisb_high` | int | 1 if `hisb_score` >= sample median, else 0 |
| `self_treat_score` | int | Self_Treat encoded 1-5 (Strongly disagree=1 ... Strongly agree=5) |

## Outcome

- **Primary:** `self_medication` (binary).
- **Secondary:** `self_medication_freq` (ordinal).

## Covariate

- `self_treat_score` (the 1-5 encoding of the Self_Treat item) is used as the
  adjustment covariate in all multivariable models, treated as a numeric (per-point) term.
