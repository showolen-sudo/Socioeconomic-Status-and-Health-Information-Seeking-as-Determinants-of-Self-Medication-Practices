# Codebook — NCME&PR 2021 analysis variables

Source survey file: `data/raw/survey_raw.csv` (numeric codes). Analysis variables
are created in `src/preprocess.py` using maps in `src/constants.py`.

## Identifiers

| Variable | Description |
| --- | --- |
| `respondent_id` | Unique respondent identifier |

## Outcomes

| Variable | Description |
| --- | --- |
| `NumOTC` | Count of daily OTC products |
| `NumHerbal` | Count of daily herbal supplements |
| `otc_use` | 1 = any daily OTC use, else 0 |
| `herbal_use` | 1 = any daily herbal use, else 0 |
| `otc_freq` / `herbal_freq` | None / One / Two / Three+ (descriptive) |

## Socioeconomic status

| Variable | Description |
| --- | --- |
| `Education` | Ordinal code (see `EDUCATION_ORDER` in `src/constants.py`) |
| `HouseIncome` | Ordinal code (see `INCOME_ORDER`) |
| `education_z` / `income_z` | Standardized codes |
| `ses_score` | Mean of education_z and income_z |
| `ses_tertile` | Low / Middle / High (reference in models: Low) |

## Health information seeking (HISB)

| Variable | Role in HISB |
| --- | --- |
| `Med7`, `Med8`, `Med9` | Included (z-scored components) |
| `DTCA_Info`, `DTCA_Prescribe` | Included as `dtca_*_bin` (1=Yes, 2=No -> 1/0) |
| `Info_*` flags / `Info_Sources_Count` | Included as `info_source_count` |
| **`Self_Treat`** | **Excluded from HISB** |

| Derived | Description |
| --- | --- |
| `hisb_score` | Mean of z-scored HISB components |
| `hisb_high` | 1 if `hisb_score` >= sample median |

## Self-treatment attitude (separate predictor)

| Variable | Description |
| --- | --- |
| `Self_Treat` | Likert score (survey range 1-7) |
| `self_treat_score` | Integer copy used in descriptives |
| `self_treat_z` | Standardized score used in logistic models |

## Information sources (binary flags)

`Info_Google`, `Info_App`, `Info_Fam`, `Info_MD`, `Info_RPh`, `Info_OtherProf`,
`Info_Web`, `Info_SocMedia`, `Info_Print` (0/1). Breadth may also be supplied as
`Info_Sources_Count` in the raw extract.

## Modelling note

Primary model formula:

```text
outcome ~ C(ses_tertile, Treatment(reference='Low')) + hisb_score + self_treat_z
```

Because Self_Treat is omitted from HISB, its coefficient is interpretable as the
association of self-treatment attitude net of the remaining HISB components and SES.
