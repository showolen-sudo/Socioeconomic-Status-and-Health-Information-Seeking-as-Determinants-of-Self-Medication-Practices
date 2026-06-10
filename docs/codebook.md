# Codebook / Data Dictionary

This codebook defines every variable in the analysis dataset. To run the study on
real data, provide `data/raw/survey_raw.csv` with the **raw** columns below (the column
names match the questionnaire).

> **Design note.** The study models the influence of **socioeconomic status (SES)** and
> **health information-seeking behaviour (HISB)** on self-medication. Self-medication is
> operationalized as the daily use of **over-the-counter (OTC) drugs** and **herbal
> supplements**, analysed as two separate outcomes. Demographic covariates are not used;
> the only predictors are SES and HISB. The Self_Treat item is treated as one component
> of the HISB composite (not a separate covariate).

## Raw survey variables (`data/raw/survey_raw.csv`)

### Socioeconomic status (independent)

| Variable | Type | Values / Range | Description |
|----------|------|----------------|-------------|
| `respondent_id` | int | 1...N | Unique respondent identifier |
| `Education` | str | `Less than high school`, `High school graduate`, `Some college`, `Bachelor's degree`, `Graduate degree` | Highest education completed |
| `HouseIncome` | str | `Under $25,000`, `$25,000-$49,999`, `$50,000-$74,999`, `$75,000-$99,999`, `$100,000 or more` | Household income bracket |

### Health information-seeking behaviour (independent)

| Variable | Type | Values / Range | Description |
|----------|------|----------------|-------------|
| `Med7` | int | 1-5 | "I like to gather as much information as I can before making a decision about medicines" |
| `Med8` | int | 1-5 | "I like to review information multiple times before making a decision about medicines" |
| `Med9` | int | 1-5 | "After I make a decision about medicines, I continue to look for related information" |
| `DTCA_Info` | str | `Yes`, `No` | Ever sought more information on a prescription after seeing an advertisement |
| `DTCA_Prescribe` | str | `Yes`, `No` | Ever asked a physician to prescribe a drug seen/heard advertised |
| `Self_Treat` | str | `Strongly disagree` ... `Strongly agree` | "I can usually self-treat with remedies that are available without a doctor's prescription" |
| `Info_Google` | int | 0, 1 | Depends on web search (Google/Bing/etc.) for medicine information |
| `Info_App` | int | 0, 1 | Depends on a phone/mobile app |
| `Info_Fam` | int | 0, 1 | Depends on family member, friend, or acquaintance |
| `Info_MD` | int | 0, 1 | Depends on a physician |
| `Info_RPh` | int | 0, 1 | Depends on a pharmacist |
| `Info_OtherProf` | int | 0, 1 | Depends on another health professional |
| `Info_Web` | int | 0, 1 | Depends on a web site |
| `Info_SocMedia` | int | 0, 1 | Depends on social media |
| `Info_Print` | int | 0, 1 | Depends on a printed resource (e.g. medical book) |
| `Info_Other` | int | 0, 1 | Depends on another (described) source |

### Self-medication (dependent)

| Variable | Type | Values / Range | Description |
|----------|------|----------------|-------------|
| `NumOTC` | int | >= 0 | Number of over-the-counter drugs taken every day |
| `NumHerbal` | int | >= 0 | Number of herbal supplements taken every day |

## Derived variables (created in `data_preprocessing.py`)

| Variable | Type | Description |
|----------|------|-------------|
| `education_code` | int | Ordinal encoding of `Education` (0-4) |
| `income_code` | int | Ordinal encoding of `HouseIncome` (0-4) |
| `education_z` / `income_z` | float | Standardized (z-score) education / income codes |
| `ses_score` | float | SES composite = mean of `education_z` and `income_z` |
| `ses_tertile` | str | `Low` / `Middle` / `High` (tertiles of `ses_score`) |
| `self_treat_score` | int | Self_Treat encoded 1-5 (Strongly disagree=1 ... Strongly agree=5) |
| `dtca_info_bin` / `dtca_prescribe_bin` | int | Yes/No -> 1/0 |
| `info_source_count` | int | Number of `Info_*` sources selected (0-10) |
| `hisb_score` | float | HISB composite: standardized mean of Med7-9, the two DTCA items, `self_treat_score`, and `info_source_count` |
| `hisb_high` | int | 1 if `hisb_score` >= sample median, else 0 |
| `otc_use` / `herbal_use` | int | 1 if `NumOTC` / `NumHerbal` >= 1, else 0 |
| `otc_freq` / `herbal_freq` | str | Ordinal count category: `None` / `One` / `Two` / `Three+` |

## Outcomes

Self-medication is analysed as **two separate outcomes**:

- **OTC drug use:** `otc_use` (binary) and `otc_freq` (ordinal).
- **Herbal supplement use:** `herbal_use` (binary) and `herbal_freq` (ordinal).

## Predictors

- **SES:** `ses_tertile` (categorical, reference = Low) and `ses_score` (continuous, used
  as the X variable in mediation).
- **HISB:** `hisb_score` (the single standardized composite described above), entered as
  a numeric per-unit term in every multivariable model.
