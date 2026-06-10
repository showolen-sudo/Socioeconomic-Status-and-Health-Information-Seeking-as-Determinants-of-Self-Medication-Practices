# Methodology

## Study design

Cross-sectional, questionnaire-based survey of adults. The analytic goal is to quantify
the influence of **socioeconomic status (SES)** and **health information-seeking
behaviour (HISB)** on **self-medication practices**, where self-medication is the daily
use of **over-the-counter (OTC) drugs** and **herbal supplements**.

## Measures

### Outcomes - self-medication (OTC and herbal)

Self-medication is operationalized from two daily-use counts and analysed as **two
separate outcomes**:

- **OTC drug use** from `NumOTC`: a binary indicator `otc_use` (any daily OTC use) and an
  ordinal frequency `otc_freq` (`None` < `One` < `Two` < `Three+`).
- **Herbal supplement use** from `NumHerbal`: a binary indicator `herbal_use` and an
  ordinal frequency `herbal_freq` with the same categories.

### Exposure 1 - Socioeconomic status (SES)

A composite index built from two standardized components:

1. Education (ordinal: Less than high school ... Graduate degree)
2. Household income bracket (ordinal: Under $25,000 ... $100,000 or more)

Each component is converted to a z-score; the SES score (`ses_score`) is their mean. The
score is divided into tertiles (**Low / Middle / High**) for interpretable group
comparisons.

### Exposure 2 - Health information-seeking behaviour (HISB)

A single standardized composite (`hisb_score`) summarising how actively a respondent
seeks medicine information. It is the mean of the z-scored components:

- `Med7`, `Med8`, `Med9` (5-point agreement with gathering / reviewing / continuing to
  seek medicine information),
- `DTCA_Info` and `DTCA_Prescribe` (Yes/No advertisement-driven items),
- `Self_Treat` (5-point agreement with self-treating using non-prescription remedies),
- `info_source_count` (number of `Info_*` information sources the respondent depends on).

The composite is analysed continuously (per-unit) and as a median split (`hisb_high`).
The Self_Treat item is part of this HISB construct rather than a separate covariate.

## Analysis plan

The plan below is run separately for each outcome (OTC use and herbal use). Output file
names are suffixed with the outcome (e.g. `model_adjusted__otc_use.csv`).

1. **Descriptive statistics.** Distributions of all variables; means/SD for continuous,
   counts/percent for categorical.
2. **Bivariate analysis.** Chi-square tests for categorical predictors vs. each outcome.
3. **Multivariable logistic regression.** Each outcome regressed on SES tertile and HISB.
   Effects reported as adjusted odds ratios (aOR) with 95% CIs. Model fit assessed with
   McFadden pseudo-R2, AIC, and the likelihood-ratio test vs. the crude SES model.
4. **Interaction.** SES x HISB interaction term tested as a sensitivity analysis.
5. **Ordinal model (frequency).** Each outcome's frequency
   (None < One < Two < Three+) is modelled with a **proportional-odds (cumulative-logit)
   ordinal logistic regression** using SES + HISB. Effects are reported as
   proportional-odds ratios (OR > 1 indicates higher odds of a *more frequent* category),
   with predicted category probabilities summarised across SES tertiles.
6. **Proportional-odds assumption (Brant test).** The parallel-lines assumption is
   assessed with the **Brant (1990) test**: we fit the J = K-1 cumulative binary logistic
   models P(Y >= j) and test (Wald-type chi-square) whether predictor slopes are equal
   across cutpoints, both overall (omnibus) and per predictor. When the assumption is
   violated, a **partial proportional-odds (generalized ordered logit)** model is fitted
   by maximum likelihood: terms satisfying the assumption keep a single shared
   coefficient, while Brant-flagged terms are freed to take cutpoint-specific coefficients
   (reported as odds ratios per cumulative cutpoint).
7. **Mediation analysis (SES -> HISB -> outcome).** We test whether HISB mediates the
   SES-self-medication association using the product-of-coefficients approach:
   - Mediator model (OLS): `hisb_score ~ ses_score` gives path *a*.
   - Outcome model (logistic): `outcome ~ ses_score + hisb_score` gives the direct effect
     *c'* (SES) and path *b* (HISB).
   - **Indirect (mediated) effect** = *a* x *b*; **total** = *c'* + *a*b* (approximate,
     log-odds scale); **proportion mediated** = indirect / total.
   Inference uses a **nonparametric case-resampling bootstrap** (B = 1000 by default) with
   percentile 95% confidence intervals. When direct and indirect effects carry opposite
   signs (competitive / inconsistent mediation), the proportion mediated is reported only
   for completeness.
8. **Subgroup analysis and effect modification.** For each configured binary subgroup
   variable (default: `DTCA_Info`, `DTCA_Prescribe`), the adjusted logistic model is
   fitted *within each stratum* and the SES and HISB odds ratios are reported per stratum.
   Effect modification is formally tested with a likelihood-ratio test comparing the
   pooled model with vs. without an SES x subgroup interaction.
9. **Missing data: multiple imputation.** Missing predictor values are handled with
   **multiple imputation by chained equations (MICE)**; the logistic model is fitted in
   each imputed dataset and the estimates pooled by **Rubin's rules** (reporting the
   fraction of missing information). Results are compared against a complete-case
   (listwise-deletion) analysis. On the complete synthetic data, missingness is injected
   MCAR for demonstration (set `mi.demo_missing: false` for real data).
10. **Predictive performance.** Each adjusted model's **discrimination** (ROC-AUC),
    **overall accuracy** (Brier score, log loss), and **calibration** (reliability curve
    and the Hosmer-Lemeshow goodness-of-fit test) are assessed. To avoid optimistic bias,
    predicted probabilities are generated out-of-sample by stratified k-fold
    cross-validation.

## Statistical software

Python >= 3.10 with `pandas`, `statsmodels`, `scipy`, `scikit-learn`, `matplotlib`, and
`seaborn`. Significance threshold alpha = 0.05 (two-sided).

## Reproducibility

A fixed random seed governs synthetic-data generation and any stochastic steps. All
outputs (tables, figures) are written to `results/` by a single orchestrating script
(`src/run_pipeline.py`).

## Ethical note

The bundled dataset is **synthetic** and contains no real individuals. When using real
survey data, ensure appropriate ethics approval and informed consent, and keep raw data
out of version control (already enforced via `.gitignore`).
