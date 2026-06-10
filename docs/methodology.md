# Methodology

## Study design

Cross-sectional, questionnaire-based survey of adults (? 18 years). The analytic
goal is to quantify the association between **socioeconomic status (SES)** and
**health informationùseeking behaviour (HISB)** with **self-medication practices**.

## Measures

### Outcome ù self-medication
Use of medicines (prescription or over-the-counter) without professional consultation
in the past 6 months. Captured as a binary indicator and, secondarily, as an ordinal
frequency (`Never`/`Rarely`/`Sometimes`/`Often`).

### Exposure 1 ù Socioeconomic status (SES)
A composite index built from three standardized components:

1. Education (ordinal: None ? Tertiary)
2. Monthly household income (continuous)
3. Occupation prestige (ordinal: Unemployed ? Professional)

Each component is converted to a z-score; the SES score is their mean. The score is
divided into tertiles (**Low / Middle / High**) for interpretable group comparisons.

### Exposure 2 ù Health informationùseeking behaviour (HISB)
A 0ù20 summated scale capturing how actively a respondent seeks health information
(e.g., from clinicians, internet, pharmacists, media). Analysed continuously and as a
median split (`hisb_high`).

### Covariate
**Self_Treat** attitude item: agreement with "I can usually self-treat with remedies
that are available without a doctor's prescription," recorded on a 5-point Likert scale
and encoded 1-5 (`self_treat_score`). It is used as the adjustment covariate in all
multivariable models, entered as a numeric (per-point) term. Demographic covariates
(age, sex, residence, health insurance) are intentionally excluded from this design.

## Analysis plan

1. **Descriptive statistics.** Distributions of all variables; means/SD for continuous,
   counts/percent for categorical.
2. **Bivariate analysis.** Chi-square tests for categorical predictors vs.
   self-medication; t-test/ANOVA for continuous predictors across the outcome.
3. **Multivariable logistic regression.** Self-medication regressed on SES tertile and
   HISB, adjusting for covariates. Effects reported as adjusted odds ratios (aOR) with
   95% CIs. Model fit assessed with pseudo-Rù, AIC, and the likelihood-ratio test.
4. **Interaction.** SES ù HISB interaction term tested as a sensitivity analysis.
5. **Ordinal model (secondary outcome).** Self-medication *frequency*
   (Never < Rarely < Sometimes < Often) modelled with a **proportional-odds
   (cumulative-logit) ordinal logistic regression** using the same predictor set.
   Effects are reported as proportional-odds ratios (OR > 1 indicates higher odds of a
   *more frequent* category). The model also yields predicted category probabilities,
   summarised across SES tertiles in `fig_freq_pred_by_ses.png`.
6. **Proportional-odds assumption (Brant test).** The parallel-lines assumption is
   assessed with the **Brant (1990) test**. We fit the J = K-1 cumulative binary
   logistic models P(Y >= j) and test (Wald-type chi-square) whether the predictor
   slopes are equal across cutpoints, both overall (omnibus) and per predictor. A small
   p-value (< 0.05) flags a violation, suggesting a partial-proportional-odds or
   multinomial alternative for that term. Output: `model_ordinal_brant.csv`.
   When the assumption is violated, a **partial proportional-odds (generalized ordered
   logit)** model is fitted by maximum likelihood: terms that satisfy the assumption keep
   a single shared coefficient, while Brant-flagged terms are freed to take
   cutpoint-specific coefficients (reported as odds ratios per cumulative cutpoint
   P(Y >= category)). Outputs: `model_partial_po.csv`, `model_partial_po_fit.csv`, and
   `fig_partial_po_or.png`.
7. **Mediation analysis (SES -> HISB -> self-medication).** We test whether health
   information-seeking mediates the SES-self-medication association using the
   product-of-coefficients approach:
   - Mediator model (OLS): `hisb_score ~ ses_score + covariates` gives path *a*.
   - Outcome model (logistic): `self_medication ~ ses_score + hisb_score + covariates`
     gives the direct effect *c'* (SES) and path *b* (HISB).
   - **Indirect (mediated) effect** = *a* x *b*; **total** = *c'* + *a*b* (approximate,
     log-odds scale); **proportion mediated** = indirect / total.
   Inference uses a **nonparametric case-resampling bootstrap** (B = 1000 by default,
   configurable) with percentile 95% confidence intervals. When the direct and indirect
   effects carry opposite signs (competitive / inconsistent mediation), the proportion
   mediated is reported only for completeness. Outputs: `mediation_results.csv` and
   `fig_mediation_effects.png`.

## Statistical software

Python ? 3.10 with `pandas`, `statsmodels`, `scipy`, `scikit-learn`,
`matplotlib`, and `seaborn`. Significance threshold ? = 0.05 (two-sided).

## Reproducibility

A fixed random seed governs synthetic-data generation and any stochastic steps. All
outputs (tables, figures) are written to `results/` by a single orchestrating script
(`src/run_pipeline.py`).

## Ethical note

The bundled dataset is **synthetic** and contains no real individuals. When using real
survey data, ensure appropriate ethics approval and informed consent, and keep raw data
out of version control (already enforced via `.gitignore`).
