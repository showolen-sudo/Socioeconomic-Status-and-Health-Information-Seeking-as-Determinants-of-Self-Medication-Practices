"""
Step 3 - Logistic regression models and odds-ratio tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SES_TERM = "C(ses_tertile, Treatment(reference='Low'))"
HISB_TERM = "hisb_score"


def fit_logit(df: pd.DataFrame, formula: str):
    """Fit binary logistic regression; returns statsmodels result object."""
    return smf.logit(formula, data=df).fit(disp=False)


def odds_ratio_table(result, model_name: str) -> pd.DataFrame:
    """Convert logit coefficients to odds ratios with 95% CIs."""
    params = result.params
    conf = result.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    table = pd.DataFrame(
        {
            "model": model_name,
            "term": params.index,
            "coef_logit": params.values,
            "odds_ratio": np.exp(params.values),
            "or_ci_low": np.exp(conf["ci_low"].values),
            "or_ci_high": np.exp(conf["ci_high"].values),
            "p_value": result.pvalues.values,
        }
    )
    for col in ("coef_logit", "odds_ratio", "or_ci_low", "or_ci_high"):
        table[col] = table[col].round(3)
    table["p_value"] = table["p_value"].round(4)
    return table.reset_index(drop=True)


def run_all_models(df: pd.DataFrame, outcomes: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Fit models for each outcome:
      - crude_ses: outcome ~ SES only
      - crude_self_treat_z: outcome ~ Self_Treat (z-scored) only
      - hisb_only: outcome ~ HISB only
      - adjusted: outcome ~ SES + HISB + Self_Treat (primary model)
    """
    tables: dict[str, pd.DataFrame] = {}
    for oc in outcomes:
        outcome = oc["name"]
        formulas = {
            "crude_ses": f"{outcome} ~ {SES_TERM}",
            "crude_self_treat_z": f"{outcome} ~ self_treat_z",
            "hisb_only": f"{outcome} ~ {HISB_TERM}",
            "adjusted": f"{outcome} ~ {SES_TERM} + {HISB_TERM} + self_treat_z",
        }
        comp = []
        for name, formula in formulas.items():
            fitted = fit_logit(df, formula)
            res = odds_ratio_table(fitted, name)
            res.insert(0, "outcome", outcome)
            tables[f"{name}__{outcome}"] = res
            comp.append(
                {
                    "model": name,
                    "n": int(fitted.nobs),
                    "pseudo_r2_mcfadden": round(float(fitted.prsquared), 4),
                    "aic": round(float(fitted.aic), 2),
                    "bic": round(float(fitted.bic), 2),
                }
            )
        tables[f"model_comparison__{outcome}"] = pd.DataFrame(comp)
    return tables
