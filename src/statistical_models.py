"""Inferential modelling: multivariable logistic regression.

For **each** self-medication outcome (OTC use, herbal use) three nested models are fit:

* ``model_crude_ses``     - outcome ~ SES tertile only.
* ``model_adjusted``      - outcome ~ SES tertile + HISB (+ any covariates).
* ``model_interaction``   - adjusted model + SES x HISB interaction.

Each model's coefficients are exported as odds ratios (OR) with 95% CIs, plus a
model-comparison table (pseudo-R2, AIC, log-likelihood, LR test vs. crude). Output
files are suffixed with the outcome name, e.g. ``model_adjusted__otc_use.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import model_spec
from .config import CONFIG, PATHS

# ``model.predict`` and the forest plot need the adjusted model's outcome label.
PRIMARY_OUTCOME = "otc_use"


def fit_logit(df: pd.DataFrame, formula: str):
    """Fit a binary logistic regression via statsmodels and return the result."""
    return smf.logit(formula, data=df).fit(disp=False)


def odds_ratio_table(result, model_name: str) -> pd.DataFrame:
    """Convert a fitted logit result into a tidy OR table with 95% CIs."""
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
    for col in ["coef_logit", "odds_ratio", "or_ci_low", "or_ci_high"]:
        table[col] = table[col].round(4)
    table["p_value"] = table["p_value"].round(4)
    return table.reset_index(drop=True)


def lr_test(restricted, full) -> tuple[float, int, float]:
    """Likelihood-ratio test comparing a restricted vs. full nested model."""
    stat = 2 * (full.llf - restricted.llf)
    dof = int(full.df_model - restricted.df_model)
    from scipy import stats as _stats

    p = _stats.chi2.sf(stat, dof) if dof > 0 else float("nan")
    return float(stat), dof, float(p)


def build_models(df: pd.DataFrame, outcome: str = PRIMARY_OUTCOME) -> dict:
    """Fit the crude, adjusted, and interaction models for one outcome."""
    f_crude = f"{outcome} ~ {model_spec.crude_ses_rhs()}"
    f_adjusted = f"{outcome} ~ {model_spec.adjusted_rhs()}"
    f_interaction = f"{outcome} ~ {model_spec.interaction_rhs()}"
    return {
        "model_crude_ses": fit_logit(df, f_crude),
        "model_adjusted": fit_logit(df, f_adjusted),
        "model_interaction": fit_logit(df, f_interaction),
    }


def comparison_table(models: dict) -> pd.DataFrame:
    rows = []
    crude = models["model_crude_ses"]
    for name, res in models.items():
        lr_stat, lr_dof, lr_p = (np.nan, 0, np.nan)
        if name != "model_crude_ses":
            lr_stat, lr_dof, lr_p = lr_test(crude, res)
        rows.append(
            {
                "model": name,
                "n": int(res.nobs),
                "log_likelihood": round(res.llf, 3),
                "pseudo_r2_mcfadden": round(res.prsquared, 4),
                "aic": round(res.aic, 2),
                "bic": round(res.bic, 2),
                "lr_vs_crude_stat": round(lr_stat, 3) if lr_dof else np.nan,
                "lr_vs_crude_dof": lr_dof,
                "lr_vs_crude_p": round(lr_p, 4) if lr_dof else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(df: pd.DataFrame) -> dict:
    """Fit models for every outcome, export tables, and return fitted results.

    Returns a nested dict: ``{outcome_name: {model_name: fitted_result}}``.
    """
    PATHS.ensure_dirs()
    all_models: dict[str, dict] = {}
    for oc in model_spec.outcomes():
        outcome = oc["name"]
        models = build_models(df, outcome)
        all_models[outcome] = models

        for name, res in models.items():
            table = odds_ratio_table(res, name)
            table.insert(0, "outcome", outcome)
            path = PATHS.tables_dir / f"{name}__{outcome}.csv"
            table.to_csv(path, index=False)
            print(f"[models] wrote {path.name}")

        comp = comparison_table(models)
        comp.insert(0, "outcome", outcome)
        comp.to_csv(PATHS.tables_dir / f"model_comparison__{outcome}.csv", index=False)
        print(f"[models] wrote model_comparison__{outcome}.csv")
    return all_models


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    _ = CONFIG
    return run(df)


if __name__ == "__main__":
    main()
