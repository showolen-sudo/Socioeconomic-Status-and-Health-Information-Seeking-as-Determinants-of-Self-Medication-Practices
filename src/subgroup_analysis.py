"""Subgroup (stratified) analysis and tests for effect modification.

For each subgroup variable (configurable; default `chronic_condition` and
`internet_access`) we:

* Fit the adjusted logistic model *within each stratum* and report the SES and HISB
  odds ratios per stratum (so readers can see whether associations differ by subgroup).
* Test for effect modification with a likelihood-ratio test comparing the pooled model
  with vs. without an SES x subgroup interaction.

Outputs:
* ``subgroup_or.csv``          - stratum-specific odds ratios.
* ``subgroup_interaction.csv`` - LR test of SES x subgroup interaction.
* ``fig_subgroup_forest.png``  - forest plot of the SES (High vs Low) OR by stratum.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from .config import CONFIG, PATHS  # noqa: E402

OUTCOME = "self_medication"
_TERMS = {
    "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
    "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
    "hisb_score": "Health info-seeking (per point)",
}


def _ses() -> str:
    ref = CONFIG["modelling"]["ses_reference"]
    return f"C(ses_tertile, Treatment(reference='{ref}'))"


def _cov_formula() -> str:
    covs = CONFIG["modelling"]["covariates"]
    numeric = {"age", "hisb_score", "ses_score", "income_monthly", "self_treat_score"}
    return " + ".join(c if c in numeric else f"C({c})" for c in covs)


def _adjusted_formula() -> str:
    return f"{OUTCOME} ~ {_ses()} + hisb_score + {_cov_formula()}"


def stratified_or(df: pd.DataFrame, subgroup_var: str) -> pd.DataFrame:
    """Adjusted SES/HISB odds ratios fitted separately within each subgroup level."""
    rows = []
    for level, g in df.groupby(subgroup_var, observed=True):
        if g[OUTCOME].nunique() < 2 or len(g) < 30:
            continue
        res = smf.logit(_adjusted_formula(), data=g).fit(disp=False)
        conf = res.conf_int()
        for term, label in _TERMS.items():
            if term not in res.params.index:
                continue
            rows.append(
                {
                    "subgroup": subgroup_var,
                    "level": str(level),
                    "n": int(len(g)),
                    "events": int(g[OUTCOME].sum()),
                    "term": label,
                    "odds_ratio": round(float(np.exp(res.params[term])), 4),
                    "or_ci_low": round(float(np.exp(conf.loc[term, 0])), 4),
                    "or_ci_high": round(float(np.exp(conf.loc[term, 1])), 4),
                    "p_value": round(float(res.pvalues[term]), 4),
                }
            )
    return pd.DataFrame(rows)


def interaction_test(df: pd.DataFrame, subgroup_var: str) -> dict:
    """LR test for an SES x subgroup interaction (effect modification)."""
    base = smf.logit(
        f"{_adjusted_formula()} + C({subgroup_var})", data=df
    ).fit(disp=False)
    full = smf.logit(
        f"{OUTCOME} ~ {_ses()}*C({subgroup_var}) + hisb_score + {_cov_formula()}",
        data=df,
    ).fit(disp=False)
    stat = 2 * (full.llf - base.llf)
    dof = int(full.df_model - base.df_model)
    p = float(stats.chi2.sf(stat, dof)) if dof > 0 else float("nan")
    return {
        "subgroup": subgroup_var,
        "lr_chi2": round(float(stat), 3),
        "df": dof,
        "p_value": round(p, 4),
        "effect_modification_0.05": bool(p < 0.05),
    }


def fig_forest(or_table: pd.DataFrame) -> None:
    """Forest plot of the SES (High vs Low) OR across all subgroup strata."""
    sub = or_table[or_table["term"] == "SES: High (vs Low)"].copy()
    if sub.empty:
        return
    sub["label"] = sub["subgroup"] + " = " + sub["level"]
    sub = sub.iloc[::-1]
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(sub))))
    ax.errorbar(
        sub["odds_ratio"], y,
        xerr=[
            sub["odds_ratio"] - sub["or_ci_low"],
            sub["or_ci_high"] - sub["odds_ratio"],
        ],
        fmt="o", color="#2b6cb0", ecolor="#90cdf4", elinewidth=3, capsize=4,
    )
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("Adjusted OR for SES: High vs Low (95% CI)")
    ax.set_title("Self-medication: SES effect by subgroup")
    path = PATHS.figures_dir / "fig_subgroup_forest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[subgroup] wrote {path.name}")


def run(df: pd.DataFrame) -> dict:
    """Run stratified analyses + interaction tests for all configured subgroups."""
    PATHS.ensure_dirs()
    variables = CONFIG.get("subgroup", {}).get("variables", [])
    or_tables, inter_rows = [], []
    for var in variables:
        if var not in df.columns:
            print(f"[subgroup] skipping '{var}' (not in data)")
            continue
        or_tables.append(stratified_or(df, var))
        inter_rows.append(interaction_test(df, var))

    or_table = pd.concat(or_tables, ignore_index=True) if or_tables else pd.DataFrame()
    inter = pd.DataFrame(inter_rows)

    or_table.to_csv(PATHS.tables_dir / "subgroup_or.csv", index=False)
    inter.to_csv(PATHS.tables_dir / "subgroup_interaction.csv", index=False)
    print(f"[subgroup] wrote subgroup_or.csv ({len(or_table)} rows)")
    print("[subgroup] wrote subgroup_interaction.csv")
    if not or_table.empty:
        fig_forest(or_table)
    return {"or_table": or_table, "interaction": inter}


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return run(df)


if __name__ == "__main__":
    main()
