"""Descriptive and bivariate analyses.

Generates:
* ``descriptives_continuous.csv``  - summary stats for continuous variables.
* ``descriptives_categorical.csv`` - counts/percentages for categorical variables.
* ``bivariate_chi2.csv``           - chi-square tests of each categorical predictor vs.
  each self-medication outcome (OTC use, herbal use).
* ``<outcome>_by_ses.csv`` / ``<outcome>_by_hisb.csv`` - key cross-tabs per outcome.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from . import model_spec
from .config import PATHS

CONTINUOUS_VARS = [
    "ses_score",
    "hisb_score",
    "self_treat_score",
    "info_source_count",
    "NumOTC",
    "NumHerbal",
]
CATEGORICAL_VARS = [
    "Education",
    "HouseIncome",
    "DTCA_Info",
    "DTCA_Prescribe",
    "Self_Treat",
    "ses_tertile",
]


def continuous_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in CONTINUOUS_VARS:
        s = df[var].astype(float)
        rows.append(
            {
                "variable": var,
                "n": int(s.notna().sum()),
                "mean": round(s.mean(), 3),
                "sd": round(s.std(), 3),
                "median": round(s.median(), 3),
                "min": round(s.min(), 3),
                "max": round(s.max(), 3),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in CATEGORICAL_VARS:
        counts = df[var].value_counts(dropna=False)
        total = counts.sum()
        for level, count in counts.items():
            rows.append(
                {
                    "variable": var,
                    "level": str(level),
                    "count": int(count),
                    "percent": round(100 * count / total, 2),
                }
            )
    return pd.DataFrame(rows)


def bivariate_chi2(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test of each categorical predictor against each outcome."""
    rows = []
    for oc in model_spec.outcomes():
        outcome = oc["name"]
        for var in CATEGORICAL_VARS:
            table = pd.crosstab(df[var], df[outcome])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(table)
            rows.append(
                {
                    "outcome": outcome,
                    "variable": var,
                    "chi2": round(chi2, 3),
                    "dof": int(dof),
                    "p_value": round(p, 4),
                    "significant_0.05": p < 0.05,
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "p_value"]).reset_index(drop=True)


def outcome_rate_by(df: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    """Prevalence of ``outcome`` within levels of ``group``."""
    g = df.groupby(group, observed=True)[outcome]
    out = pd.DataFrame(
        {
            "n": g.size(),
            f"{outcome}_n": g.sum(),
            f"{outcome}_rate_pct": (100 * g.mean()).round(2),
        }
    ).reset_index()
    return out


def run(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute all descriptive tables and persist them to ``results/tables``."""
    PATHS.ensure_dirs()
    outputs = {
        "descriptives_continuous": continuous_summary(df),
        "descriptives_categorical": categorical_summary(df),
        "bivariate_chi2": bivariate_chi2(df),
    }
    for oc in model_spec.outcomes():
        outcome = oc["name"]
        outputs[f"{outcome}_by_ses"] = outcome_rate_by(df, outcome, "ses_tertile")
        outputs[f"{outcome}_by_hisb"] = outcome_rate_by(df, outcome, "hisb_high")

    for name, table in outputs.items():
        path = PATHS.tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"[descriptive] wrote {path.name} ({len(table)} rows)")
    return outputs


def main() -> dict[str, pd.DataFrame]:
    df = pd.read_csv(PATHS.processed_data)
    return run(df)


if __name__ == "__main__":
    main()
