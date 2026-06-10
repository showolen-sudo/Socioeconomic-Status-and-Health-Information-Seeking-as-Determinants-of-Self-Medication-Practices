"""Descriptive and bivariate analyses.

Generates:
* ``descriptives_continuous.csv`` - summary stats for continuous variables.
* ``descriptives_categorical.csv`` - counts/percentages for categorical variables.
* ``bivariate_chi2.csv`` - chi-square tests of each categorical predictor vs. outcome.
* ``self_medication_by_ses.csv`` / ``self_medication_by_hisb.csv`` - key cross-tabs.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .config import PATHS

CONTINUOUS_VARS = ["income_monthly", "hisb_score", "ses_score"]
CATEGORICAL_VARS = [
    "education",
    "occupation",
    "chronic_condition",
    "internet_access",
    "self_treat",
    "ses_tertile",
]
OUTCOME = "self_medication"


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
    """Chi-square test of association between each categorical predictor and outcome."""
    rows = []
    for var in CATEGORICAL_VARS:
        if var == OUTCOME:
            continue
        table = pd.crosstab(df[var], df[OUTCOME])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "variable": var,
                "chi2": round(chi2, 3),
                "dof": int(dof),
                "p_value": round(p, 4),
                "significant_0.05": p < 0.05,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def outcome_rate_by(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Self-medication prevalence within levels of ``group``."""
    g = df.groupby(group, observed=True)[OUTCOME]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "self_medication_n": g.sum(),
            "self_medication_rate_pct": (100 * g.mean()).round(2),
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
        "self_medication_by_ses": outcome_rate_by(df, "ses_tertile"),
        "self_medication_by_hisb": outcome_rate_by(df, "hisb_high"),
    }
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
