"""
Step 2 - Descriptive statistics and unadjusted prevalence tables.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def continuous_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD, median, min, max for key continuous variables."""
    vars_ = (
        "ses_score",
        "hisb_score",
        "self_treat_score",
        "info_source_count",
        "NumOTC",
        "NumHerbal",
    )
    rows = []
    for var in vars_:
        s = df[var].astype(float)
        rows.append(
            {
                "variable": var,
                "n": int(s.notna().sum()),
                "mean": round(float(s.mean()), 3),
                "sd": round(float(s.std(ddof=0)), 3),
                "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def prevalence_by_ses(df: pd.DataFrame) -> pd.DataFrame:
    """Percent using OTC/herbal by SES tertile (unadjusted)."""
    otc = (
        df.groupby("ses_tertile", observed=True)["otc_use"]
        .agg(n="count", rate="mean")
        .reset_index()
    )
    herb = (
        df.groupby("ses_tertile", observed=True)["herbal_use"]
        .agg(n="count", herb_rate="mean")
        .reset_index()
    )
    out = otc.merge(herb[["ses_tertile", "herb_rate"]], on="ses_tertile")
    out["otc_use_pct"] = (out["rate"] * 100).round(1)
    out["herbal_use_pct"] = (out["herb_rate"] * 100).round(1)
    return out[["ses_tertile", "n", "otc_use_pct", "herbal_use_pct"]]


def prevalence_by_hisb(df: pd.DataFrame) -> pd.DataFrame:
    """Percent using OTC/herbal above vs below median HISB (unadjusted)."""
    labels = {0: "Below median HISB", 1: "At/above median HISB"}
    otc = (
        df.groupby("hisb_high")["otc_use"].agg(n="count", rate="mean").reset_index()
    )
    herb = (
        df.groupby("hisb_high")["herbal_use"]
        .agg(n="count", herb_rate="mean")
        .reset_index()
    )
    merged = otc.merge(herb[["hisb_high", "herb_rate"]], on="hisb_high")
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "HISB group": labels[int(r["hisb_high"])],
                "N": int(r["n"]),
                "OTC use (%)": round(float(r["rate"]) * 100, 1),
                "Herbal use (%)": round(float(r["herb_rate"]) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def prevalence_by_self_treat(df: pd.DataFrame) -> pd.DataFrame:
    """Percent using OTC/herbal at each Self_Treat Likert level (unadjusted)."""
    rows = []
    scores = sorted(df["self_treat_score"].dropna().unique())
    for code in scores:
        sub = df[df["self_treat_score"] == code]
        if sub.empty:
            continue
        rows.append(
            {
                "Self_Treat score": int(code),
                "N": len(sub),
                "OTC use (%)": round(float(sub["otc_use"].mean()) * 100, 1),
                "Herbal use (%)": round(float(sub["herbal_use"].mean()) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def spearman_self_treat(df: pd.DataFrame, outcomes: list[dict]) -> pd.DataFrame:
    """Rank correlation between Self_Treat score and each binary outcome."""
    rows = []
    for oc in outcomes:
        rho, p = stats.spearmanr(df["self_treat_score"], df[oc["name"]])
        rows.append(
            {
                "outcome": oc["name"],
                "outcome_label": oc["label"],
                "spearman_rho": round(float(rho), 3),
                "p_value": round(float(p), 4),
            }
        )
    return pd.DataFrame(rows)
