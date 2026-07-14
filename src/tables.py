"""
Step 4 - Presentation tables (including the single combined OR table).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def study_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One-page summary of design and sample."""
    n = len(df)
    otc_n = int(df["otc_use"].sum())
    herb_n = int(df["herbal_use"].sum())
    return pd.DataFrame(
        [
            {
                "Item": "Study title",
                "Value": "SES, HISB, and Self-Treatment Attitude as Predictors of Self-Medication",
            },
            {"Item": "Data", "Value": "2021 NCME survey (see docs/codebook.md)"},
            {"Item": "Sample size (N)", "Value": str(n)},
            {
                "Item": "OTC use (any daily)",
                "Value": f"{otc_n} ({100 * df['otc_use'].mean():.1f}%)",
            },
            {
                "Item": "Herbal use (any daily)",
                "Value": f"{herb_n} ({100 * df['herbal_use'].mean():.1f}%)",
            },
            {
                "Item": "HISB definition",
                "Value": "Med7-9 + DTCA items + info sources (Self_Treat excluded)",
            },
            {
                "Item": "Self_Treat",
                "Value": "Separate predictor; Likert (survey 1-7), z-scored in models",
            },
            {
                "Item": "Primary model",
                "Value": "logit(outcome ~ SES tertile + HISB + Self_Treat_z)",
            },
            {"Item": "SES reference", "Value": "Low tertile (OR = 1.00)"},
            {"Item": "Significance", "Value": "alpha = 0.05 (two-sided)"},
            {
                "Item": "Generated (UTC)",
                "Value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            },
        ]
    )


def adjusted_or_long(model_tables: dict, outcomes: list[dict]) -> pd.DataFrame:
    """Long-format adjusted OR table with 95% CIs."""
    labels = {
        "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
        "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
        "hisb_score": "Health information-seeking (per 1 SD)",
        "self_treat_z": "Self-treatment attitude (per 1 SD)",
    }
    rows = []
    for oc in outcomes:
        raw = model_tables.get(f"adjusted__{oc['name']}")
        if raw is None:
            continue
        for _, r in raw.iterrows():
            term = r["term"]
            if term == "Intercept":
                continue
            rows.append(
                {
                    "Outcome": oc["label"],
                    "Predictor": labels.get(term, term),
                    "Adjusted OR": round(float(r["odds_ratio"]), 3),
                    "95% CI lower": round(float(r["or_ci_low"]), 3),
                    "95% CI upper": round(float(r["or_ci_high"]), 3),
                    "p-value": format_p(float(r["p_value"])),
                }
            )
    return pd.DataFrame(rows)


def combined_predictor_table(model_tables: dict) -> pd.DataFrame:
    """
    Single summary table for professor review:
    all predictors with OR and p-value for OTC and Herbal outcomes.
    """
    otc = model_tables["adjusted__otc_use"].set_index("term")
    herb = model_tables["adjusted__herbal_use"].set_index("term")
    rows = [
        {
            "Predictor": "SES: Low",
            "OTC OR": 1.0,
            "OTC p-value": "Reference",
            "Herbal OR": 1.0,
            "Herbal p-value": "Reference",
        }
    ]
    for term, label in (
        ("C(ses_tertile, Treatment(reference='Low'))[T.Middle]", "SES: Middle"),
        ("C(ses_tertile, Treatment(reference='Low'))[T.High]", "SES: High"),
    ):
        o, h = otc.loc[term], herb.loc[term]
        rows.append(
            {
                "Predictor": label,
                "OTC OR": round(float(o["odds_ratio"]), 3),
                "OTC p-value": format_p(float(o["p_value"])),
                "Herbal OR": round(float(h["odds_ratio"]), 3),
                "Herbal p-value": format_p(float(h["p_value"])),
            }
        )
    for term, label in (
        ("hisb_score", "Health information-seeking (per 1 SD)"),
        ("self_treat_z", "Self-treatment attitude (per 1 SD)"),
    ):
        o, h = otc.loc[term], herb.loc[term]
        rows.append(
            {
                "Predictor": label,
                "OTC OR": round(float(o["odds_ratio"]), 3),
                "OTC p-value": format_p(float(o["p_value"])),
                "Herbal OR": round(float(h["odds_ratio"]), 3),
                "Herbal p-value": format_p(float(h["p_value"])),
            }
        )
    return pd.DataFrame(rows)


def self_treat_effect(model_tables: dict, outcomes: list[dict]) -> pd.DataFrame:
    """Crude vs adjusted Self_Treat OR for each outcome."""
    rows = []
    for oc in outcomes:
        for key, label in (
            ("crude_self_treat_z", "Crude (Self_Treat only)"),
            ("adjusted", "Adjusted (SES + HISB + Self_Treat)"),
        ):
            raw = model_tables[f"{key}__{oc['name']}"]
            st = raw[raw["term"] == "self_treat_z"]
            if st.empty:
                continue
            r = st.iloc[0]
            rows.append(
                {
                    "Outcome": oc["label"],
                    "Model": label,
                    "Self_Treat OR (per 1 SD)": round(float(r["odds_ratio"]), 3),
                    "95% CI": f"{r['or_ci_low']:.2f} - {r['or_ci_high']:.2f}",
                    "p-value": format_p(float(r["p_value"])),
                }
            )
    return pd.DataFrame(rows)
