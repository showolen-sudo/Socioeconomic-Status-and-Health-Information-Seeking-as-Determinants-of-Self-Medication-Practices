"""Figure generation for the study.

All figures are written as PNGs to ``results/figures``. Uses a non-interactive
matplotlib backend so it runs headless (CI, servers).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import PATHS  # noqa: E402
from .statistical_models import odds_ratio_table  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
OUTCOME = "self_medication"


def _save(fig: plt.Figure, name: str) -> None:
    path = PATHS.figures_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {path.name}")


def fig_outcome_rate_by_ses(df: pd.DataFrame) -> None:
    rate = (
        df.groupby("ses_tertile", observed=True)[OUTCOME].mean().mul(100).reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=rate, x="ses_tertile", y=OUTCOME, ax=ax, hue="ses_tertile",
                palette="viridis", legend=False)
    ax.set_xlabel("Socioeconomic status (tertile)")
    ax.set_ylabel("Self-medication (%)")
    ax.set_title("Self-medication prevalence by SES")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3)
    _save(fig, "fig_selfmed_by_ses.png")


def fig_hisb_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df, x="hisb_score", hue=OUTCOME, multiple="stack",
        bins=range(0, 22), palette="Set2", ax=ax,
    )
    ax.set_xlabel("Health information-seeking score (0-20)")
    ax.set_ylabel("Respondents")
    ax.set_title("HISB distribution by self-medication status")
    _save(fig, "fig_hisb_distribution.png")


def fig_ses_score_by_outcome(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=OUTCOME, y="ses_score", ax=ax, hue=OUTCOME,
                palette="pastel", legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_xlabel("Self-medication")
    ax.set_ylabel("SES composite (z)")
    ax.set_title("SES score by self-medication status")
    _save(fig, "fig_ses_score_by_outcome.png")


def _pretty_term(term: str) -> str:
    """Map verbose patsy term names to readable labels for plotting."""
    mapping = {
        "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
        "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
        "self_treat_score": "Self-treat agreement (per point)",
        "hisb_score": "Health info-seeking (per point)",
    }
    return mapping.get(term, term)


def fig_forest_odds_ratios(adjusted_model) -> None:
    """Forest plot of adjusted odds ratios (excluding the intercept)."""
    table = odds_ratio_table(adjusted_model, "model_adjusted")
    table = table[table["term"] != "Intercept"].copy()
    table["term"] = table["term"].map(_pretty_term)
    table = table.iloc[::-1]  # nicer top-down ordering

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(table))))
    y = np.arange(len(table))
    ax.errorbar(
        table["odds_ratio"], y,
        xerr=[
            table["odds_ratio"] - table["or_ci_low"],
            table["or_ci_high"] - table["odds_ratio"],
        ],
        fmt="o", color="#2b6cb0", ecolor="#90cdf4", elinewidth=3, capsize=4,
    )
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["term"])
    ax.set_xlabel("Adjusted odds ratio (95% CI)")
    ax.set_title("Predictors of self-medication")
    _save(fig, "fig_forest_odds_ratios.png")


def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["income_monthly", "hisb_score", "ses_score", OUTCOME]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, "fig_correlation_heatmap.png")


def run(df: pd.DataFrame, adjusted_model=None) -> None:
    """Render all figures."""
    PATHS.ensure_dirs()
    fig_outcome_rate_by_ses(df)
    fig_hisb_distribution(df)
    fig_ses_score_by_outcome(df)
    fig_correlation_heatmap(df)
    if adjusted_model is not None:
        fig_forest_odds_ratios(adjusted_model)


def main() -> None:
    from .statistical_models import build_models

    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    models = build_models(df)
    run(df, adjusted_model=models["model_adjusted"])


if __name__ == "__main__":
    main()
