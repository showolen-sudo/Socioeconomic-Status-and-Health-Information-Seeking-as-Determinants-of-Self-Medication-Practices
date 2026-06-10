"""Figure generation for the study.

All figures are written as PNGs to ``results/figures``. Uses a non-interactive
matplotlib backend so it runs headless (CI, servers). Per-outcome figures are suffixed
with the outcome name (e.g. ``fig_forest_odds_ratios__otc_use.png``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import model_spec  # noqa: E402
from .config import PATHS  # noqa: E402
from .statistical_models import odds_ratio_table  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, name: str) -> None:
    path = PATHS.figures_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {path.name}")


def fig_outcome_rate_by_ses(df: pd.DataFrame, outcome: str, label: str) -> None:
    rate = (
        df.groupby("ses_tertile", observed=True)[outcome].mean().mul(100).reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=rate, x="ses_tertile", y=outcome, ax=ax, hue="ses_tertile",
                palette="viridis", legend=False)
    ax.set_xlabel("Socioeconomic status (tertile)")
    ax.set_ylabel(f"{label} (%)")
    ax.set_title(f"{label} prevalence by SES")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", padding=3)
    _save(fig, f"fig_rate_by_ses__{outcome}.png")


def fig_hisb_distribution(df: pd.DataFrame, outcome: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df, x="hisb_score", hue=outcome, multiple="stack",
        bins=30, palette="Set2", ax=ax,
    )
    ax.set_xlabel("HISB composite score (standardized)")
    ax.set_ylabel("Respondents")
    ax.set_title(f"HISB distribution by {label.lower()} status")
    _save(fig, f"fig_hisb_distribution__{outcome}.png")


def fig_count_distribution(df: pd.DataFrame) -> None:
    """Side-by-side distribution of the daily OTC and herbal counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, title in zip(
        axes, ["NumOTC", "NumHerbal"], ["OTC drugs", "Herbal supplements"], strict=True
    ):
        vals = df[col].clip(upper=5)
        sns.histplot(vals, bins=range(0, 7), ax=ax, color="#2b6cb0")
        ax.set_xlabel(f"Number of {title} taken daily")
        ax.set_title(title)
    axes[0].set_ylabel("Respondents")
    fig.suptitle("Daily self-medication counts")
    _save(fig, "fig_count_distribution.png")


def _pretty_term(term: str) -> str:
    mapping = {
        "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
        "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
        "hisb_score": "Health info-seeking (per unit)",
    }
    return mapping.get(term, term)


def fig_forest_odds_ratios(adjusted_model, outcome: str, label: str) -> None:
    """Forest plot of adjusted odds ratios (excluding the intercept)."""
    table = odds_ratio_table(adjusted_model, "model_adjusted")
    table = table[table["term"] != "Intercept"].copy()
    table["term"] = table["term"].map(_pretty_term)
    table = table.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.7 * len(table))))
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
    ax.set_title(f"Predictors of {label.lower()}")
    _save(fig, f"fig_forest_odds_ratios__{outcome}.png")


def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = ["ses_score", "hisb_score", "self_treat_score", "info_source_count",
            "NumOTC", "NumHerbal"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    _save(fig, "fig_correlation_heatmap.png")


def run(df: pd.DataFrame, models: dict | None = None) -> None:
    """Render all figures. ``models`` maps outcome name -> {model_name: result}."""
    PATHS.ensure_dirs()
    fig_count_distribution(df)
    fig_correlation_heatmap(df)
    for oc in model_spec.outcomes():
        outcome, label = oc["name"], oc["label"]
        fig_outcome_rate_by_ses(df, outcome, label)
        fig_hisb_distribution(df, outcome, label)
        if models is not None and outcome in models:
            fig_forest_odds_ratios(models[outcome]["model_adjusted"], outcome, label)


def main() -> None:
    from .statistical_models import build_models

    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    models = {oc["name"]: build_models(df, oc["name"]) for oc in model_spec.outcomes()}
    run(df, models=models)


if __name__ == "__main__":
    main()
