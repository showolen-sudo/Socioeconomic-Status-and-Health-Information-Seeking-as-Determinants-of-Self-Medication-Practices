"""Discrimination and calibration metrics for the adjusted self-medication models.

Quantifies how well the adjusted logistic model (outcome ~ SES + HISB) predicts each
self-medication outcome. To avoid optimistic (in-sample) bias, predicted probabilities
are also generated out-of-sample via stratified k-fold cross-validation.

Reported per outcome:
* **Discrimination:** ROC-AUC (apparent and cross-validated).
* **Overall accuracy:** Brier score and log loss.
* **Calibration:** reliability curve + Hosmer-Lemeshow goodness-of-fit test.

Outputs are suffixed with the outcome name, e.g. ``discrimination_metrics__otc_use.csv``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from . import model_spec  # noqa: E402
from .config import CONFIG, PATHS  # noqa: E402


def _design(df: pd.DataFrame, outcome: str):
    formula = model_spec.adjusted_rhs()
    X = patsy.dmatrix(formula, df, return_type="dataframe").drop(columns=["Intercept"])
    y = df[outcome].astype(int).to_numpy()
    return X.to_numpy(), y


def _predictions(df: pd.DataFrame, outcome: str, n_splits: int, seed: int):
    X, y = _design(df, outcome)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    p_apparent = model.predict_proba(X)[:, 1]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return y, p_apparent, p_cv


def discrimination_table(y, p_apparent, p_cv) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "ROC_AUC",
                "apparent": round(roc_auc_score(y, p_apparent), 4),
                "cross_validated": round(roc_auc_score(y, p_cv), 4),
            },
            {
                "metric": "Brier_score",
                "apparent": round(brier_score_loss(y, p_apparent), 4),
                "cross_validated": round(brier_score_loss(y, p_cv), 4),
            },
            {
                "metric": "log_loss",
                "apparent": round(log_loss(y, p_apparent), 4),
                "cross_validated": round(log_loss(y, p_cv), 4),
            },
        ]
    )


def hosmer_lemeshow(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Hosmer-Lemeshow goodness-of-fit test using quantile bins of predicted risk."""
    order = np.argsort(p)
    y_sorted, p_sorted = np.asarray(y)[order], np.asarray(p)[order]
    groups = np.array_split(np.arange(len(p)), n_bins)
    stat = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        obs = y_sorted[g].sum()
        exp = p_sorted[g].sum()
        n = len(g)
        denom = exp * (1 - exp / n)
        if denom > 0:
            stat += (obs - exp) ** 2 / denom
    dof = max(n_bins - 2, 1)
    p_value = float(stats.chi2.sf(stat, dof))
    return pd.DataFrame(
        [
            {
                "test": "Hosmer-Lemeshow",
                "chi2": round(float(stat), 3),
                "df": dof,
                "p_value": round(p_value, 4),
                "well_calibrated_0.05": bool(p_value >= 0.05),
            }
        ]
    )


def fig_roc(y, p_cv, outcome: str, label: str) -> None:
    fpr, tpr, _ = roc_curve(y, p_cv)
    auc = roc_auc_score(y, p_cv)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"CV ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve - {label} (cross-validated)")
    ax.legend(loc="lower right")
    path = PATHS.figures_dir / f"fig_roc_curve__{outcome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] wrote {path.name}")


def fig_calibration(prob_true, prob_pred, outcome: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", color="#dd6b20", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration curve - {label} (cross-validated)")
    ax.legend(loc="upper left")
    path = PATHS.figures_dir / f"fig_calibration_curve__{outcome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] wrote {path.name}")


def run_for_outcome(df: pd.DataFrame, outcome: str, label: str, n_splits: int, n_bins: int, seed: int) -> dict:
    y, p_apparent, p_cv = _predictions(df, outcome, n_splits=n_splits, seed=seed)

    disc = discrimination_table(y, p_apparent, p_cv)
    disc.insert(0, "outcome", outcome)
    disc.to_csv(PATHS.tables_dir / f"discrimination_metrics__{outcome}.csv", index=False)

    hl = hosmer_lemeshow(y, p_cv, n_bins=n_bins)
    hl.insert(0, "outcome", outcome)
    hl.to_csv(PATHS.tables_dir / f"hosmer_lemeshow__{outcome}.csv", index=False)

    prob_true, prob_pred = calibration_curve(y, p_cv, n_bins=n_bins, strategy="quantile")
    pd.DataFrame(
        {"prob_pred": prob_pred.round(4), "prob_true": prob_true.round(4)}
    ).to_csv(PATHS.tables_dir / f"calibration_curve__{outcome}.csv", index=False)
    print(f"[calibration] wrote metrics for {outcome}")

    fig_roc(y, p_cv, outcome, label)
    fig_calibration(prob_true, prob_pred, outcome, label)
    return {"discrimination": disc, "hosmer_lemeshow": hl}


def run(df: pd.DataFrame) -> dict:
    """Compute discrimination + calibration metrics for each outcome."""
    PATHS.ensure_dirs()
    cfg = CONFIG.get("calibration", {})
    n_splits = int(cfg.get("n_splits", 5))
    n_bins = int(cfg.get("n_calibration_bins", 10))
    seed = int(CONFIG["seed"])

    results = {}
    for oc in model_spec.outcomes():
        results[oc["name"]] = run_for_outcome(
            df, oc["name"], oc["label"], n_splits, n_bins, seed
        )
    return results


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return run(df)


if __name__ == "__main__":
    main()
