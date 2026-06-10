"""Discrimination and calibration metrics for the adjusted self-medication model.

Quantifies how well the adjusted logistic model (self-medication ~ SES + HISB +
self_treat_score) predicts. To avoid optimistic (in-sample) bias, predicted
probabilities are also generated out-of-sample via stratified k-fold cross-validation.

Reported:
* **Discrimination:** ROC-AUC (apparent and cross-validated).
* **Overall accuracy:** Brier score and log loss.
* **Calibration:** reliability curve + Hosmer-Lemeshow goodness-of-fit test.

Outputs:
* ``discrimination_metrics.csv`` - AUC / Brier / log-loss (apparent vs CV).
* ``calibration_curve.csv``      - binned observed vs. predicted probabilities.
* ``hosmer_lemeshow.csv``        - HL chi-square statistic and p-value.
* ``fig_roc_curve.png`` / ``fig_calibration_curve.png``.
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

from .config import CONFIG, PATHS  # noqa: E402

OUTCOME = "self_medication"


def _design(df: pd.DataFrame):
    ref = CONFIG["modelling"]["ses_reference"]
    formula = (
        f"C(ses_tertile, Treatment(reference='{ref}')) + hisb_score + self_treat_score"
    )
    X = patsy.dmatrix(formula, df, return_type="dataframe").drop(columns=["Intercept"])
    y = df[OUTCOME].astype(int).to_numpy()
    return X.to_numpy(), y


def _predictions(df: pd.DataFrame, n_splits: int, seed: int):
    X, y = _design(df)
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


def fig_roc(y, p_cv) -> None:
    fpr, tpr, _ = roc_curve(y, p_cv)
    auc = roc_auc_score(y, p_cv)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"CV ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (cross-validated)")
    ax.legend(loc="lower right")
    path = PATHS.figures_dir / "fig_roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] wrote {path.name}")


def fig_calibration(prob_true, prob_pred) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", color="#dd6b20", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (cross-validated)")
    ax.legend(loc="upper left")
    path = PATHS.figures_dir / "fig_calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] wrote {path.name}")


def run(df: pd.DataFrame) -> dict:
    """Compute discrimination + calibration metrics and persist tables/figures."""
    PATHS.ensure_dirs()
    cfg = CONFIG.get("calibration", {})
    n_splits = int(cfg.get("n_splits", 5))
    n_bins = int(cfg.get("n_calibration_bins", 10))
    seed = int(CONFIG["seed"])

    y, p_apparent, p_cv = _predictions(df, n_splits=n_splits, seed=seed)

    disc = discrimination_table(y, p_apparent, p_cv)
    disc.to_csv(PATHS.tables_dir / "discrimination_metrics.csv", index=False)
    print("[calibration] wrote discrimination_metrics.csv")

    hl = hosmer_lemeshow(y, p_cv, n_bins=n_bins)
    hl.to_csv(PATHS.tables_dir / "hosmer_lemeshow.csv", index=False)
    print("[calibration] wrote hosmer_lemeshow.csv")

    prob_true, prob_pred = calibration_curve(y, p_cv, n_bins=n_bins, strategy="quantile")
    pd.DataFrame({"prob_pred": prob_pred.round(4), "prob_true": prob_true.round(4)}).to_csv(
        PATHS.tables_dir / "calibration_curve.csv", index=False
    )
    print("[calibration] wrote calibration_curve.csv")

    fig_roc(y, p_cv)
    fig_calibration(prob_true, prob_pred)
    return {"discrimination": disc, "hosmer_lemeshow": hl}


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return run(df)


if __name__ == "__main__":
    main()
