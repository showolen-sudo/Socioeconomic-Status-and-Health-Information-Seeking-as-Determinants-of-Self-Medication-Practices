"""Mediation analysis: SES -> health information-seeking -> self-medication.

Tests whether health information-seeking behaviour (HISB) *mediates* the effect of
socioeconomic status (SES) on self-medication, using the standard product-of-coefficients
approach with a nonparametric bootstrap for inference (Preacher & Hayes).

Model setup (X = SES composite, M = HISB score, Y = self-medication):

* Mediator model (OLS):      M ~ a*X + covariates                 -> path a
* Outcome model (logistic):  Y ~ c'*X + b*M + covariates          -> paths c' (direct), b
* Indirect (mediated) effect = a * b   (log-odds scale)
* Total effect (approx.)      = c' + a*b
* Proportion mediated         = (a*b) / (c' + a*b)

Note: because Y is binary (logistic), effects involving Y are on the log-odds scale and
the total = direct + indirect decomposition is approximate. When the direct and indirect
effects have opposite signs ("competitive"/inconsistent mediation), the proportion
mediated is not straightforwardly interpretable and is reported for completeness only.

Outputs:
* ``mediation_results.csv``   - point estimates with bootstrap 95% CIs.
* ``fig_mediation_effects.png`` - direct vs. indirect vs. total effect with CIs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from .config import CONFIG, PATHS  # noqa: E402

X_VAR = "ses_score"
M_VAR = "hisb_score"
Y_VAR = "self_medication"

_EFFECT_LABELS = {
    "a": "a: SES -> HISB",
    "b": "b: HISB -> self-med (logit)",
    "c_prime": "c': direct (SES -> self-med)",
    "indirect": "a*b: indirect (via HISB)",
    "total": "c: total effect",
    "prop_mediated": "proportion mediated",
}


def _cov_formula() -> str:
    covs = CONFIG["modelling"]["covariates"]
    numeric = {"age", "hisb_score", "ses_score", "income_monthly", "self_treat_score"}
    return " + ".join(c if c in numeric else f"C({c})" for c in covs)


def fit_once(df: pd.DataFrame) -> dict[str, float]:
    """Fit the mediator and outcome models once; return path/effect estimates."""
    cov = _cov_formula()
    mediator = smf.ols(f"{M_VAR} ~ {X_VAR} + {cov}", data=df).fit()
    outcome = smf.logit(f"{Y_VAR} ~ {X_VAR} + {M_VAR} + {cov}", data=df).fit(disp=False)

    a = float(mediator.params[X_VAR])
    b = float(outcome.params[M_VAR])
    c_prime = float(outcome.params[X_VAR])
    indirect = a * b
    total = c_prime + indirect
    prop = indirect / total if total != 0 else np.nan
    return {
        "a": a,
        "b": b,
        "c_prime": c_prime,
        "indirect": indirect,
        "total": total,
        "prop_mediated": prop,
    }


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Nonparametric (case-resampling) bootstrap of the mediation estimates."""
    rng = np.random.default_rng(seed)
    n = len(df)
    keys = list(_EFFECT_LABELS)
    store: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, n)]
        try:
            est = fit_once(sample)
        except Exception:  # pragma: no cover - rare singular bootstrap draw
            continue
        for k in keys:
            store[k].append(est[k])
    return {k: np.asarray(v) for k, v in store.items()}


def results_table(point: dict[str, float], boot: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, label in _EFFECT_LABELS.items():
        arr = boot[key]
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "effect": label,
                "estimate": round(point[key], 4),
                "ci_low": round(float(np.percentile(arr, 2.5)), 4),
                "ci_high": round(float(np.percentile(arr, 97.5)), 4),
                "boot_n": int(arr.size),
            }
        )
    return pd.DataFrame(rows)


def fig_effects(point: dict[str, float], boot: dict[str, np.ndarray]) -> None:
    keys = ["c_prime", "indirect", "total"]
    labels = ["Direct (c')", "Indirect (a*b)", "Total (c)"]
    est = [point[k] for k in keys]
    lo = [np.percentile(boot[k], 2.5) for k in keys]
    hi = [np.percentile(boot[k], 97.5) for k in keys]
    err_low = [e - lo_i for e, lo_i in zip(est, lo, strict=True)]
    err_high = [hi_i - e for e, hi_i in zip(est, hi, strict=True)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6cb0", "#dd6b20", "#2f855a"]
    ax.bar(labels, est, yerr=[err_low, err_high], capsize=6, color=colors, alpha=0.85)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_ylabel("Effect on self-medication (log-odds)")
    ax.set_title("Decomposition of the SES effect (mediated by HISB)")
    for i, e in enumerate(est):
        ax.annotate(f"{e:.3f}", (i, e), ha="center",
                    va="bottom" if e >= 0 else "top", fontsize=11)
    path = PATHS.figures_dir / "fig_mediation_effects.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mediation] wrote {path.name}")


def run(df: pd.DataFrame, n_boot: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Run the full mediation analysis and persist outputs."""
    PATHS.ensure_dirs()
    n_boot = int(n_boot if n_boot is not None else CONFIG.get("mediation", {}).get("n_boot", 1000))
    seed = int(seed if seed is not None else CONFIG["seed"])

    point = fit_once(df)
    boot = bootstrap(df, n_boot=n_boot, seed=seed)
    table = results_table(point, boot)

    table.to_csv(PATHS.tables_dir / "mediation_results.csv", index=False)
    print(f"[mediation] wrote mediation_results.csv (bootstrap B={n_boot})")
    fig_effects(point, boot)
    return table


def main() -> pd.DataFrame:
    df = pd.read_csv(PATHS.processed_data)
    return run(df)


if __name__ == "__main__":
    main()
