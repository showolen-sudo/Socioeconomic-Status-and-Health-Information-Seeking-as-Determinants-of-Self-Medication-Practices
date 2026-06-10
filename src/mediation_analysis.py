"""Mediation analysis: SES -> health information-seeking -> self-medication.

Tests whether health information-seeking behaviour (HISB) *mediates* the effect of
socioeconomic status (SES) on each self-medication outcome (OTC use, herbal use), using
the product-of-coefficients approach with a nonparametric bootstrap for inference
(Preacher & Hayes).

Model setup (X = SES composite, M = HISB score, Y = outcome):

* Mediator model (OLS):      M ~ a*X (+ covariates)            -> path a
* Outcome model (logistic):  Y ~ c'*X + b*M (+ covariates)     -> paths c' (direct), b
* Indirect (mediated) effect = a * b   (log-odds scale)
* Total effect (approx.)      = c' + a*b
* Proportion mediated         = (a*b) / (c' + a*b)

Because Y is binary (logistic), effects involving Y are on the log-odds scale and the
total = direct + indirect decomposition is approximate. When direct and indirect effects
have opposite signs ("competitive" mediation), the proportion mediated is reported for
completeness only.

Outputs are suffixed with the outcome name, e.g. ``mediation_results__otc_use.csv``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from . import model_spec  # noqa: E402
from .config import CONFIG, PATHS  # noqa: E402

X_VAR = "ses_score"
M_VAR = "hisb_score"

_EFFECT_LABELS = {
    "a": "a: SES -> HISB",
    "b": "b: HISB -> outcome (logit)",
    "c_prime": "c': direct (SES -> outcome)",
    "indirect": "a*b: indirect (via HISB)",
    "total": "c: total effect",
    "prop_mediated": "proportion mediated",
}


def _cov_suffix() -> str:
    terms = model_spec.covariate_terms()
    return (" + " + " + ".join(terms)) if terms else ""


def fit_once(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    """Fit the mediator and outcome models once; return path/effect estimates."""
    cov = _cov_suffix()
    mediator = smf.ols(f"{M_VAR} ~ {X_VAR}{cov}", data=df).fit()
    outcome_model = smf.logit(
        f"{outcome} ~ {X_VAR} + {M_VAR}{cov}", data=df
    ).fit(disp=False)

    a = float(mediator.params[X_VAR])
    b = float(outcome_model.params[M_VAR])
    c_prime = float(outcome_model.params[X_VAR])
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


def bootstrap(df: pd.DataFrame, outcome: str, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Nonparametric (case-resampling) bootstrap of the mediation estimates."""
    rng = np.random.default_rng(seed)
    n = len(df)
    keys = list(_EFFECT_LABELS)
    store: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = df.iloc[rng.integers(0, n, n)]
        try:
            est = fit_once(sample, outcome)
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


def fig_effects(point: dict[str, float], boot: dict[str, np.ndarray], outcome: str, label: str) -> None:
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
    ax.set_ylabel(f"Effect on {label} (log-odds)")
    ax.set_title(f"Decomposition of the SES effect on {label} (mediated by HISB)")
    for i, e in enumerate(est):
        ax.annotate(f"{e:.3f}", (i, e), ha="center",
                    va="bottom" if e >= 0 else "top", fontsize=11)
    path = PATHS.figures_dir / f"fig_mediation_effects__{outcome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[mediation] wrote {path.name}")


def run(df: pd.DataFrame, n_boot: int | None = None, seed: int | None = None) -> dict:
    """Run the full mediation analysis for each outcome and persist outputs."""
    PATHS.ensure_dirs()
    n_boot = int(n_boot if n_boot is not None else CONFIG.get("mediation", {}).get("n_boot", 1000))
    seed = int(seed if seed is not None else CONFIG["seed"])

    tables = {}
    for oc in model_spec.outcomes():
        outcome, label = oc["name"], oc["label"]
        point = fit_once(df, outcome)
        boot = bootstrap(df, outcome, n_boot=n_boot, seed=seed)
        table = results_table(point, boot)
        table.insert(0, "outcome", outcome)
        table.to_csv(PATHS.tables_dir / f"mediation_results__{outcome}.csv", index=False)
        print(f"[mediation] wrote mediation_results__{outcome}.csv (bootstrap B={n_boot})")
        fig_effects(point, boot, outcome, label)
        tables[outcome] = table
    return tables


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    return run(df)


if __name__ == "__main__":
    main()
