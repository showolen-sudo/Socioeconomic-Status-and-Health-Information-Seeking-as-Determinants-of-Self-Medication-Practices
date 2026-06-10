"""Multiple imputation (MICE) for missing predictor data, with Rubin's-rules pooling.

Compares two strategies for fitting the adjusted self-medication model when predictors
have missing values:

* **Complete-case analysis** - listwise deletion (rows with any missing value dropped).
* **Multiple imputation** - statsmodels MICE generates several imputed datasets, fits
  the logistic model in each, and pools the estimates via Rubin's rules.

On the synthetic data (which is complete), missingness is injected MCAR into the
configured predictor columns so the demonstration is meaningful. On real data with
genuine missingness, set ``mi.demo_missing: false`` in the config to impute the observed
gaps directly. The demonstration runs for each outcome (OTC use, herbal use).

Outputs:
* ``mi_pooled__<outcome>.csv``          - MI-pooled odds ratios (with fraction missing info).
* ``mi_vs_completecase__<outcome>.csv`` - side-by-side OR comparison of both strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.imputation import mice

from . import model_spec
from .config import CONFIG, PATHS

SES_CODE = {"Low": 0, "Middle": 1, "High": 2}
_TERM_LABELS = {
    "C(ses_code, Treatment(reference=0))[T.1]": "SES: Middle (vs Low)",
    "C(ses_code, Treatment(reference=0))[T.2]": "SES: High (vs Low)",
    "hisb_score": "Health info-seeking (per unit)",
    "Intercept": "Intercept",
}


def _formula(outcome: str) -> str:
    return f"{outcome} ~ C(ses_code, Treatment(reference=0)) + hisb_score"


def _model_frame(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Numeric-only frame MICE can operate on."""
    return pd.DataFrame(
        {
            outcome: df[outcome].astype(int).to_numpy(),
            "ses_code": df["ses_tertile"].astype(str).map(SES_CODE).to_numpy(),
            "hisb_score": df["hisb_score"].astype(float).to_numpy(),
        }
    )


def inject_missing(df: pd.DataFrame, rate: float, cols: list, seed: int) -> pd.DataFrame:
    """Set a random MCAR fraction of values to NaN in the given columns."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    for col in cols:
        if col not in out.columns:
            continue
        mask = rng.random(n) < rate
        out.loc[mask, col] = np.nan
    return out


def _or_table(names, params, bse, source: str, fmi=None) -> pd.DataFrame:
    params = np.asarray(params, dtype=float)
    bse = np.asarray(bse, dtype=float)
    fmi = np.asarray(fmi, dtype=float) if fmi is not None else None
    rows = []
    for i, term in enumerate(names):
        coef, se = float(params[i]), float(bse[i])
        z = coef / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        row = {
            "source": source,
            "term": _TERM_LABELS.get(term, term),
            "odds_ratio": round(float(np.exp(coef)), 4),
            "or_ci_low": round(float(np.exp(coef - 1.96 * se)), 4),
            "or_ci_high": round(float(np.exp(coef + 1.96 * se)), 4),
            "p_value": round(p, 4) if np.isfinite(p) else np.nan,
        }
        if fmi is not None:
            row["fraction_missing_info"] = round(float(fmi[i]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def complete_case(df_missing: pd.DataFrame, outcome: str):
    """Fit the listwise-deletion GLM; return (or_table, parameter_names)."""
    res = smf.glm(_formula(outcome), data=df_missing, family=sm.families.Binomial()).fit()
    names = list(res.params.index)
    table = _or_table(names, res.params.to_numpy(), res.bse.to_numpy(), "complete_case")
    return table, names


def multiple_imputation(df_missing: pd.DataFrame, outcome: str, n_imp: int, n_burnin: int, names=None) -> pd.DataFrame:
    imp = mice.MICEData(df_missing)
    mi = mice.MICE(_formula(outcome), sm.GLM, imp, init_kwds={"family": sm.families.Binomial()})
    result = mi.fit(n_burnin, n_imp)
    if names is None:
        ref = smf.glm(_formula(outcome), data=df_missing.dropna(), family=sm.families.Binomial()).fit()
        names = list(ref.params.index)
    fmi = getattr(result, "frac_miss_info", None)
    return _or_table(names, result.params, result.bse, "multiple_imputation", fmi=fmi)


def run(df: pd.DataFrame) -> dict:
    """Run the MI demonstration for each outcome and persist pooled + comparison tables."""
    PATHS.ensure_dirs()
    cfg = CONFIG.get("mi", {})
    seed = int(CONFIG["seed"])
    results = {}

    for oc in model_spec.outcomes():
        outcome = oc["name"]
        frame = _model_frame(df, outcome)

        if cfg.get("demo_missing", True):
            frame_missing = inject_missing(
                frame,
                rate=float(cfg.get("demo_missing_rate", 0.15)),
                cols=cfg.get("demo_missing_cols", ["hisb_score"]),
                seed=seed,
            )
        else:
            frame_missing = frame

        miss_summary = frame_missing.isna().mean().round(4)
        print(f"[mi] {outcome} missing fractions: {miss_summary.to_dict()}")

        cc, names = complete_case(frame_missing, outcome)
        mi_pooled = multiple_imputation(
            frame_missing,
            outcome,
            n_imp=int(cfg.get("n_imputations", 20)),
            n_burnin=int(cfg.get("n_burnin", 10)),
            names=names,
        )

        mi_pooled.insert(0, "outcome", outcome)
        mi_pooled.to_csv(PATHS.tables_dir / f"mi_pooled__{outcome}.csv", index=False)

        comparison = pd.concat([cc, mi_pooled[cc.columns]], ignore_index=True)
        comparison.insert(0, "outcome", outcome)
        comparison.to_csv(PATHS.tables_dir / f"mi_vs_completecase__{outcome}.csv", index=False)
        print(f"[mi] wrote mi_pooled__{outcome}.csv + mi_vs_completecase__{outcome}.csv")
        results[outcome] = {"complete_case": cc, "mi_pooled": mi_pooled}
    return results


def main() -> dict:
    df = pd.read_csv(PATHS.processed_data)
    return run(df)


if __name__ == "__main__":
    main()
