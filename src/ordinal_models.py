"""Ordinal (proportional-odds) regression on self-medication *frequency*.

The secondary outcome ``self_medication_freq`` is an ordered category:

    Never < Rarely < Sometimes < Often

We fit a proportional-odds ordinal logistic regression (cumulative logit link) with
the same predictor set as the binary model: SES tertile, health information-seeking
score, and the demographic covariates. Coefficients are exported as proportional-odds
ratios (OR > 1 => higher odds of being in a *more frequent* category).

Outputs (written to ``results/``):
* ``model_ordinal_freq.csv``        - OR table (slopes only, with 95% CIs).
* ``model_ordinal_thresholds.csv``  - estimated cutpoints between categories.
* ``model_ordinal_fit.csv``         - log-likelihood, pseudo-R2, AIC/BIC.
* ``fig_freq_pred_by_ses.png``      - predicted category probabilities by SES.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402

from .config import CONFIG, PATHS  # noqa: E402

FREQ_ORDER = ["Never", "Rarely", "Sometimes", "Often"]
OUTCOME = "self_medication_freq"

_PRETTY_TERMS = {
    "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
    "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
    "self_treat_score": "Self-treat agreement (per point)",
    "hisb_score": "Health info-seeking (per point)",
}


def _design_formula() -> str:
    """Right-hand-side formula (intercept dropped later for the ordinal model)."""
    ref = CONFIG["modelling"]["ses_reference"]
    ses = f"C(ses_tertile, Treatment(reference='{ref}'))"
    covariates = CONFIG["modelling"]["covariates"]
    numeric = {"age", "hisb_score", "ses_score", "income_monthly", "self_treat_score"}
    cov_terms = [c if c in numeric else f"C({c})" for c in covariates]
    return " + ".join([ses, "hisb_score", *cov_terms])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the ordered frequency outcome and SES category set."""
    df = df.copy()
    df[OUTCOME] = pd.Categorical(df[OUTCOME], categories=FREQ_ORDER, ordered=True)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    df = df.dropna(subset=[OUTCOME])
    return df


def build_design(df: pd.DataFrame):
    """Build the (endog, exog) pair for OrderedModel.

    The intercept is removed because OrderedModel estimates category thresholds in its
    place; treatment (reference) coding is preserved for every categorical predictor.
    """
    X = patsy.dmatrix(_design_formula(), df, return_type="dataframe")
    design_info = X.design_info
    X = X.drop(columns=["Intercept"])
    y = df[OUTCOME]
    return y, X, design_info


def fit(df: pd.DataFrame):
    """Fit the proportional-odds model and return the statsmodels result."""
    y, X, design_info = build_design(df)
    model = OrderedModel(y, X, distr="logit")
    result = model.fit(method="bfgs", disp=False)
    # Stash design info for prediction grids.
    result._design_info = design_info
    result._exog_columns = list(X.columns)
    return result


def odds_ratio_table(result) -> pd.DataFrame:
    """Slope coefficients as proportional-odds ratios with 95% CIs."""
    slope_terms = result._exog_columns
    params = result.params.loc[slope_terms]
    conf = result.conf_int().loc[slope_terms]
    conf.columns = ["ci_low", "ci_high"]
    table = pd.DataFrame(
        {
            "term": [_PRETTY_TERMS.get(t, t) for t in slope_terms],
            "raw_term": slope_terms,
            "coef_logit": params.values,
            "odds_ratio": np.exp(params.values),
            "or_ci_low": np.exp(conf["ci_low"].values),
            "or_ci_high": np.exp(conf["ci_high"].values),
            "p_value": result.pvalues.loc[slope_terms].values,
        }
    )
    for col in ["coef_logit", "odds_ratio", "or_ci_low", "or_ci_high"]:
        table[col] = table[col].round(4)
    table["p_value"] = table["p_value"].round(4)
    return table.reset_index(drop=True)


def thresholds_table(result) -> pd.DataFrame:
    """Estimated cutpoints separating adjacent frequency categories."""
    slope_terms = set(result._exog_columns)
    cut = result.params[[t for t in result.params.index if t not in slope_terms]]
    return pd.DataFrame({"boundary": cut.index, "estimate": cut.values.round(4)})


def fit_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n": int(result.nobs),
                "log_likelihood": round(result.llf, 3),
                "pseudo_r2_mcfadden": round(result.prsquared, 4),
                "aic": round(result.aic, 2),
                "bic": round(result.bic, 2),
            }
        ]
    )


def _reference_row(df: pd.DataFrame) -> dict:
    """Covariate values held fixed in the prediction grid (means / reference levels)."""
    return {
        "hisb_score": df["hisb_score"].mean(),
        "self_treat_score": df["self_treat_score"].mean(),
    }


def predicted_probs_by_ses(result, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted probability of each frequency category across SES tertiles."""
    base = _reference_row(df)
    grid = pd.DataFrame(
        [{**base, "ses_tertile": level} for level in ["Low", "Middle", "High"]]
    )
    grid["ses_tertile"] = pd.Categorical(
        grid["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    Xnew = patsy.dmatrix(result._design_info, grid, return_type="dataframe")
    Xnew = Xnew.drop(columns=["Intercept"])
    probs = result.predict(Xnew)
    out = pd.DataFrame(np.asarray(probs), columns=FREQ_ORDER)
    out.insert(0, "ses_tertile", ["Low", "Middle", "High"])
    return out


def fig_predicted_by_ses(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(pred))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(FREQ_ORDER)))
    for cat, color in zip(FREQ_ORDER, colors, strict=True):
        ax.bar(pred["ses_tertile"], pred[cat], bottom=bottom, label=cat, color=color)
        bottom += pred[cat].to_numpy()
    ax.set_xlabel("Socioeconomic status (tertile)")
    ax.set_ylabel("Predicted probability")
    ax.set_title("Self-medication frequency by SES (proportional-odds model)")
    ax.set_ylim(0, 1)
    ax.legend(title="Frequency", bbox_to_anchor=(1.02, 1), loc="upper left")
    path = PATHS.figures_dir / "fig_freq_pred_by_ses.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ordinal] wrote {path.name}")


def brant_test(df: pd.DataFrame) -> pd.DataFrame:
    """Brant (1990) test of the proportional-odds (parallel-lines) assumption.

    Fits the J = K-1 binary "cumulative" logistic models P(Y >= j) and tests whether
    the predictor slopes are equal across them (the parallel-lines assumption). Returns
    an omnibus row plus a per-predictor row. A small p-value (< 0.05) indicates the
    assumption is violated for that term / overall.
    """
    data = prepare(df)
    X = patsy.dmatrix(_design_formula(), data, return_type="dataframe")
    predictor_names = [c for c in X.columns if c != "Intercept"]
    Xmat = X.to_numpy()
    y = data[OUTCOME].cat.codes.to_numpy()
    n, p_full = Xmat.shape
    p = p_full - 1  # predictors excluding intercept
    K = len(FREQ_ORDER)
    J = K - 1

    betas = np.zeros((J, p_full))
    pis = np.zeros((n, J))
    for idx, cut in enumerate(range(1, K)):
        z = (y >= cut).astype(int)
        res = sm.Logit(z, Xmat).fit(disp=False)
        betas[idx] = res.params
        pis[:, idx] = res.predict(Xmat)

    def xtwx(weights: np.ndarray) -> np.ndarray:
        return Xmat.T @ (Xmat * weights[:, None])

    inv_blocks = [np.linalg.pinv(xtwx(pis[:, j] * (1 - pis[:, j]))) for j in range(J)]

    # Full covariance of stacked predictor coefficients (intercepts dropped).
    var = np.zeros((J * p, J * p))
    for j in range(J):
        for m in range(J):
            if j == m:
                cov = inv_blocks[j]
            else:
                a_, b_ = min(j, m), max(j, m)
                w_jm = pis[:, b_] * (1 - pis[:, a_])  # cov of cumulative indicators
                cov = inv_blocks[j] @ xtwx(w_jm) @ inv_blocks[m]
            var[j * p:(j + 1) * p, m * p:(m + 1) * p] = cov[1:, 1:]

    beta_stack = betas[:, 1:].reshape(-1)

    def _wald(contrast: np.ndarray) -> tuple[float, int]:
        diff = contrast @ beta_stack
        vcv = contrast @ var @ contrast.T
        stat = float(diff @ np.linalg.solve(vcv, diff))
        return stat, contrast.shape[0]

    # Omnibus: beta_m - beta_1 = 0 for m = 2..J across all predictors.
    omni = np.zeros(((J - 1) * p, J * p))
    for m in range(1, J):
        omni[(m - 1) * p:m * p, 0:p] = -np.eye(p)
        omni[(m - 1) * p:m * p, m * p:(m + 1) * p] = np.eye(p)
    stat, dof = _wald(omni)
    rows = [
        {
            "variable": "Omnibus",
            "X2": round(stat, 3),
            "df": dof,
            "p_value": round(float(stats.chi2.sf(stat, dof)), 4),
            "violated_0.05": bool(stats.chi2.sf(stat, dof) < 0.05),
        }
    ]

    # Per-predictor contrasts.
    for k, name in enumerate(predictor_names):
        ck = np.zeros((J - 1, J * p))
        for m in range(1, J):
            ck[m - 1, k] = -1
            ck[m - 1, m * p + k] = 1
        stat, dof = _wald(ck)
        rows.append(
            {
                "variable": _PRETTY_TERMS.get(name, name),
                "X2": round(stat, 3),
                "df": dof,
                "p_value": round(float(stats.chi2.sf(stat, dof)), 4),
                "violated_0.05": bool(stats.chi2.sf(stat, dof) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def run(df: pd.DataFrame):
    """Fit the ordinal model, export tables + figure, return the fitted result."""
    PATHS.ensure_dirs()
    data = prepare(df)
    result = fit(data)

    odds_ratio_table(result).to_csv(PATHS.tables_dir / "model_ordinal_freq.csv", index=False)
    print("[ordinal] wrote model_ordinal_freq.csv")
    thresholds_table(result).to_csv(
        PATHS.tables_dir / "model_ordinal_thresholds.csv", index=False
    )
    print("[ordinal] wrote model_ordinal_thresholds.csv")
    fit_table(result).to_csv(PATHS.tables_dir / "model_ordinal_fit.csv", index=False)
    print("[ordinal] wrote model_ordinal_fit.csv")

    brant_test(df).to_csv(PATHS.tables_dir / "model_ordinal_brant.csv", index=False)
    print("[ordinal] wrote model_ordinal_brant.csv (proportional-odds assumption)")

    pred = predicted_probs_by_ses(result, data)
    pred.round(4).to_csv(PATHS.tables_dir / "ordinal_pred_by_ses.csv", index=False)
    fig_predicted_by_ses(pred)
    return result


def main():
    df = pd.read_csv(PATHS.processed_data)
    return run(df)


if __name__ == "__main__":
    main()
