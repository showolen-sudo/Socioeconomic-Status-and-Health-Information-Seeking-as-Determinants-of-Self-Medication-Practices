"""Ordinal (proportional-odds) regression on self-medication *frequency*.

Each outcome's daily-use count is collapsed into an ordered category:

    None < One < Two < Three+

For every outcome (OTC drugs, herbal supplements) we fit a proportional-odds ordinal
logistic regression (cumulative logit link) with the study predictors: SES tertile and
the HISB composite. We then test the parallel-lines assumption (Brant) and, where it is
violated, fit a partial proportional-odds model that frees the offending term(s).

Outputs are suffixed with the outcome name, e.g. ``model_ordinal_freq__otc_use.csv``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import expit  # noqa: E402
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402

from . import model_spec  # noqa: E402
from .config import CONFIG, PATHS  # noqa: E402

FREQ_ORDER = ["None", "One", "Two", "Three+"]

_PRETTY_TERMS = {
    "C(ses_tertile, Treatment(reference='Low'))[T.Middle]": "SES: Middle (vs Low)",
    "C(ses_tertile, Treatment(reference='Low'))[T.High]": "SES: High (vs Low)",
    "hisb_score": "Health info-seeking (per unit)",
}


def _design_formula() -> str:
    """Right-hand-side formula (intercept dropped later for the ordinal model)."""
    return model_spec.adjusted_rhs()


def prepare(df: pd.DataFrame, freq_col: str) -> pd.DataFrame:
    """Return a copy with the ordered frequency outcome and SES category set."""
    df = df.copy()
    df[freq_col] = pd.Categorical(df[freq_col], categories=FREQ_ORDER, ordered=True)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    df = df.dropna(subset=[freq_col])
    return df


def build_design(df: pd.DataFrame, freq_col: str):
    """Build the (endog, exog, design_info) triple for OrderedModel."""
    X = patsy.dmatrix(_design_formula(), df, return_type="dataframe")
    design_info = X.design_info
    X = X.drop(columns=["Intercept"])
    y = df[freq_col]
    return y, X, design_info


def fit(df: pd.DataFrame, freq_col: str):
    """Fit the proportional-odds model and return the statsmodels result."""
    y, X, design_info = build_design(df, freq_col)
    model = OrderedModel(y, X, distr="logit")
    result = model.fit(method="bfgs", disp=False)
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
    """Predictor values held fixed in the prediction grid (means / reference levels)."""
    return {"hisb_score": df["hisb_score"].mean()}


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


def fig_predicted_by_ses(pred: pd.DataFrame, outcome: str, label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(pred))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(FREQ_ORDER)))
    for cat, color in zip(FREQ_ORDER, colors, strict=True):
        ax.bar(pred["ses_tertile"], pred[cat], bottom=bottom, label=cat, color=color)
        bottom += pred[cat].to_numpy()
    ax.set_xlabel("Socioeconomic status (tertile)")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"{label} frequency by SES (proportional-odds model)")
    ax.set_ylim(0, 1)
    ax.legend(title="Frequency", bbox_to_anchor=(1.02, 1), loc="upper left")
    path = PATHS.figures_dir / f"fig_freq_pred_by_ses__{outcome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ordinal] wrote {path.name}")


def brant_test(df: pd.DataFrame, freq_col: str) -> pd.DataFrame:
    """Brant (1990) test of the proportional-odds (parallel-lines) assumption.

    Fits the J = K-1 binary "cumulative" logistic models P(Y >= j) and tests whether
    the predictor slopes are equal across them. Returns an omnibus row plus a
    per-predictor row. A small p-value (< 0.05) indicates the assumption is violated.
    """
    data = prepare(df, freq_col)
    X = patsy.dmatrix(_design_formula(), data, return_type="dataframe")
    predictor_names = [c for c in X.columns if c != "Intercept"]
    Xmat = X.to_numpy()
    y = data[freq_col].cat.codes.to_numpy()
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

    var = np.zeros((J * p, J * p))
    for j in range(J):
        for m in range(J):
            if j == m:
                cov = inv_blocks[j]
            else:
                a_, b_ = min(j, m), max(j, m)
                w_jm = pis[:, b_] * (1 - pis[:, a_])
                cov = inv_blocks[j] @ xtwx(w_jm) @ inv_blocks[m]
            var[j * p:(j + 1) * p, m * p:(m + 1) * p] = cov[1:, 1:]

    beta_stack = betas[:, 1:].reshape(-1)

    def _wald(contrast: np.ndarray) -> tuple[float, int]:
        diff = contrast @ beta_stack
        vcv = contrast @ var @ contrast.T
        stat = float(diff @ np.linalg.solve(vcv, diff))
        return stat, contrast.shape[0]

    omni = np.zeros(((J - 1) * p, J * p))
    for m in range(1, J):
        omni[(m - 1) * p:m * p, 0:p] = -np.eye(p)
        omni[(m - 1) * p:m * p, m * p:(m + 1) * p] = np.eye(p)
    stat, dof = _wald(omni)
    rows = [
        {
            "variable": "Omnibus",
            "raw_term": "",
            "X2": round(stat, 3),
            "df": dof,
            "p_value": round(float(stats.chi2.sf(stat, dof)), 4),
            "violated_0.05": bool(stats.chi2.sf(stat, dof) < 0.05),
        }
    ]

    for k, name in enumerate(predictor_names):
        ck = np.zeros((J - 1, J * p))
        for m in range(1, J):
            ck[m - 1, k] = -1
            ck[m - 1, m * p + k] = 1
        stat, dof = _wald(ck)
        rows.append(
            {
                "variable": _PRETTY_TERMS.get(name, name),
                "raw_term": name,
                "X2": round(stat, 3),
                "df": dof,
                "p_value": round(float(stats.chi2.sf(stat, dof)), 4),
                "violated_0.05": bool(stats.chi2.sf(stat, dof) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _threshold_labels() -> list[str]:
    """Human-readable labels for the J cumulative cutpoints P(Y >= category)."""
    return [f">= {FREQ_ORDER[t]}" for t in range(1, len(FREQ_ORDER))]


def fit_partial_po(df: pd.DataFrame, freq_col: str, nonprop_terms=None, alpha: float = 0.05) -> dict:
    """Fit a partial proportional-odds (generalized ordered logit) model by MLE.

    Terms satisfying the proportional-odds assumption share one coefficient across all
    cutpoints; terms flagged by the Brant test (p < ``alpha``) are *freed* to take
    cutpoint-specific coefficients. When ``nonprop_terms`` is None the violating terms
    are selected automatically from :func:`brant_test`.
    """
    data = prepare(df, freq_col)
    _, X, _ = build_design(data, freq_col)
    cols = list(X.columns)
    Xmat = X.to_numpy()
    y = data[freq_col].cat.codes.to_numpy()
    n = len(y)
    K = len(FREQ_ORDER)
    J = K - 1

    if nonprop_terms is None:
        bt = brant_test(df, freq_col)
        mask = (bt["raw_term"] != "") & (bt["p_value"] < alpha)
        nonprop_terms = list(bt.loc[mask, "raw_term"])
    nonprop_terms = [c for c in nonprop_terms if c in cols]

    prop_idx = [i for i, c in enumerate(cols) if c not in nonprop_terms]
    nonprop_idx = [i for i, c in enumerate(cols) if c in nonprop_terms]
    n_prop, n_non = len(prop_idx), len(nonprop_idx)

    x_prop = Xmat[:, prop_idx] if n_prop else np.zeros((n, 0))
    x_non = Xmat[:, nonprop_idx] if n_non else np.zeros((n, 0))

    def unpack(theta: np.ndarray):
        a = theta[:J]
        b = theta[J:J + n_prop]
        g = theta[J + n_prop:].reshape(n_non, J) if n_non else np.zeros((0, J))
        return a, b, g

    def cum_probs(theta: np.ndarray) -> np.ndarray:
        a, b, g = unpack(theta)
        base = x_prop @ b if n_prop else np.zeros(n)
        cum = np.empty((n, J))
        for t in range(J):
            lp = a[t] + base + (x_non @ g[:, t] if n_non else 0.0)
            cum[:, t] = expit(lp)
        return cum

    def neg_loglik(theta: np.ndarray) -> float:
        cum = cum_probs(theta)
        prob = np.empty((n, K))
        prob[:, 0] = 1 - cum[:, 0]
        for c in range(1, K - 1):
            prob[:, c] = cum[:, c - 1] - cum[:, c]
        prob[:, K - 1] = cum[:, K - 2]
        pi = np.clip(prob[np.arange(n), y], 1e-9, 1.0)
        return -np.sum(np.log(pi))

    theta0 = np.zeros(J + n_prop + n_non * J)
    for t in range(J):
        frac = np.clip((y >= (t + 1)).mean(), 1e-3, 1 - 1e-3)
        theta0[t] = np.log(frac / (1 - frac))

    res = optimize.minimize(neg_loglik, theta0, method="BFGS")
    theta = res.x
    cov = np.atleast_2d(res.hess_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    labels = _threshold_labels()
    rows = []
    for j, idx in enumerate(prop_idx):
        pos = J + j
        rows.append(_or_row(cols[idx], "proportional (all cutpoints)", theta[pos], se[pos]))
    for m, idx in enumerate(nonprop_idx):
        for t in range(J):
            pos = J + n_prop + m * J + t
            rows.append(_or_row(cols[idx], labels[t], theta[pos], se[pos]))
    or_table = pd.DataFrame(rows)

    n_params = theta.size
    llf = -res.fun
    aic = 2 * n_params - 2 * llf
    bic = np.log(n) * n_params - 2 * llf
    fit_stats = pd.DataFrame(
        [
            {
                "n": n,
                "n_params": n_params,
                "log_likelihood": round(llf, 3),
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "nonproportional_terms": ", ".join(
                    _PRETTY_TERMS.get(c, c) for c in nonprop_terms
                )
                or "(none)",
                "converged": bool(res.success),
            }
        ]
    )
    return {
        "or_table": or_table,
        "fit": fit_stats,
        "nonprop_terms": nonprop_terms,
        "loglik": llf,
    }


def _or_row(raw_term: str, threshold: str, coef: float, se: float) -> dict:
    z = coef / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "term": _PRETTY_TERMS.get(raw_term, raw_term),
        "threshold": threshold,
        "coef_logit": round(float(coef), 4),
        "odds_ratio": round(float(np.exp(coef)), 4),
        "or_ci_low": round(float(np.exp(coef - 1.96 * se)), 4),
        "or_ci_high": round(float(np.exp(coef + 1.96 * se)), 4),
        "p_value": round(p, 4) if np.isfinite(p) else np.nan,
    }


def fig_partial_po(or_table: pd.DataFrame, nonprop_terms: list, outcome: str) -> None:
    """Plot cutpoint-specific odds ratios for the non-proportional terms."""
    if not nonprop_terms:
        return
    pretty = [_PRETTY_TERMS.get(c, c) for c in nonprop_terms]
    sub = or_table[
        or_table["term"].isin(pretty)
        & (or_table["threshold"] != "proportional (all cutpoints)")
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    thresholds = list(dict.fromkeys(sub["threshold"]))
    x = np.arange(len(thresholds))
    for term in pretty:
        t = sub[sub["term"] == term]
        ax.plot(x, t["odds_ratio"], marker="o", label=term)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("Cumulative cutpoint")
    ax.set_ylabel("Odds ratio")
    ax.set_title("Partial proportional odds: cutpoint-specific effects")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    path = PATHS.figures_dir / f"fig_partial_po_or__{outcome}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ordinal] wrote {path.name}")


def run_for_outcome(df: pd.DataFrame, outcome: str, freq_col: str, label: str):
    """Fit the ordinal + PPO models for a single outcome and export everything."""
    data = prepare(df, freq_col)
    result = fit(data, freq_col)

    odds_ratio_table(result).to_csv(
        PATHS.tables_dir / f"model_ordinal_freq__{outcome}.csv", index=False
    )
    thresholds_table(result).to_csv(
        PATHS.tables_dir / f"model_ordinal_thresholds__{outcome}.csv", index=False
    )
    fit_table(result).to_csv(
        PATHS.tables_dir / f"model_ordinal_fit__{outcome}.csv", index=False
    )
    print(f"[ordinal] wrote model_ordinal_* for {outcome}")

    brant = brant_test(df, freq_col)
    brant.to_csv(PATHS.tables_dir / f"model_ordinal_brant__{outcome}.csv", index=False)
    print(f"[ordinal] wrote model_ordinal_brant__{outcome}.csv")

    ppo = fit_partial_po(df, freq_col)
    ppo["or_table"].to_csv(PATHS.tables_dir / f"model_partial_po__{outcome}.csv", index=False)
    ppo["fit"].to_csv(PATHS.tables_dir / f"model_partial_po_fit__{outcome}.csv", index=False)
    print(
        f"[ordinal] wrote model_partial_po__{outcome}.csv "
        f"(non-proportional terms: {ppo['fit'].iloc[0]['nonproportional_terms']})"
    )
    fig_partial_po(ppo["or_table"], ppo["nonprop_terms"], outcome)

    pred = predicted_probs_by_ses(result, data)
    pred.round(4).to_csv(PATHS.tables_dir / f"ordinal_pred_by_ses__{outcome}.csv", index=False)
    fig_predicted_by_ses(pred, outcome, label)
    return result


def run(df: pd.DataFrame):
    """Fit ordinal + Brant + partial PO models for every outcome."""
    PATHS.ensure_dirs()
    results = {}
    for oc in model_spec.outcomes():
        results[oc["name"]] = run_for_outcome(df, oc["name"], oc["freq_var"], oc["label"])
    return results


def main():
    df = pd.read_csv(PATHS.processed_data)
    _ = CONFIG
    return run(df)


if __name__ == "__main__":
    main()
