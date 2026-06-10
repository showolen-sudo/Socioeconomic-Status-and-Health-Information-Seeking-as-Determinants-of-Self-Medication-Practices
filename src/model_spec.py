"""Shared model-specification helpers.

Centralises the predictor formulas and the (multi-)outcome list so every analysis
module builds models consistently. Predictors of interest are socioeconomic status
(SES tertile) and health information-seeking behaviour (HISB composite); optional
adjustment covariates can be added via ``config.modelling.covariates``.
"""

from __future__ import annotations

from .config import CONFIG

# Terms that should enter formulas as numeric (everything else is wrapped in C()).
NUMERIC_TERMS = {
    "hisb_score",
    "ses_score",
    "self_treat_score",
    "income_code",
    "education_code",
    "info_source_count",
}

HISB_TERM = "hisb_score"


def ses_reference() -> str:
    return CONFIG["modelling"]["ses_reference"]


def ses_term() -> str:
    """Treatment-coded SES tertile term with the configured reference level."""
    return f"C(ses_tertile, Treatment(reference='{ses_reference()}'))"


def covariate_terms() -> list[str]:
    """Optional adjustment covariates, numeric left bare and categorical wrapped in C()."""
    covs = CONFIG["modelling"].get("covariates") or []
    return [c if c in NUMERIC_TERMS else f"C({c})" for c in covs]


def outcomes() -> list[dict]:
    """List of outcome specs: each has name, label, count_var, freq_var."""
    return list(CONFIG["modelling"]["outcomes"])


def _join(parts) -> str:
    return " + ".join(p for p in parts if p)


def crude_ses_rhs() -> str:
    return ses_term()


def adjusted_rhs() -> str:
    """SES + HISB (+ any configured covariates)."""
    return _join([ses_term(), HISB_TERM, *covariate_terms()])


def interaction_rhs() -> str:
    """SES x HISB interaction (+ any configured covariates)."""
    return _join([f"{ses_term()} * {HISB_TERM}", *covariate_terms()])
