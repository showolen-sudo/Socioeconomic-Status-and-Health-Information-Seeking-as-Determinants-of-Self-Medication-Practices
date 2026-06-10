"""Generate a realistic *synthetic* survey dataset.

The data-generating process intentionally embeds plausible associations so the rest of
the pipeline (descriptives, regression, figures) produces meaningful, interpretable
output. It contains **no real individuals**.

Predictors of interest: socioeconomic status (via education, income, occupation),
health information-seeking behaviour (HISB), and a self-treatment attitude item
(Self_Treat). Demographic covariates are intentionally excluded from this design.

Design choices:
* Self-medication probability is generated from a logistic model in which lower SES
  increases the odds, higher health information-seeking increases the odds, and
  stronger agreement with the Self_Treat statement increases the odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, PATHS

EDUCATION_LEVELS = ["None", "Primary", "Secondary", "Tertiary"]
OCCUPATION_LEVELS = ["Unemployed", "Manual", "Skilled", "Professional"]
FREQ_LABELS = ["Never", "Rarely", "Sometimes", "Often"]
# Likert response to: "I can usually self-treat with remedies that are available
# without a doctor's prescription." (Self_Treat)
SELF_TREAT_LEVELS = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Build and return a synthetic survey :class:`~pandas.DataFrame`.

    Parameters
    ----------
    n:
        Number of respondents. Defaults to ``config.synthetic.n_respondents``.
    seed:
        Random seed. Defaults to ``config.seed`` for full reproducibility.
    """
    syn = CONFIG["synthetic"]
    n = int(n if n is not None else syn["n_respondents"])
    seed = int(seed if seed is not None else CONFIG["seed"])
    rng = np.random.default_rng(seed)

    # --- Socioeconomic components (correlated) ------------------------------
    # A latent "advantage" factor drives education, income, and occupation jointly.
    latent = rng.normal(0, 1, size=n)

    edu_idx = np.clip(
        np.round(1.4 + 0.9 * latent + rng.normal(0, 0.7, size=n)).astype(int), 0, 3
    )
    education = np.array(EDUCATION_LEVELS)[edu_idx]

    occ_idx = np.clip(
        np.round(1.3 + 0.8 * latent + rng.normal(0, 0.8, size=n)).astype(int), 0, 3
    )
    occupation = np.array(OCCUPATION_LEVELS)[occ_idx]

    income = np.exp(6.0 + 0.45 * latent + 0.15 * occ_idx + rng.normal(0, 0.4, size=n))
    income = np.round(income, 2)

    chronic = rng.choice(["Yes", "No"], size=n, p=[0.22, 0.78])

    # --- Health information-seeking behaviour (0-20) ------------------------
    # Higher with internet access and education.
    internet_p = _sigmoid(-0.4 + 0.9 * latent)
    internet_access = np.where(rng.random(n) < internet_p, "Yes", "No")

    hisb_linear = (
        9
        + 1.6 * edu_idx
        + 2.5 * (internet_access == "Yes")
        + rng.normal(0, 2.2, size=n)
    )
    hisb_score = np.clip(np.round(hisb_linear), 0, 20).astype(int)

    ses_std = (latent - latent.mean()) / latent.std()
    hisb_std = (hisb_score - hisb_score.mean()) / hisb_score.std()

    # --- Self-treatment attitude (Self_Treat, 5-point Likert) ---------------
    # Agreement with: "I can usually self-treat with remedies available without a
    # doctor's prescription." More agreement at lower SES, higher info-seeking,
    # and among those with a chronic condition.
    attitude = (
        -0.40 * ses_std
        + 0.30 * hisb_std
        + 0.30 * (chronic == "Yes")
        + rng.normal(0, 1, size=n)
    )
    self_treat_code = np.digitize(attitude, [-1.0, -0.3, 0.3, 1.0])  # 0..4
    self_treat = np.array(SELF_TREAT_LEVELS)[self_treat_code]
    self_treat_score = self_treat_code + 1  # 1..5

    # --- Outcome: self-medication (logistic DGP) ----------------------------
    logit = (
        -0.5
        - 0.50 * ses_std                       # lower SES -> more self-medication
        + 0.40 * hisb_std                       # more info-seeking -> more self-medication
        + 0.45 * (self_treat_score - 3)         # stronger agreement -> more self-medication
        + 0.20 * (chronic == "Yes")
    )
    prob = _sigmoid(logit)
    self_medication = (rng.random(n) < prob).astype(int)

    # Ordinal frequency consistent with the binary outcome.
    freq = np.full(n, "Never", dtype=object)
    users = self_medication == 1
    freq[users] = rng.choice(FREQ_LABELS[1:], size=users.sum(), p=[0.30, 0.45, 0.25])

    df = pd.DataFrame(
        {
            "respondent_id": np.arange(1, n + 1),
            "education": education,
            "income_monthly": income,
            "occupation": occupation,
            "chronic_condition": chronic,
            "hisb_score": hisb_score,
            "internet_access": internet_access,
            "self_treat": self_treat,
            "self_medication": self_medication,
            "self_medication_freq": freq,
        }
    )
    return df


def main() -> pd.DataFrame:
    """Generate the dataset and write it to ``config.paths.raw_data``."""
    PATHS.ensure_dirs()
    df = generate()
    df.to_csv(PATHS.raw_data, index=False)
    print(f"[generate] wrote {len(df):,} rows -> {PATHS.raw_data}")
    print(f"[generate] self-medication prevalence: {df['self_medication'].mean():.1%}")
    return df


if __name__ == "__main__":
    main()
