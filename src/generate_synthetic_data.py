"""Generate a realistic *synthetic* survey dataset.

The data-generating process intentionally embeds plausible associations so the rest of
the pipeline (descriptives, regression, figures) produces meaningful, interpretable
output. It contains **no real individuals**.

Design choices:
* Self-medication probability is generated from a logistic model in which lower SES and
  higher health information-seeking increase the odds, with smaller covariate effects.
* Marginal rates (urban/female/insured shares) are configurable via ``config.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, PATHS

EDUCATION_LEVELS = ["None", "Primary", "Secondary", "Tertiary"]
OCCUPATION_LEVELS = ["Unemployed", "Manual", "Skilled", "Professional"]
FREQ_LABELS = ["Never", "Rarely", "Sometimes", "Often"]


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

    # --- Demographics -------------------------------------------------------
    age = rng.integers(18, 91, size=n)
    sex = rng.choice(["Female", "Male"], size=n, p=[syn["female_share"], 1 - syn["female_share"]])
    residence = rng.choice(
        ["Urban", "Rural"], size=n, p=[syn["urban_share"], 1 - syn["urban_share"]]
    )
    health_insurance = rng.choice(
        ["Yes", "No"], size=n, p=[syn["insured_share"], 1 - syn["insured_share"]]
    )
    chronic = rng.choice(["Yes", "No"], size=n, p=[0.22, 0.78])

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

    # Log-normal income, shifted by latent advantage and occupation.
    income = np.exp(6.0 + 0.45 * latent + 0.15 * occ_idx + rng.normal(0, 0.4, size=n))
    income = np.round(income, 2)

    # --- Health information-seeking behaviour (0-20) ------------------------
    # Higher with internet access, education, and (slightly) being younger.
    internet_p = _sigmoid(-0.5 + 0.8 * latent + (residence == "Urban") * 0.6)
    internet_access = np.where(rng.random(n) < internet_p, "Yes", "No")

    hisb_linear = (
        8
        + 1.6 * edu_idx
        + 2.5 * (internet_access == "Yes")
        - 0.03 * (age - 40)
        + rng.normal(0, 2.2, size=n)
    )
    hisb_score = np.clip(np.round(hisb_linear), 0, 20).astype(int)

    # --- Outcome: self-medication (logistic DGP) ----------------------------
    ses_std = (latent - latent.mean()) / latent.std()
    hisb_std = (hisb_score - hisb_score.mean()) / hisb_score.std()

    logit = (
        -0.4
        - 0.55 * ses_std                       # lower SES -> higher self-medication
        + 0.45 * hisb_std                       # more info-seeking -> more self-medication
        + 0.35 * (health_insurance == "No")     # uninsured -> more self-medication
        + 0.25 * (residence == "Rural")
        + 0.20 * (chronic == "Yes")
        + 0.004 * (age - 40)
    )
    prob = _sigmoid(logit)
    self_medication = (rng.random(n) < prob).astype(int)

    # Ordinal frequency consistent with the binary outcome and probability.
    freq = np.full(n, "Never", dtype=object)
    smokers = self_medication == 1
    freq_choice = rng.choice(
        FREQ_LABELS[1:], size=smokers.sum(), p=[0.30, 0.45, 0.25]
    )
    freq[smokers] = freq_choice

    df = pd.DataFrame(
        {
            "respondent_id": np.arange(1, n + 1),
            "age": age,
            "sex": sex,
            "residence": residence,
            "education": education,
            "income_monthly": income,
            "occupation": occupation,
            "health_insurance": health_insurance,
            "chronic_condition": chronic,
            "hisb_score": hisb_score,
            "internet_access": internet_access,
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
