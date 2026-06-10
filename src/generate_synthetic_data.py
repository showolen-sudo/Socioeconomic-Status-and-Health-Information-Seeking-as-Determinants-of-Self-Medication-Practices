"""Generate a realistic *synthetic* survey dataset.

The data-generating process intentionally embeds plausible associations so the rest of
the pipeline (descriptives, regression, figures) produces meaningful, interpretable
output. It contains **no real individuals**.

Variables mirror the real questionnaire:

* Socioeconomic status (SES): ``Education`` and ``HouseIncome``.
* Health information-seeking behaviour (HISB):
    - ``Med7``/``Med8``/``Med9`` (5-point agreement items about gathering / reviewing /
      continuing to seek medicine information),
    - ``DTCA_Info`` / ``DTCA_Prescribe`` (Yes/No advertisement-driven items),
    - ``Self_Treat`` (5-point agreement with self-treating using non-prescription
      remedies),
    - ``Info_*`` checkboxes for the information sources a respondent depends on.
* Self-medication (outcomes): ``NumOTC`` and ``NumHerbal`` - the number of over-the-counter
  drugs and herbal supplements taken every day.

Design choices (per the analysis plan):
* Higher SES -> more active information seeking (positive SES -> HISB path).
* Lower SES and higher HISB independently raise the expected number of OTC drugs and
  herbal supplements used (Poisson counts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, PATHS

EDUCATION_LEVELS = [
    "Less than high school",
    "High school graduate",
    "Some college",
    "Bachelor's degree",
    "Graduate degree",
]
INCOME_LEVELS = [
    "Under $25,000",
    "$25,000-$49,999",
    "$50,000-$74,999",
    "$75,000-$99,999",
    "$100,000 or more",
]
SELF_TREAT_LEVELS = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]
INFO_SOURCES = [
    "Info_Google",
    "Info_App",
    "Info_Fam",
    "Info_MD",
    "Info_RPh",
    "Info_OtherProf",
    "Info_Web",
    "Info_SocMedia",
    "Info_Print",
    "Info_Other",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def _likert(center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round a latent score into a 1-5 Likert response."""
    return np.clip(np.round(center + rng.normal(0, 0.8, size=center.shape)), 1, 5).astype(int)


def generate(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Build and return a synthetic survey :class:`~pandas.DataFrame`."""
    syn = CONFIG["synthetic"]
    n = int(n if n is not None else syn["n_respondents"])
    seed = int(seed if seed is not None else CONFIG["seed"])
    rng = np.random.default_rng(seed)

    # --- Socioeconomic status: Education + HouseIncome (correlated) ----------
    latent = rng.normal(0, 1, size=n)  # latent SES advantage
    edu_idx = np.clip(np.round(2 + 1.0 * latent + rng.normal(0, 0.8, n)).astype(int), 0, 4)
    inc_idx = np.clip(
        np.round(2 + 0.8 * latent + 0.25 * edu_idx - 0.5 + rng.normal(0, 0.8, n)).astype(int),
        0,
        4,
    )
    education = np.array(EDUCATION_LEVELS)[edu_idx]
    house_income = np.array(INCOME_LEVELS)[inc_idx]
    ses_std = _z(0.5 * edu_idx + 0.5 * inc_idx)

    # --- Health information-seeking behaviour --------------------------------
    seek = rng.normal(0, 1, size=n)  # latent active-seeking propensity

    # Med7-9: agreement with gathering / reviewing / continuing to seek info.
    med7 = _likert(3 + 0.85 * seek + 0.20 * ses_std, rng)
    med8 = _likert(3 + 0.80 * seek + 0.15 * ses_std, rng)
    med9 = _likert(3 + 0.75 * seek + 0.10 * ses_std, rng)

    # DTCA (direct-to-consumer advertisement) items, Yes/No.
    dtca_info = np.where(rng.random(n) < _sigmoid(-0.2 + 0.8 * seek + 0.15 * ses_std), "Yes", "No")
    dtca_presc = np.where(rng.random(n) < _sigmoid(-0.7 + 0.7 * seek), "Yes", "No")

    # Self_Treat attitude: more agreement at lower SES and higher seeking.
    attitude = -0.35 * ses_std + 0.30 * seek + rng.normal(0, 1, n)
    self_treat_code = np.digitize(attitude, [-1.0, -0.3, 0.3, 1.0])  # 0..4
    self_treat = np.array(SELF_TREAT_LEVELS)[self_treat_code]

    # Information sources (checkboxes). Online/app/social skew higher with SES &
    # seeking; physician/pharmacist are more universal.
    source_logits = {
        "Info_Google": -0.2 + 0.9 * seek + 0.5 * ses_std,
        "Info_App": -1.1 + 0.7 * seek + 0.4 * ses_std,
        "Info_Fam": -0.6 + 0.2 * seek - 0.2 * ses_std,
        "Info_MD": 0.4 + 0.1 * seek + 0.2 * ses_std,
        "Info_RPh": -0.1 + 0.3 * seek + 0.1 * ses_std,
        "Info_OtherProf": -1.3 + 0.3 * seek,
        "Info_Web": -0.5 + 0.8 * seek + 0.4 * ses_std,
        "Info_SocMedia": -1.0 + 0.6 * seek - 0.1 * ses_std,
        "Info_Print": -1.4 + 0.2 * seek + 0.2 * ses_std,
        "Info_Other": -2.2 + 0.1 * seek,
    }
    sources = {
        name: (rng.random(n) < _sigmoid(logit)).astype(int)
        for name, logit in source_logits.items()
    }
    source_count = np.sum(list(sources.values()), axis=0)

    # --- Internal HISB composite (mirrors data_preprocessing) ----------------
    components = [
        _z(med7), _z(med8), _z(med9),
        _z(dtca_info == "Yes"), _z(dtca_presc == "Yes"),
        _z(self_treat_code + 1), _z(source_count),
    ]
    hisb_std = _z(np.mean(components, axis=0))

    # --- Outcomes: daily counts of OTC drugs and herbal supplements ----------
    # Lower SES and higher information-seeking each raise expected counts.
    otc_rate = np.exp(-0.2 - 0.50 * ses_std + 0.40 * hisb_std)
    herbal_rate = np.exp(-0.5 - 0.35 * ses_std + 0.45 * hisb_std)
    num_otc = rng.poisson(otc_rate)
    num_herbal = rng.poisson(herbal_rate)

    data = {
        "respondent_id": np.arange(1, n + 1),
        "Education": education,
        "HouseIncome": house_income,
        "Med7": med7,
        "Med8": med8,
        "Med9": med9,
        "DTCA_Info": dtca_info,
        "DTCA_Prescribe": dtca_presc,
        "Self_Treat": self_treat,
    }
    data.update(sources)
    data["NumOTC"] = num_otc
    data["NumHerbal"] = num_herbal
    return pd.DataFrame(data)


def main() -> pd.DataFrame:
    """Generate the dataset and write it to ``config.paths.raw_data``."""
    PATHS.ensure_dirs()
    df = generate()
    df.to_csv(PATHS.raw_data, index=False)
    print(f"[generate] wrote {len(df):,} rows -> {PATHS.raw_data}")
    print(
        f"[generate] OTC use: {(df['NumOTC'] >= 1).mean():.1%} | "
        f"herbal use: {(df['NumHerbal'] >= 1).mean():.1%}"
    )
    return df


if __name__ == "__main__":
    main()
