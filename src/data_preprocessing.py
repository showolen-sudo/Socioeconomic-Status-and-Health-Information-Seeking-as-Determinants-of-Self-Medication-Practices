"""Clean raw survey data and engineer analysis variables.

Produces an analysis-ready table with:

* the **SES composite** (from ``Education`` + ``HouseIncome``) and its tertiles,
* the **HISB composite** - a single standardized score combining the active
  information-seeking items (``Med7``-``Med9``, ``DTCA_Info``, ``DTCA_Prescribe``,
  ``Self_Treat``) with information-source breadth (count of ``Info_*`` selected),
* the two **self-medication outcomes** derived from the daily-use counts
  (``NumOTC`` -> ``otc_use``/``otc_freq``; ``NumHerbal`` -> ``herbal_use``/``herbal_freq``).

See ``docs/codebook.md`` for the full data dictionary.
"""

from __future__ import annotations

import pandas as pd

from .config import CONFIG, PATHS

EDUCATION_ORDER = {
    "Less than high school": 0,
    "High school graduate": 1,
    "Some college": 2,
    "Bachelor's degree": 3,
    "Graduate degree": 4,
}
INCOME_ORDER = {
    "Under $25,000": 0,
    "$25,000-$49,999": 1,
    "$50,000-$74,999": 2,
    "$75,000-$99,999": 3,
    "$100,000 or more": 4,
}
SELF_TREAT_ORDER = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
MED_ITEMS = ["Med7", "Med8", "Med9"]
DTCA_ITEMS = ["DTCA_Info", "DTCA_Prescribe"]
INFO_SOURCES = [
    "Info_Google", "Info_App", "Info_Fam", "Info_MD", "Info_RPh",
    "Info_OtherProf", "Info_Web", "Info_SocMedia", "Info_Print", "Info_Other",
]
FREQ_ORDER = ["None", "One", "Two", "Three+"]

REQUIRED_COLUMNS = {
    "respondent_id",
    "Education",
    "HouseIncome",
    *MED_ITEMS,
    *DTCA_ITEMS,
    "Self_Treat",
    *INFO_SOURCES,
    "NumOTC",
    "NumHerbal",
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def _yes_no(series: pd.Series) -> pd.Series:
    """Map Yes/No (case-insensitive) to 1/0."""
    return (
        series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype("float")
    )


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV and validate that required columns are present."""
    path = path or PATHS.raw_data
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")
    return df


def build_ses_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create the standardized SES composite (Education + HouseIncome) and tertiles."""
    df["education_code"] = df["Education"].astype(str).str.strip().map(EDUCATION_ORDER)
    df["income_code"] = df["HouseIncome"].astype(str).str.strip().map(INCOME_ORDER)
    for col, src in [("education_code", "Education"), ("income_code", "HouseIncome")]:
        if df[col].isna().any():
            bad = sorted(df.loc[df[col].isna(), src].unique())
            raise ValueError(f"Unrecognized {src} values: {bad}")

    df["education_z"] = _zscore(df["education_code"])
    df["income_z"] = _zscore(df["income_code"])
    df["ses_score"] = df[["education_z", "income_z"]].mean(axis=1)

    df["ses_tertile"] = pd.qcut(df["ses_score"], q=3, labels=["Low", "Middle", "High"])
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return df


def add_self_treat_score(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Self_Treat Likert item to a 1-5 numeric score."""
    df["Self_Treat"] = df["Self_Treat"].astype(str).str.strip()
    df["self_treat_score"] = df["Self_Treat"].map(SELF_TREAT_ORDER)
    if df["self_treat_score"].isna().any():
        bad = sorted(df.loc[df["self_treat_score"].isna(), "Self_Treat"].unique())
        raise ValueError(f"Unrecognized Self_Treat responses: {bad}")
    df["self_treat_score"] = df["self_treat_score"].astype(int)
    return df


def build_hisb_index(df: pd.DataFrame) -> pd.DataFrame:
    """Single standardized HISB composite from all information-seeking items.

    Components (each z-scored, then averaged):
    * Med7, Med8, Med9 (1-5 agreement),
    * DTCA_Info, DTCA_Prescribe (Yes/No -> 1/0),
    * self_treat_score (1-5),
    * info_source_count (number of Info_* sources selected, 0-10).
    """
    for item in DTCA_ITEMS:
        df[f"{item.lower()}_bin"] = _yes_no(df[item])
    df["info_source_count"] = df[INFO_SOURCES].sum(axis=1).astype(int)

    component_cols = [
        *MED_ITEMS,
        "dtca_info_bin",
        "dtca_prescribe_bin",
        "self_treat_score",
        "info_source_count",
    ]
    z = pd.DataFrame({c: _zscore(df[c].astype(float)) for c in component_cols})
    df["hisb_score"] = z.mean(axis=1)

    median = df["hisb_score"].median()
    df["hisb_high"] = (df["hisb_score"] >= median).astype(int)
    return df


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Binary use and ordinal frequency for OTC drugs and herbal supplements."""
    df["NumOTC"] = df["NumOTC"].astype(int).clip(lower=0)
    df["NumHerbal"] = df["NumHerbal"].astype(int).clip(lower=0)

    df["otc_use"] = (df["NumOTC"] >= 1).astype(int)
    df["herbal_use"] = (df["NumHerbal"] >= 1).astype(int)

    for count_col, freq_col in [("NumOTC", "otc_freq"), ("NumHerbal", "herbal_freq")]:
        binned = pd.cut(
            df[count_col].clip(upper=3),
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
            labels=FREQ_ORDER,
        )
        df[freq_col] = pd.Categorical(binned, categories=FREQ_ORDER, ordered=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full preprocessing chain and return analysis-ready data."""
    df = df.copy()
    df = df.drop_duplicates(subset="respondent_id")

    df = build_ses_index(df)
    df = add_self_treat_score(df)
    df = build_hisb_index(df)
    df = add_outcomes(df)
    return df.reset_index(drop=True)


def main() -> pd.DataFrame:
    """Run preprocessing and write the analysis table to disk."""
    PATHS.ensure_dirs()
    raw = load_raw()
    analysis = clean(raw)
    analysis.to_csv(PATHS.processed_data, index=False)
    print(f"[preprocess] wrote {len(analysis):,} rows -> {PATHS.processed_data}")
    print(
        "[preprocess] SES tertiles:",
        analysis["ses_tertile"].value_counts().to_dict(),
    )
    _ = CONFIG  # config reserved for future preprocessing parameters
    return analysis


if __name__ == "__main__":
    main()
