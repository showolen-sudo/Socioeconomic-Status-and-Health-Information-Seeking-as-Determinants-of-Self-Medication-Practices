"""Clean raw survey data and engineer analysis variables.

Produces an analysis-ready table with the SES composite index, its tertiles, and a few
convenience recodes (see ``docs/codebook.md``).
"""

from __future__ import annotations

import pandas as pd

from .config import CONFIG, PATHS

EDUCATION_ORDER = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
OCCUPATION_ORDER = {"Unemployed": 0, "Manual": 1, "Skilled": 2, "Professional": 3}
SELF_TREAT_ORDER = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
REQUIRED_COLUMNS = {
    "respondent_id",
    "education",
    "income_monthly",
    "occupation",
    "chronic_condition",
    "hisb_score",
    "internet_access",
    "self_treat",
    "self_medication",
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV and validate that required columns are present."""
    path = path or PATHS.raw_data
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")
    return df


def build_ses_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create the standardized SES composite and its tertiles."""
    df["education_code"] = df["education"].map(EDUCATION_ORDER)
    df["occupation_code"] = df["occupation"].map(OCCUPATION_ORDER)

    df["income_z"] = _zscore(df["income_monthly"])
    df["education_z"] = _zscore(df["education_code"])
    df["occupation_z"] = _zscore(df["occupation_code"])

    df["ses_score"] = df[["income_z", "education_z", "occupation_z"]].mean(axis=1)

    df["ses_tertile"] = pd.qcut(df["ses_score"], q=3, labels=["Low", "Middle", "High"])
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return df


def add_hisb_split(df: pd.DataFrame) -> pd.DataFrame:
    median = df["hisb_score"].median()
    df["hisb_high"] = (df["hisb_score"] >= median).astype(int)
    return df


def add_self_treat_score(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Self_Treat Likert item to a 1-5 numeric score."""
    df["self_treat"] = df["self_treat"].astype(str).str.strip()
    df["self_treat_score"] = df["self_treat"].map(SELF_TREAT_ORDER)
    if df["self_treat_score"].isna().any():
        bad = sorted(df.loc[df["self_treat_score"].isna(), "self_treat"].unique())
        raise ValueError(f"Unrecognized self_treat responses: {bad}")
    df["self_treat_score"] = df["self_treat_score"].astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full preprocessing chain and return analysis-ready data."""
    df = df.copy()
    df = df.drop_duplicates(subset="respondent_id")
    df = df[df["income_monthly"] >= 0]
    df["self_medication"] = df["self_medication"].astype(int)

    df = build_ses_index(df)
    df = add_hisb_split(df)
    df = add_self_treat_score(df)
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
