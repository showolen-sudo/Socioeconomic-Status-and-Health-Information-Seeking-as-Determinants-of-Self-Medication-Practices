"""
Step 1 - Load raw survey CSV and build analysis variables.

HISB excludes Self_Treat so Self_Treat can enter the logistic model separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import (
    DTCA_ITEMS,
    EDUCATION_ORDER,
    FREQ_ORDER,
    HISB_COMPONENTS,
    INCOME_ORDER,
    INFO_SOURCES,
    REQUIRED_COLUMNS,
    SELF_TREAT_ORDER,
)


def zscore(series: pd.Series) -> pd.Series:
    """Standardize to mean 0, SD 1 (population SD, ddof=0)."""
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std


def yes_no(series: pd.Series) -> pd.Series:
    """Map Yes/No strings to 1/0; also accepts 0/1 numeric coding."""
    mapped = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)
    return mapped.astype("float")


def load_raw(path: Path | str) -> pd.DataFrame:
    """Load CSV and verify all required NCME columns exist."""
    df = pd.read_csv(path)
    if "Info_Other" not in df.columns:
        df["Info_Other"] = 0
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {sorted(missing)}")
    return df


def _map_ordered(series: pd.Series, order: dict[str, int], name: str) -> pd.Series:
    """Map label strings; fall back to numeric codes when already coded."""
    mapped = series.astype(str).str.strip().map(order)
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)
    if mapped.isna().any():
        bad = sorted(series.loc[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unrecognized {name} values: {bad}")
    return mapped


def build_ses_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    SES = mean of z-scored education and income codes.
    Tertiles split the sample into Low / Middle / High groups.
    """
    df = df.copy()
    df["education_code"] = _map_ordered(df["Education"], EDUCATION_ORDER, "Education")
    df["income_code"] = _map_ordered(df["HouseIncome"], INCOME_ORDER, "HouseIncome")

    df["education_z"] = zscore(df["education_code"])
    df["income_z"] = zscore(df["income_code"])
    df["ses_score"] = df[["education_z", "income_z"]].mean(axis=1)

    df["ses_tertile"] = pd.qcut(df["ses_score"], q=3, labels=["Low", "Middle", "High"])
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return df


def add_self_treat_score(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Likert Self_Treat item as integer 1 (disagree) to 5 (agree)."""
    df = df.copy()
    df["self_treat_score"] = _map_ordered(df["Self_Treat"], SELF_TREAT_ORDER, "Self_Treat")
    df["self_treat_score"] = df["self_treat_score"].astype(int)
    df["self_treat_z"] = zscore(df["self_treat_score"].astype(float))
    return df


def build_hisb_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    HISB composite (v2): z-score each component, then take the mean.

    Components: Med7, Med8, Med9, DTCA_Info, DTCA_Prescribe, info_source_count.
    Self_Treat is intentionally EXCLUDED so it can be modelled separately.
    """
    df = df.copy()
    for item in DTCA_ITEMS:
        df[f"{item.lower()}_bin"] = yes_no(df[item])

    df["info_source_count"] = df[INFO_SOURCES].sum(axis=1).astype(int)

    z = pd.DataFrame({c: zscore(df[c].astype(float)) for c in HISB_COMPONENTS})
    df["hisb_score"] = z.mean(axis=1)
    df["hisb_high"] = (df["hisb_score"] >= df["hisb_score"].median()).astype(int)
    return df


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Outcomes from daily-use counts:
      otc_use / herbal_use = 1 if count >= 1, else 0
      otc_freq / herbal_freq = None / One / Two / Three+
    """
    df = df.copy()
    df["NumOTC"] = df["NumOTC"].astype(int).clip(lower=0)
    df["NumHerbal"] = df["NumHerbal"].astype(int).clip(lower=0)
    df["otc_use"] = (df["NumOTC"] >= 1).astype(int)
    df["herbal_use"] = (df["NumHerbal"] >= 1).astype(int)

    for count_col, freq_col in (("NumOTC", "otc_freq"), ("NumHerbal", "herbal_freq")):
        binned = pd.cut(
            df[count_col].clip(upper=3),
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
            labels=FREQ_ORDER,
        )
        df[freq_col] = pd.Categorical(binned, categories=FREQ_ORDER, ordered=True)
    return df


def preprocess(raw_path: Path | str) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns one row per respondent."""
    df = load_raw(raw_path)
    df = df.drop_duplicates(subset=["respondent_id"])
    df = build_ses_index(df)
    df = add_self_treat_score(df)
    df = build_hisb_index(df)
    df = add_outcomes(df)
    return df.reset_index(drop=True)
