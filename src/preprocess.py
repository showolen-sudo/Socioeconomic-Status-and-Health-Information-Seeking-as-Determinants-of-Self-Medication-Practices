"""Load and preprocess the NCME 2021 survey for manuscript logistic analysis."""

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
    s = series.astype(float)
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0.0
    return (s - s.mean()) / sd


def yes_no(series: pd.Series) -> pd.Series:
    """Map Yes/No or NCME 1=Yes/2=No coding to 1/0."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) == 1).astype(int)
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "2": 0})
    )
    return mapped.fillna(0).astype(int)


def _ordinal_code(series: pd.Series, mapping: dict) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    mapped = series.astype(str).str.strip().map(mapping)
    # Fall back to numeric parse for digit strings not in mapping
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = mapped.fillna(numeric)
    return mapped.astype(float)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    return df


def build_ses_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["education_code"] = _ordinal_code(df["Education"], EDUCATION_ORDER)
    df["income_code"] = _ordinal_code(df["HouseIncome"], INCOME_ORDER)
    if df["education_code"].isna().any() or df["income_code"].isna().any():
        bad_edu = df.loc[df["education_code"].isna(), "Education"].unique().tolist()
        bad_inc = df.loc[df["income_code"].isna(), "HouseIncome"].unique().tolist()
        raise ValueError(f"Unmapped Education/HouseIncome values: {bad_edu=} {bad_inc=}")
    df["education_z"] = zscore(df["education_code"])
    df["income_z"] = zscore(df["income_code"])
    df["ses_score"] = (df["education_z"] + df["income_z"]) / 2.0
    try:
        df["ses_tertile"] = pd.qcut(
            df["ses_score"],
            q=3,
            labels=["Low", "Middle", "High"],
            duplicates="drop",
        )
    except ValueError:
        ranks = df["ses_score"].rank(method="average")
        df["ses_tertile"] = pd.qcut(
            ranks,
            q=3,
            labels=["Low", "Middle", "High"],
            duplicates="drop",
        )
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"],
        categories=["Low", "Middle", "High"],
        ordered=True,
    )
    return df


def add_self_treat_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["self_treat_score"] = _ordinal_code(df["Self_Treat"], SELF_TREAT_ORDER)
    if df["self_treat_score"].isna().any():
        bad = df.loc[df["self_treat_score"].isna(), "Self_Treat"].unique().tolist()
        raise ValueError(f"Unmapped Self_Treat values: {bad}")
    df["self_treat_z"] = zscore(df["self_treat_score"])
    return df


def build_hisb_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    HISB composite: z-score each component, then take the mean.

    Components: Med7, Med8, Med9, DTCA_Info, DTCA_Prescribe, info_source_count.
    Self_Treat is intentionally EXCLUDED so it can be modelled separately.
    """
    df = df.copy()
    for item in DTCA_ITEMS:
        df[f"{item.lower()}_bin"] = yes_no(df[item])

    present_sources = [c for c in INFO_SOURCES if c in df.columns]
    if "Info_Sources_Count" in df.columns:
        df["info_source_count"] = pd.to_numeric(df["Info_Sources_Count"], errors="coerce").fillna(0).astype(int)
    elif present_sources:
        df["info_source_count"] = df[present_sources].sum(axis=1).astype(int)
    else:
        raise ValueError("No information-source columns found for HISB breadth.")

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
    df["NumOTC"] = pd.to_numeric(df["NumOTC"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    df["NumHerbal"] = pd.to_numeric(df["NumHerbal"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    if "NumOTC_Any" in df.columns:
        df["otc_use"] = pd.to_numeric(df["NumOTC_Any"], errors="coerce").fillna((df["NumOTC"] >= 1).astype(int)).astype(int)
    else:
        df["otc_use"] = (df["NumOTC"] >= 1).astype(int)
    if "NumHerbal_Any" in df.columns:
        df["herbal_use"] = pd.to_numeric(df["NumHerbal_Any"], errors="coerce").fillna((df["NumHerbal"] >= 1).astype(int)).astype(int)
    else:
        df["herbal_use"] = (df["NumHerbal"] >= 1).astype(int)

    for count_col, freq_col in (("NumOTC", "otc_freq"), ("NumHerbal", "herbal_freq")):
        binned = pd.cut(
            df[count_col].clip(upper=3),
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5],
            labels=FREQ_ORDER,
        )
        df[freq_col] = pd.Categorical(binned, categories=FREQ_ORDER, ordered=True)
    return df


def preprocess(raw_path: Path) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns one row per respondent."""
    df = load_raw(raw_path)
    df = df.drop_duplicates(subset=["respondent_id"])
    df = build_ses_index(df)
    df = add_self_treat_score(df)
    df = build_hisb_index(df)
    df = add_outcomes(df)
    return df.reset_index(drop=True)
