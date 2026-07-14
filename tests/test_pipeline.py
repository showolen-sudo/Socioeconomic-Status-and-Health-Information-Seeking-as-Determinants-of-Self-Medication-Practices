"""Smoke test for manuscript preprocess + descriptives + logit pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.descriptives import continuous_summary, prevalence_by_ses
from src.models import run_all_models
from src.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_smoke(tmp_path: Path):
    raw = ROOT / "data" / "raw" / "survey_raw.csv"
    if not raw.exists():
        from src.constants import INFO_SOURCES, MED_ITEMS

        n = 60
        data = {
            "respondent_id": list(range(n)),
            "Education": [((i % 7) + 1) for i in range(n)],
            "HouseIncome": [((i % 8) + 1) for i in range(n)],
            "Self_Treat": [((i % 7) + 1) for i in range(n)],
            "NumOTC": [i % 4 for i in range(n)],
            "NumHerbal": [(i + 1) % 3 for i in range(n)],
            "DTCA_Info": [1 if i % 2 == 0 else 2 for i in range(n)],
            "DTCA_Prescribe": [1 if i % 3 == 0 else 2 for i in range(n)],
        }
        for m in MED_ITEMS:
            data[m] = [((i % 5) + 1) for i in range(n)]
        for c in INFO_SOURCES:
            data[c] = [i % 2 for i in range(n)]
        raw = tmp_path / "survey_raw.csv"
        pd.DataFrame(data).to_csv(raw, index=False)

    df = preprocess(raw)
    assert len(df) >= 30
    assert "hisb_score" in df.columns
    assert "self_treat_z" in df.columns
    assert "otc_use" in df.columns
    assert df["hisb_score"].notna().all()

    desc = continuous_summary(df)
    assert not desc.empty
    prev = prevalence_by_ses(df)
    assert {"ses_tertile", "otc_use_pct", "herbal_use_pct"} <= set(prev.columns)
    assert set(prev["ses_tertile"].astype(str)) >= {"Low", "Middle", "High"}

    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        outcomes = yaml.safe_load(f)["modelling"]["outcomes"]
    tables = run_all_models(df, outcomes)
    assert "adjusted__otc_use" in tables
    assert "odds_ratio" in tables["adjusted__otc_use"].columns
