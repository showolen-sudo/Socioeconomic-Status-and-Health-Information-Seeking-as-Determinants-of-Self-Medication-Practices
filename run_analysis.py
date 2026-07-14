"""Run the manuscript analysis (descriptives + logistic regression).

Usage (from repo root):
    python run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.descriptives import (
    continuous_summary,
    prevalence_by_hisb,
    prevalence_by_self_treat,
    prevalence_by_ses,
    spearman_self_treat,
)
from src.export_excel import export_excel, save_csv_tables
from src.models import run_all_models
from src.preprocess import preprocess
from src.tables import (
    adjusted_or_long,
    combined_predictor_table,
    self_treat_effect,
    study_overview,
)


def load_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    return (ROOT / cfg["paths"][key]).resolve()


def main() -> None:
    cfg = load_config()
    raw_path = resolve(cfg, "raw_data")
    processed_path = resolve(cfg, "processed_data")
    tables_dir = resolve(cfg, "tables_dir")
    excel_path = resolve(cfg, "excel_output")
    outcomes = cfg["modelling"]["outcomes"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw survey file not found: {raw_path}\n"
            "Place survey_raw.csv under data/raw/ and re-run."
        )

    print(f"Loading and preprocessing: {raw_path}")
    df = preprocess(raw_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Wrote analysis dataset ({len(df)} rows): {processed_path}")

    print("Running descriptive tables...")
    tables = {
        "overview": study_overview(df),
        "descriptives": continuous_summary(df),
        "prev_ses": prevalence_by_ses(df),
        "prev_hisb": prevalence_by_hisb(df),
        "prev_self_treat": prevalence_by_self_treat(df),
        "spearman": spearman_self_treat(df, outcomes),
    }

    print("Fitting logistic models...")
    model_tables = run_all_models(df, outcomes)
    tables.update(model_tables)
    tables["combined_or"] = combined_predictor_table(model_tables)
    tables["adjusted_or_long"] = adjusted_or_long(model_tables, outcomes)
    tables["self_treat_effect"] = self_treat_effect(model_tables, outcomes)

    print(f"Writing CSV tables -> {tables_dir}")
    save_csv_tables(tables, tables_dir)
    print(f"Writing Excel workbook -> {excel_path}")
    export_excel(tables, excel_path)
    print("Done.")


if __name__ == "__main__":
    main()
