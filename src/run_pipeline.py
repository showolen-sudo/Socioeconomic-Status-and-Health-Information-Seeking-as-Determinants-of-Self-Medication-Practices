"""End-to-end analysis pipeline orchestrator.

Run with::

    python -m src.run_pipeline               # generate synthetic data + full analysis
    python -m src.run_pipeline --no-generate  # use existing data/raw/survey_raw.csv

Stages: generate -> preprocess -> descriptives -> binary models -> ordinal/PPO ->
mediation -> subgroup -> multiple imputation -> calibration -> figures -> Excel export.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from . import (
    calibration,
    data_preprocessing,
    descriptive_analysis,
    export_excel,
    generate_synthetic_data,
    mediation_analysis,
    multiple_imputation,
    ordinal_models,
    statistical_models,
    subgroup_analysis,
    visualization,
)
from .config import PATHS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-medication analysis pipeline.")
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip synthetic data generation and use the existing raw data file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    start = time.perf_counter()
    PATHS.ensure_dirs()

    print("=" * 70)
    print("Self-medication / SES / HISB analysis pipeline")
    print("=" * 70)

    # 1. Data
    if args.no_generate:
        if not PATHS.raw_data.exists():
            raise FileNotFoundError(
                f"--no-generate set but {PATHS.raw_data} does not exist. "
                "Provide your survey data or run without --no-generate."
            )
        print(f"[1/11] Using existing raw data: {PATHS.raw_data}")
    else:
        print("[1/11] Generating synthetic data ...")
        generate_synthetic_data.main()

    # 2. Preprocess
    print("[2/11] Preprocessing ...")
    analysis = data_preprocessing.main()
    analysis["ses_tertile"] = pd.Categorical(
        analysis["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )

    # 3. Descriptives
    print("[3/11] Descriptive & bivariate analysis ...")
    descriptive_analysis.run(analysis)

    # 4. Models
    print("[4/11] Fitting logistic-regression models ...")
    models = statistical_models.run(analysis)

    # 5. Ordinal model on self-medication frequency (+ Brant test + partial PO)
    print("[5/11] Fitting ordinal model + Brant test + partial proportional odds ...")
    ordinal_models.run(analysis)

    # 6. Mediation: SES -> HISB -> self-medication
    print("[6/11] Running mediation analysis (bootstrap) ...")
    mediation_analysis.run(analysis)

    # 7. Subgroup / effect-modification analysis
    print("[7/11] Running subgroup analysis ...")
    subgroup_analysis.run(analysis)

    # 8. Multiple imputation for missing data
    print("[8/11] Running multiple imputation (MICE) ...")
    multiple_imputation.run(analysis)

    # 9. Calibration & discrimination metrics
    print("[9/11] Computing calibration & discrimination metrics ...")
    calibration.run(analysis)

    # 10. Figures
    print("[10/11] Rendering figures ...")
    visualization.run(analysis, models=models)

    # 11. Excel workbook (all tables + overview)
    print("[11/11] Exporting Excel workbook ...")
    excel_path = export_excel.export_excel()

    elapsed = time.perf_counter() - start
    print("=" * 70)
    print(f"Done in {elapsed:.1f}s.")
    print(f"  Tables : {PATHS.tables_dir}")
    print(f"  Figures: {PATHS.figures_dir}")
    print(f"  Excel  : {excel_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
