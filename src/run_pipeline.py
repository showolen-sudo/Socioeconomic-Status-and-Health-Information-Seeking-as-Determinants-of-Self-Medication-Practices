"""End-to-end analysis pipeline orchestrator.

Run with::

    python -m src.run_pipeline               # generate synthetic data + full analysis
    python -m src.run_pipeline --no-generate  # use existing data/raw/survey_raw.csv

Stages: generate -> preprocess -> descriptives -> models -> figures.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from . import (
    data_preprocessing,
    descriptive_analysis,
    generate_synthetic_data,
    mediation_analysis,
    ordinal_models,
    statistical_models,
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
        print(f"[1/7] Using existing raw data: {PATHS.raw_data}")
    else:
        print("[1/7] Generating synthetic data ...")
        generate_synthetic_data.main()

    # 2. Preprocess
    print("[2/7] Preprocessing ...")
    analysis = data_preprocessing.main()
    analysis["ses_tertile"] = pd.Categorical(
        analysis["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )

    # 3. Descriptives
    print("[3/7] Descriptive & bivariate analysis ...")
    descriptive_analysis.run(analysis)

    # 4. Models
    print("[4/7] Fitting logistic-regression models ...")
    models = statistical_models.run(analysis)

    # 5. Ordinal model on self-medication frequency (+ Brant test)
    print("[5/7] Fitting ordinal (proportional-odds) model + Brant test ...")
    ordinal_models.run(analysis)

    # 6. Mediation: SES -> HISB -> self-medication
    print("[6/7] Running mediation analysis (bootstrap) ...")
    mediation_analysis.run(analysis)

    # 7. Figures
    print("[7/7] Rendering figures ...")
    visualization.run(analysis, adjusted_model=models["model_adjusted"])

    elapsed = time.perf_counter() - start
    print("=" * 70)
    print(f"Done in {elapsed:.1f}s.")
    print(f"  Tables : {PATHS.tables_dir}")
    print(f"  Figures: {PATHS.figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
