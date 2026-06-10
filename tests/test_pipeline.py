"""Smoke + sanity tests for the analysis pipeline.

These run on a small synthetic sample so they are fast and deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import (
    data_preprocessing,
    descriptive_analysis,
    generate_synthetic_data,
    mediation_analysis,
    ordinal_models,
    statistical_models,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_synthetic_data.generate(n=600, seed=7)


@pytest.fixture(scope="module")
def analysis(raw: pd.DataFrame) -> pd.DataFrame:
    df = data_preprocessing.clean(raw)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return df


def test_generate_shape_and_columns(raw: pd.DataFrame) -> None:
    assert len(raw) == 600
    assert data_preprocessing.REQUIRED_COLUMNS <= set(raw.columns)


def test_outcome_is_binary(raw: pd.DataFrame) -> None:
    assert set(raw["self_medication"].unique()) <= {0, 1}


def test_generate_is_deterministic() -> None:
    a = generate_synthetic_data.generate(n=200, seed=123)
    b = generate_synthetic_data.generate(n=200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_ses_index_built(analysis: pd.DataFrame) -> None:
    assert "ses_score" in analysis.columns
    assert set(analysis["ses_tertile"].dropna().unique()) <= {"Low", "Middle", "High"}
    # z-scored components should be ~mean 0.
    assert abs(analysis["income_z"].mean()) < 1e-6


def test_descriptives_run(analysis: pd.DataFrame, tmp_path, monkeypatch) -> None:
    # PATHS is a frozen dataclass, so swap the whole object with a temp-dir variant.
    temp_paths = dataclasses.replace(descriptive_analysis.PATHS, tables_dir=tmp_path)
    monkeypatch.setattr(descriptive_analysis, "PATHS", temp_paths)
    out = descriptive_analysis.run(analysis)
    assert "bivariate_chi2" in out
    assert (tmp_path / "descriptives_continuous.csv").exists()


def test_logistic_model_recovers_ses_direction(analysis: pd.DataFrame) -> None:
    models = statistical_models.build_models(analysis)
    table = statistical_models.odds_ratio_table(models["model_adjusted"], "adj")
    # High SES (vs Low reference) should have OR < 1 (protective), per the DGP.
    high = table[table["term"].str.contains("High")]
    assert not high.empty
    assert high["odds_ratio"].iloc[0] < 1.0


def test_model_comparison_columns(analysis: pd.DataFrame) -> None:
    models = statistical_models.build_models(analysis)
    comp = statistical_models.comparison_table(models)
    assert {"model", "aic", "pseudo_r2_mcfadden"} <= set(comp.columns)
    assert np.isfinite(comp["aic"]).all()


def test_ordinal_model_or_table(analysis: pd.DataFrame) -> None:
    result = ordinal_models.fit(ordinal_models.prepare(analysis))
    table = ordinal_models.odds_ratio_table(result)
    # Slopes only (no thresholds) -> one row per predictor term.
    assert len(table) == len(result._exog_columns)
    assert {"odds_ratio", "or_ci_low", "or_ci_high", "p_value"} <= set(table.columns)
    assert (table["odds_ratio"] > 0).all()


def test_ordinal_predicted_probs_sum_to_one(analysis: pd.DataFrame) -> None:
    data = ordinal_models.prepare(analysis)
    result = ordinal_models.fit(data)
    pred = ordinal_models.predicted_probs_by_ses(result, data)
    row_sums = pred[ordinal_models.FREQ_ORDER].sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_brant_test_structure(analysis: pd.DataFrame) -> None:
    brant = ordinal_models.brant_test(analysis)
    assert brant.iloc[0]["variable"] == "Omnibus"
    assert {"X2", "df", "p_value"} <= set(brant.columns)
    assert ((brant["p_value"] >= 0) & (brant["p_value"] <= 1)).all()
    assert (brant["X2"] >= 0).all()


def test_mediation_paths_match_dgp(analysis: pd.DataFrame) -> None:
    point = mediation_analysis.fit_once(analysis)
    # DGP: higher SES -> more HISB (a>0); more HISB -> more self-med (b>0);
    # but SES protective directly (c'<0) => competitive mediation.
    assert point["a"] > 0
    assert point["b"] > 0
    assert point["c_prime"] < 0
    assert np.isclose(point["indirect"], point["a"] * point["b"])


def test_mediation_bootstrap_table(analysis: pd.DataFrame) -> None:
    point = mediation_analysis.fit_once(analysis)
    boot = mediation_analysis.bootstrap(analysis, n_boot=40, seed=1)
    table = mediation_analysis.results_table(point, boot)
    assert {"effect", "estimate", "ci_low", "ci_high"} <= set(table.columns)
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_partial_po_frees_flagged_term(analysis: pd.DataFrame) -> None:
    # Force a known non-proportional term so the test is deterministic and fast.
    ppo = ordinal_models.fit_partial_po(analysis, nonprop_terms=["self_treat_score"])
    table = ppo["or_table"]
    assert {"term", "threshold", "odds_ratio", "p_value"} <= set(table.columns)
    # The freed term must have one row per cumulative cutpoint (K-1 = 3).
    freed = table[table["term"] == "Self-treat agreement (per point)"]
    assert len(freed) == len(ordinal_models.FREQ_ORDER) - 1
    assert (table["odds_ratio"] > 0).all()


def test_partial_po_improves_fit(analysis: pd.DataFrame) -> None:
    # Freeing a term cannot reduce the maximized log-likelihood vs. proportional odds.
    prop = ordinal_models.fit(ordinal_models.prepare(analysis))
    ppo = ordinal_models.fit_partial_po(analysis, nonprop_terms=["self_treat_score"])
    assert ppo["loglik"] >= prop.llf - 1e-3
