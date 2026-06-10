"""Smoke + sanity tests for the analysis pipeline.

These run on a small synthetic sample so they are fast and deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import (
    calibration,
    data_preprocessing,
    descriptive_analysis,
    generate_synthetic_data,
    mediation_analysis,
    multiple_imputation,
    ordinal_models,
    statistical_models,
    subgroup_analysis,
)

OTC = "otc_use"
OTC_FREQ = "otc_freq"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_synthetic_data.generate(n=800, seed=7)


@pytest.fixture(scope="module")
def analysis(raw: pd.DataFrame) -> pd.DataFrame:
    df = data_preprocessing.clean(raw)
    df["ses_tertile"] = pd.Categorical(
        df["ses_tertile"], categories=["Low", "Middle", "High"], ordered=True
    )
    return df


def test_generate_shape_and_columns(raw: pd.DataFrame) -> None:
    assert len(raw) == 800
    assert data_preprocessing.REQUIRED_COLUMNS <= set(raw.columns)


def test_outcomes_are_binary(analysis: pd.DataFrame) -> None:
    assert set(analysis["otc_use"].unique()) <= {0, 1}
    assert set(analysis["herbal_use"].unique()) <= {0, 1}


def test_generate_is_deterministic() -> None:
    a = generate_synthetic_data.generate(n=200, seed=123)
    b = generate_synthetic_data.generate(n=200, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_ses_index_built(analysis: pd.DataFrame) -> None:
    assert "ses_score" in analysis.columns
    assert set(analysis["ses_tertile"].dropna().unique()) <= {"Low", "Middle", "High"}
    assert abs(analysis["income_z"].mean()) < 1e-6


def test_hisb_composite_built(analysis: pd.DataFrame) -> None:
    # The HISB composite is a standardized mean of its components (~mean 0).
    assert "hisb_score" in analysis.columns
    assert abs(analysis["hisb_score"].mean()) < 1e-6
    assert analysis["info_source_count"].between(0, 10).all()


def test_descriptives_run(analysis: pd.DataFrame, tmp_path, monkeypatch) -> None:
    temp_paths = dataclasses.replace(descriptive_analysis.PATHS, tables_dir=tmp_path)
    monkeypatch.setattr(descriptive_analysis, "PATHS", temp_paths)
    out = descriptive_analysis.run(analysis)
    assert "bivariate_chi2" in out
    assert (tmp_path / "descriptives_continuous.csv").exists()


def test_logistic_model_recovers_ses_direction(analysis: pd.DataFrame) -> None:
    models = statistical_models.build_models(analysis, OTC)
    table = statistical_models.odds_ratio_table(models["model_adjusted"], "adj")
    # High SES (vs Low reference) should have OR < 1 (protective), per the DGP.
    high = table[table["term"].str.contains("High")]
    assert not high.empty
    assert high["odds_ratio"].iloc[0] < 1.0


def test_model_comparison_columns(analysis: pd.DataFrame) -> None:
    models = statistical_models.build_models(analysis, OTC)
    comp = statistical_models.comparison_table(models)
    assert {"model", "aic", "pseudo_r2_mcfadden"} <= set(comp.columns)
    assert np.isfinite(comp["aic"]).all()


def test_ordinal_model_or_table(analysis: pd.DataFrame) -> None:
    result = ordinal_models.fit(ordinal_models.prepare(analysis, OTC_FREQ), OTC_FREQ)
    table = ordinal_models.odds_ratio_table(result)
    assert len(table) == len(result._exog_columns)
    assert {"odds_ratio", "or_ci_low", "or_ci_high", "p_value"} <= set(table.columns)
    assert (table["odds_ratio"] > 0).all()


def test_ordinal_predicted_probs_sum_to_one(analysis: pd.DataFrame) -> None:
    data = ordinal_models.prepare(analysis, OTC_FREQ)
    result = ordinal_models.fit(data, OTC_FREQ)
    pred = ordinal_models.predicted_probs_by_ses(result, data)
    row_sums = pred[ordinal_models.FREQ_ORDER].sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_brant_test_structure(analysis: pd.DataFrame) -> None:
    brant = ordinal_models.brant_test(analysis, OTC_FREQ)
    assert brant.iloc[0]["variable"] == "Omnibus"
    assert {"X2", "df", "p_value"} <= set(brant.columns)
    assert ((brant["p_value"] >= 0) & (brant["p_value"] <= 1)).all()
    assert (brant["X2"] >= 0).all()


def test_mediation_paths_match_dgp(analysis: pd.DataFrame) -> None:
    point = mediation_analysis.fit_once(analysis, OTC)
    # DGP: higher SES -> more HISB (a>0); more HISB -> more use (b>0);
    # but SES protective directly (c'<0) => competitive mediation.
    assert point["a"] > 0
    assert point["b"] > 0
    assert point["c_prime"] < 0
    assert np.isclose(point["indirect"], point["a"] * point["b"])


def test_mediation_bootstrap_table(analysis: pd.DataFrame) -> None:
    point = mediation_analysis.fit_once(analysis, OTC)
    boot = mediation_analysis.bootstrap(analysis, OTC, n_boot=40, seed=1)
    table = mediation_analysis.results_table(point, boot)
    assert {"effect", "estimate", "ci_low", "ci_high"} <= set(table.columns)
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_partial_po_frees_flagged_term(analysis: pd.DataFrame) -> None:
    ppo = ordinal_models.fit_partial_po(analysis, OTC_FREQ, nonprop_terms=["hisb_score"])
    table = ppo["or_table"]
    assert {"term", "threshold", "odds_ratio", "p_value"} <= set(table.columns)
    freed = table[table["term"] == "Health info-seeking (per unit)"]
    assert len(freed) == len(ordinal_models.FREQ_ORDER) - 1
    assert (table["odds_ratio"] > 0).all()


def test_partial_po_improves_fit(analysis: pd.DataFrame) -> None:
    prop = ordinal_models.fit(ordinal_models.prepare(analysis, OTC_FREQ), OTC_FREQ)
    ppo = ordinal_models.fit_partial_po(analysis, OTC_FREQ, nonprop_terms=["hisb_score"])
    assert ppo["loglik"] >= prop.llf - 1e-3


def test_subgroup_stratified_and_interaction(analysis: pd.DataFrame) -> None:
    or_table = subgroup_analysis.stratified_or(analysis, OTC, "DTCA_Info")
    assert {"level", "term", "odds_ratio", "or_ci_low", "or_ci_high"} <= set(or_table.columns)
    assert (or_table["odds_ratio"] > 0).all()
    inter = subgroup_analysis.interaction_test(analysis, OTC, "DTCA_Info")
    assert 0.0 <= inter["p_value"] <= 1.0


def test_multiple_imputation_pooling(analysis: pd.DataFrame) -> None:
    frame = multiple_imputation._model_frame(analysis, OTC)
    missing = multiple_imputation.inject_missing(
        frame, rate=0.15, cols=["hisb_score"], seed=3
    )
    assert missing["hisb_score"].isna().any()
    pooled = multiple_imputation.multiple_imputation(missing, OTC, n_imp=3, n_burnin=2)
    assert {"term", "odds_ratio", "p_value"} <= set(pooled.columns)
    assert (pooled["odds_ratio"] > 0).all()


def test_calibration_metrics_ranges(analysis: pd.DataFrame) -> None:
    y, p_apparent, p_cv = calibration._predictions(analysis, OTC, n_splits=5, seed=0)
    disc = calibration.discrimination_table(y, p_apparent, p_cv)
    auc_cv = disc.loc[disc["metric"] == "ROC_AUC", "cross_validated"].iloc[0]
    assert 0.5 <= auc_cv <= 1.0
    hl = calibration.hosmer_lemeshow(y, p_cv, n_bins=10)
    assert 0.0 <= hl["p_value"].iloc[0] <= 1.0
