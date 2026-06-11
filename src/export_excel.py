"""Export all analysis tables to a single formatted Excel workbook.

Reads CSV outputs from ``results/tables/`` (current multi-outcome pipeline) and
writes ``results/Self_Medication_Analysis_Results.xlsx`` with one sheet per table,
plus overview and variable-dictionary sheets.

Run standalone::

    python -m src.export_excel

Or as the final stage of ``run_pipeline``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import model_spec
from .config import CONFIG, PATHS

OUTPUT_NAME = "Self_Medication_Analysis_Results.xlsx"

# Ordered list of (sheet_title, csv_stem_or_None, description).
# Stems without ``__outcome`` are shared; stems with ``{outcome}`` are per-outcome.
SHEET_PLAN: list[tuple[str, str | None, str]] = [
    ("Overview", None, "Study summary and key adjusted-model findings"),
    ("Variable Dictionary", None, "Variable definitions used in the analysis"),
    ("Descriptives - Continuous", "descriptives_continuous", "Summary statistics for continuous variables"),
    ("Descriptives - Categorical", "descriptives_categorical", "Frequencies and percentages"),
    ("Bivariate Chi-square", "bivariate_chi2", "Chi-square tests: predictors vs each outcome"),
    ("OTC Rate by SES", "otc_use_by_ses", "OTC use prevalence by SES tertile"),
    ("OTC Rate by HISB", "otc_use_by_hisb", "OTC use prevalence by HISB (high vs low)"),
    ("Herbal Rate by SES", "herbal_use_by_ses", "Herbal use prevalence by SES tertile"),
    ("Herbal Rate by HISB", "herbal_use_by_hisb", "Herbal use prevalence by HISB (high vs low)"),
]

OUTCOME_SHEETS: list[tuple[str, str, str]] = [
    ("Logistic - Crude SES", "model_crude_ses", "Outcome ~ SES tertile only"),
    ("Logistic - Adjusted", "model_adjusted", "Outcome ~ SES + HISB (adjusted odds ratios)"),
    ("Logistic - Interaction", "model_interaction", "Outcome ~ SES * HISB + covariates"),
    ("Model Comparison", "model_comparison", "AIC, pseudo-R2, LR tests vs crude model"),
    ("Ordinal - Frequency OR", "model_ordinal_freq", "Proportional-odds ORs for count frequency"),
    ("Ordinal - Thresholds", "model_ordinal_thresholds", "Estimated cutpoints between frequency levels"),
    ("Ordinal - Fit Stats", "model_ordinal_fit", "Ordinal model fit statistics"),
    ("Ordinal - Brant Test", "model_ordinal_brant", "Test of proportional-odds assumption"),
    ("Partial PO - OR", "model_partial_po", "Partial proportional-odds model (cutpoint-specific ORs)"),
    ("Partial PO - Fit", "model_partial_po_fit", "Partial PO model fit and freed terms"),
    ("Ordinal Pred by SES", "ordinal_pred_by_ses", "Predicted frequency probabilities by SES tertile"),
    ("Mediation", "mediation_results", "SES -> HISB -> outcome (bootstrap CIs)"),
    ("Subgroup - Stratified OR", "subgroup_or", "SES/HISB ORs within DTCA subgroups"),
    ("Subgroup - Interaction", "subgroup_interaction", "LR test for SES x subgroup effect modification"),
    ("MI - Pooled OR", "mi_pooled", "Multiple imputation pooled odds ratios (Rubin)"),
    ("MI vs Complete Case", "mi_vs_completecase", "Side-by-side MI vs listwise deletion"),
    ("Discrimination", "discrimination_metrics", "ROC-AUC, Brier score, log loss (apparent vs CV)"),
    ("Calibration Curve", "calibration_curve", "Binned observed vs predicted probabilities"),
    ("Hosmer-Lemeshow", "hosmer_lemeshow", "Calibration goodness-of-fit test"),
]


def _excel_path() -> Path:
    return PATHS.tables_dir.parent / OUTPUT_NAME


def _load_csv(stem: str) -> pd.DataFrame | None:
    path = PATHS.tables_dir / f"{stem}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _safe_sheet_name(prefix: str, suffix: str = "", max_len: int = 31) -> str:
    """Build an Excel-safe sheet name (max 31 characters)."""
    name = f"{prefix}{suffix}" if suffix else prefix
    for ch in "[]:*?/\\":
        name = name.replace(ch, "")
    return name[:max_len]


def _overview_rows() -> pd.DataFrame:
    """Build the overview / metadata sheet."""
    outcomes = model_spec.outcomes()
    rows = [
        ("Study", "Socioeconomic Status and Health Information Seeking as Determinants of Self-Medication Practices"),
        ("Authors", "Nurudeen Showole, MD; Suhila Sawesi, PhD"),
        ("Institution", "Grand Valley State University"),
        ("Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
        ("Random seed", CONFIG.get("seed", "")),
        ("", ""),
        ("Independent variables", ""),
        ("  SES", "Composite of Education + HouseIncome (tertiles: Low / Middle / High)"),
        ("  HISB", "Composite of Med7-9, DTCA_Info, DTCA_Prescribe, Self_Treat, Info_* source count"),
        ("", ""),
        ("Dependent variables (separate models)", ""),
    ]
    for oc in outcomes:
        rows.append(
            (f"  {oc['label']}", f"Binary: {oc['name']} from {oc['count_var']} >= 1; Ordinal: {oc['freq_var']}"),
        )
    rows.extend(
        [
            ("", ""),
            ("Primary analysis", "Multivariable logistic regression: outcome ~ SES tertile + HISB"),
            ("Secondary analysis", "Ordinal proportional-odds on frequency (None / One / Two / Three+)"),
            ("Mediation", "SES (ses_score) -> HISB (hisb_score) -> outcome; bootstrap B=1000"),
            ("Subgroup variables", ", ".join(CONFIG.get("subgroup", {}).get("variables", []))),
            ("Significance", "alpha = 0.05 (two-sided)"),
            ("", ""),
            ("Key findings (adjusted model)", ""),
        ]
    )
    for oc in outcomes:
        adj = _load_csv(f"model_adjusted__{oc['name']}")
        if adj is None or adj.empty:
            continue
        rows.append((f"--- {oc['label']} ---", ""))
        for _, r in adj.iterrows():
            if r.get("term") == "Intercept":
                continue
            term = str(r.get("term", ""))
            or_val = r.get("odds_ratio", "")
            lo = r.get("or_ci_low", "")
            hi = r.get("or_ci_high", "")
            p = r.get("p_value", "")
            rows.append(
                (
                    term,
                    f"OR = {or_val} (95% CI {lo}-{hi}), p = {p}",
                )
            )
    return pd.DataFrame(rows, columns=["Item", "Detail"])


def _variable_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Education", "Raw", "Highest education completed (ordinal)"),
            ("HouseIncome", "Raw", "Household income bracket (ordinal)"),
            ("Med7", "Raw", "Gather information before medicine decisions (1-5)"),
            ("Med8", "Raw", "Review information multiple times (1-5)"),
            ("Med9", "Raw", "Continue seeking information after decision (1-5)"),
            ("DTCA_Info", "Raw", "Sought info after prescription advertisement (Yes/No)"),
            ("DTCA_Prescribe", "Raw", "Asked physician for advertised drug (Yes/No)"),
            ("Self_Treat", "Raw", "Agreement with self-treating without prescription (1-5 Likert)"),
            ("Info_Google ... Info_Other", "Raw", "Information sources depended on (0/1 each)"),
            ("NumOTC", "Raw", "Number of OTC drugs taken daily"),
            ("NumHerbal", "Raw", "Number of herbal supplements taken daily"),
            ("ses_score", "Derived", "Standardized mean of education_z and income_z"),
            ("ses_tertile", "Derived", "Low / Middle / High (tertiles of ses_score)"),
            ("hisb_score", "Derived", "Standardized HISB composite (Med7-9, DTCA, Self_Treat, Info count)"),
            ("hisb_high", "Derived", "1 if hisb_score >= median"),
            ("otc_use", "Outcome", "1 if NumOTC >= 1"),
            ("herbal_use", "Outcome", "1 if NumHerbal >= 1"),
            ("otc_freq / herbal_freq", "Outcome", "None / One / Two / Three+ (from daily counts)"),
        ],
        columns=["Variable", "Type", "Description"],
    )


def _format_sheet(ws, title: str, description: str, header_row: int, ncols: int) -> None:
    """Apply title, description, and header styling (data already written below)."""
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=13, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="2B6CB0")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(ncols, 1))
    ws["A2"] = description
    ws["A2"].font = Font(italic=True, size=10)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(ncols, 1))

    header_fill = PatternFill("solid", fgColor="E2E8F0")
    for col in range(1, ncols + 1):
        cell = ws.cell(row=header_row, column=col)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for col in range(1, ncols + 1):
        letter = get_column_letter(col)
        max_len = 12
        for row in ws.iter_rows(min_row=header_row, max_row=ws.max_row, min_col=col, max_col=col):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, min(len(str(cell.value)) + 2, 50))
        ws.column_dimensions[letter].width = max_len

    ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate


def _write_table(
    writer,
    sheet_name: str,
    title: str,
    description: str,
    df: pd.DataFrame,
) -> None:
    """Write one formatted data sheet (title rows 1-2, header row 3, data from row 4)."""
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
    _format_sheet(writer.sheets[sheet_name], title, description, header_row=3, ncols=len(df.columns))


def export_excel(output_path: Path | None = None) -> Path:
    """Write the full analysis workbook and return its path."""
    PATHS.ensure_dirs()
    out = output_path or _excel_path()

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        overview = _overview_rows()
        _write_table(writer, "Overview", "Analysis Overview", "Study metadata and key adjusted-model results", overview)

        vardict = _variable_dictionary()
        _write_table(writer, "Variable Dictionary", "Variable Dictionary", "Raw and derived variables", vardict)

        for sheet_title, stem, desc in SHEET_PLAN[2:]:
            df = _load_csv(stem) if stem else None
            if df is None:
                continue
            name = _safe_sheet_name(sheet_title)
            _write_table(writer, name, sheet_title, desc, df)

        for oc in model_spec.outcomes():
            outcome, label = oc["name"], oc["label"]
            tag = "OTC" if outcome == "otc_use" else "Herb"
            for sheet_title, stem, desc in OUTCOME_SHEETS:
                df = _load_csv(f"{stem}__{outcome}")
                if df is None or df.empty:
                    continue
                full_title = f"{tag} - {sheet_title}"
                name = _safe_sheet_name(tag, f" {sheet_title[:22]}")
                _write_table(writer, name, full_title, f"{label}: {desc}", df)

        if PATHS.processed_data.exists():
            sample = pd.read_csv(PATHS.processed_data).head(500)
            _write_table(
                writer,
                "Data Sample",
                "Analysis Dataset (sample)",
                "First 500 rows of the processed analysis file",
                sample,
            )

    print(f"[excel] wrote {out}")
    return out


def main() -> Path:
    return export_excel()


if __name__ == "__main__":
    main()
