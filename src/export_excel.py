"""Export analysis tables to CSV files and a multi-sheet Excel workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


def _format_sheet(ws, title: str, description: str, ncols: int) -> None:
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = description
    ws["A2"].font = Font(italic=True, size=10, color="444444")
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF")
    for col in range(1, max(ncols, 1) + 1):
        cell = ws.cell(row=3, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "A4"
    for column_cells in ws.columns:
        length = 0
        col_letter = column_cells[0].column_letter
        for cell in column_cells:
            if cell.value is not None:
                length = max(length, min(len(str(cell.value)), 48))
        ws.column_dimensions[col_letter].width = max(12, length + 2)


SHEET_META = {
    "overview": ("Study overview", "Sample size and key crude prevalences."),
    "descriptives": ("Descriptive statistics", "Continuous summaries and category shares."),
    "prev_ses": ("Prevalence by SES", "Outcome prevalence stratified by SES tertile."),
    "prev_hisb": ("Prevalence by HISB", "Outcome prevalence by HISB median split."),
    "prev_self_treat": ("Prevalence by Self-Treat", "Outcome prevalence by Self_Treat Likert score."),
    "spearman": ("Spearman correlations", "Rank correlations of predictors with outcomes."),
    "combined_or": ("Primary adjusted ORs", "Multivariable logistic ORs (SES + HISB + Self_Treat_z)."),
    "adjusted_or_long": ("Adjusted ORs (long)", "Full adjusted model coefficients."),
    "self_treat_effect": ("Self-Treat effects", "Crude vs adjusted Self_Treat_z odds ratios."),
}


def save_csv_tables(tables: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        path = tables_dir / f"{name}.csv"
        df.to_csv(path, index=False)


def export_excel(tables: dict, excel_path: Path) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        preferred = list(SHEET_META.keys())
        ordered = [k for k in preferred if k in tables] + [
            k for k in tables if k not in preferred and isinstance(tables[k], pd.DataFrame)
        ]
        for name in ordered:
            df = tables[name]
            if not isinstance(df, pd.DataFrame):
                continue
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False, startrow=2)
            ws = writer.sheets[sheet]
            title, desc = SHEET_META.get(name, (name, "Analysis output table."))
            _format_sheet(ws, title, desc, ncols=len(df.columns))
