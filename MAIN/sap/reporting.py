from __future__ import annotations

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from .config import REPORT_DIR


def export_csv(df: pd.DataFrame, name: str) -> Path:
    path = REPORT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def export_pdf(df: pd.DataFrame, name: str) -> Path:
    path = REPORT_DIR / f"{name}.pdf"
    preview = df.copy().astype(str)
    data = [preview.columns.tolist()] + preview.values.tolist()

    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4))
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b1055")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    doc.build([table])
    return path
