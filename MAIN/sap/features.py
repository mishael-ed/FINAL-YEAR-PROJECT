from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .config import CA_MAX_SCORE, EXAM_MAX_SCORE
from .schema import (
    BEHAVIOR_ORDER,
    CATEGORICAL_BASE,
    GENDER_ORDER,
    GRADE_ORDER,
    NUMERIC_BASE,
    OUTCOME_ORDER,
)

#the goal in this file is to take the raw excel sheet data and transform it into a clean, feature-rich format suitable for modeling. This involves:

def clean_data(df): 
    #to avoid us messing up with the original dataframe create a copy of it 

    data = df.copy() 

    #loop thru those columns and only keep them if they exist in the data 

    numeric_cols = [c for c in NUMERIC_BASE if c in data.columns]
    cat_cols = [c for c in CATEGORICAL_BASE if c in data.columns]

    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        if data[c].isna().all():
            data[c] = 0.0 #very important so that the median fitting doesnt fail when all values are missing

    if numeric_cols:
        data[numeric_cols] = SimpleImputer(strategy="median").fit_transform(data[numeric_cols])

    if cat_cols:
        for c in cat_cols:
            if data[c].isna().all():
                data[c] = "Unknown"
        data[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(data[cat_cols])

    data["Gender Enc"] = data["Gender"].map(GENDER_ORDER).fillna(0) if "Gender" in data.columns else 0
    data["Behavioral Enc"] = data["Behavioral Rating"].map(BEHAVIOR_ORDER).fillna(2) if "Behavioral Rating" in data.columns else 2
    data["Grade Enc"] = data["Grade"].map(GRADE_ORDER).fillna(3) if "Grade" in data.columns else 3
    data["Outcome Enc"] = data["Final Outcome"].map(OUTCOME_ORDER).fillna(1) if "Final Outcome" in data.columns else 1

    term_num = (
        pd.to_numeric(data["Term"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
        if "Term" in data.columns
        else pd.Series([0] * len(data))
    )
    data["Term Num"] = term_num.fillna(0)

    # Derive attendance percent if not explicitly provided.
    if "Attendance %" not in data.columns and {"Classes Held", "Classes Attended"}.issubset(data.columns):
        held = pd.to_numeric(data["Classes Held"], errors="coerce").replace(0, np.nan)
        att = pd.to_numeric(data["Classes Attended"], errors="coerce")
        data["Attendance %"] = ((att / held) * 100.0).fillna(0.0)

    if "CA Score" in data.columns:
        if "CA Total Score" in data.columns:
            ca_total = pd.to_numeric(data["CA Total Score"], errors="coerce")
            ca_total = ca_total.where(ca_total > 0, CA_MAX_SCORE).fillna(CA_MAX_SCORE)
            data["CA %"] = (data["CA Score"] / ca_total) * 100.0
        else:
            data["CA %"] = (data["CA Score"] / CA_MAX_SCORE) * 100.0
    else:
        data["CA %"] = 0.0

    if "Exam Score (%)" in data.columns:
        data["Exam %"] = pd.to_numeric(data["Exam Score (%)"], errors="coerce").fillna(0.0)
    elif "Exam Score" in data.columns:
        data["Exam %"] = (pd.to_numeric(data["Exam Score"], errors="coerce").fillna(0.0) / EXAM_MAX_SCORE) * 100.0
    else:
        data["Exam %"] = 0.0

  
    has_exam_signal = data["Exam %"] > 0
    data["Overall %"] = np.where(has_exam_signal, (data["CA %"] * 0.6) + (data["Exam %"] * 0.4), data["CA %"])

    return data


def build_term_features(df):
    rows = []
    for (student_id, term), g in df.groupby(["Student ID", "Term"], dropna=False):
        row = {
            "Student ID": student_id,
            "Student Name": g["Student Name"].iloc[0] if "Student Name" in g.columns else str(student_id),
            "Term": term,
            "Class": g["Class"].iloc[0],
            "Gender Enc": float(g["Gender Enc"].iloc[0]),
            "Term Num": float(g["Term Num"].iloc[0]),
            "avg_ca": float(g["CA Score"].mean()),
            "avg_ca_pct": float(g["CA %"].mean()),
            "avg_exam_pct": float(g["Exam %"].mean()),
            "avg_overall_pct": float(g["Overall %"].mean()),
            "avg_attendance": float(g["Attendance %"].mean()),
            "min_attendance": float(g["Attendance %"].min()),
            "avg_behavior": float(g["Behavioral Enc"].mean()),
            "subject_count": int(g["Subject"].nunique()),
            # Keep this feature available for both train/predict without target leakage:
            # infer likely failed subjects from performance signals, not Final Outcome.
            "subject_failure_count": int((g["CA %"] < 50).sum()),
            "at_risk": int((g["Outcome Enc"] == 0).any()) if "Outcome Enc" in g.columns else 0,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["Student ID", "Term Num", "Term"])
    out["attendance_trend"] = out.groupby("Student ID")["avg_attendance"].transform(lambda s: s - s.iloc[0])
    out["ca_trend"] = out.groupby("Student ID")["avg_ca_pct"].transform(lambda s: s - s.iloc[0])
    # predict forward: current-term features -> next-term risk.
    out["target_next_term_risk"] = out.groupby("Student ID")["at_risk"].shift(-1)
    return out


RF_FEATURES = [
    "Gender Enc",
    "avg_ca_pct",
    "avg_overall_pct",
    "avg_attendance",
    "min_attendance",
    "avg_behavior",
    "subject_count",
    "subject_failure_count",
    "attendance_trend",
    "ca_trend",
]


def build_rf_xy(term_df: pd.DataFrame):
    x = term_df[RF_FEATURES].copy()
    if "target_next_term_risk" in term_df.columns:
        y = term_df["target_next_term_risk"]
    else:
        y = term_df["at_risk"]
    y = y.fillna(0).astype(int)
    return x, y
