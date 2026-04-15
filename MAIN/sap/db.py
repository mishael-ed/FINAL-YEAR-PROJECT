from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .io import normalize_columns


RAW_COLUMNS = [
    "Student ID",
    "Student Name",
    "Gender",
    "Term",
    "Class",
    "Subject",
    "CA Score",
    "CA Total Score",
    "Exam Score",
    "Exam Score (%)",
    "Grade",
    "Classes Held",
    "Classes Attended",
    "Attendance %",
    "Behavioral Rating",
    "Final Outcome",
]

RAW_RENAME = {
    "Student ID": "student_id",
    "Student Name": "student_name",
    "Gender": "gender",
    "Term": "term",
    "Class": "class_name",
    "Subject": "subject",
    "CA Score": "ca_score",
    "CA Total Score": "ca_total_score",
    "Exam Score": "exam_score",
    "Exam Score (%)": "exam_score_pct",
    "Grade": "grade",
    "Classes Held": "classes_held",
    "Classes Attended": "classes_attended",
    "Attendance %": "attendance_pct",
    "Behavioral Rating": "behavioral_rating",
    "Final Outcome": "final_outcome",
}

RAW_FROM_DB = {v: k for k, v in RAW_RENAME.items()}

PRED_COLUMNS = [
    "Student ID",
    "Student Name",
    "Term",
    "Class",
    "avg_ca_pct",
    "avg_overall_pct",
    "avg_attendance",
    "subject_failure_count",
    "pass_probability",
    "predicted_outcome",
    "risk_tier",
    "main_issue",
]

PRED_RENAME = {
    "Student ID": "student_id",
    "Student Name": "student_name",
    "Term": "term",
    "Class": "class_name",
    "avg_ca_pct": "avg_ca_pct",
    "avg_overall_pct": "avg_overall_pct",
    "avg_attendance": "avg_attendance",
    "subject_failure_count": "subject_failure_count",
    "pass_probability": "pass_probability",
    "predicted_outcome": "predicted_outcome",
    "risk_tier": "risk_tier",
    "main_issue": "main_issue",
}


def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True)


def ensure_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS imported_records (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    student_id VARCHAR(128),
                    student_name VARCHAR(255),
                    gender VARCHAR(32),
                    term VARCHAR(64),
                    class_name VARCHAR(64),
                    subject VARCHAR(128),
                    ca_score DOUBLE,
                    ca_total_score DOUBLE,
                    exam_score DOUBLE,
                    exam_score_pct DOUBLE,
                    grade VARCHAR(16),
                    classes_held DOUBLE,
                    classes_attended DOUBLE,
                    attendance_pct DOUBLE,
                    behavioral_rating VARCHAR(32),
                    final_outcome VARCHAR(16),
                    source_label VARCHAR(64),
                    ingested_at DATETIME
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(64),
                    student_id VARCHAR(128),
                    student_name VARCHAR(255),
                    term VARCHAR(64),
                    class_name VARCHAR(64),
                    avg_ca_pct DOUBLE,
                    avg_overall_pct DOUBLE,
                    avg_attendance DOUBLE,
                    subject_failure_count INT,
                    pass_probability DOUBLE,
                    predicted_outcome VARCHAR(16),
                    risk_tier VARCHAR(16),
                    main_issue VARCHAR(255),
                    source_label VARCHAR(64),
                    created_at DATETIME
                )
                """
            )
        )


def test_connection(db_url: str) -> None:
    engine = get_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    ensure_tables(engine)


def _normalize_raw_for_db(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_columns(df).copy()
    for col in RAW_COLUMNS:
        if col not in work.columns:
            work[col] = None
    out = work[RAW_COLUMNS].rename(columns=RAW_RENAME)
    out = out.where(pd.notnull(out), None)
    return out


def save_imported_records(df: pd.DataFrame, db_url: str, source_label: str) -> int:
    engine = get_engine(db_url)
    ensure_tables(engine)
    payload = _normalize_raw_for_db(df)
    payload["source_label"] = source_label
    payload["ingested_at"] = datetime.utcnow()
    payload.to_sql("imported_records", con=engine, if_exists="append", index=False)
    return int(len(payload))


def fetch_imported_records(
    db_url: str,
    limit: int = 5000,
    term: str | None = None,
    class_name: str | None = None,
) -> pd.DataFrame:
    engine = get_engine(db_url)
    ensure_tables(engine)

    query = """
        SELECT *
        FROM imported_records
        WHERE 1=1
    """
    params: dict[str, object] = {}
    if term:
        query += " AND term = :term"
        params["term"] = term
    if class_name:
        query += " AND class_name = :class_name"
        params["class_name"] = class_name
    query += " ORDER BY id DESC LIMIT :limit"
    params["limit"] = int(limit)

    with engine.connect() as conn:
        db_df = pd.read_sql(text(query), conn, params=params)

    if db_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    db_df = db_df.sort_values("id", ascending=True).copy()
    out = db_df.rename(columns=RAW_FROM_DB)
    for col in RAW_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[RAW_COLUMNS].copy()


def get_db_filter_options(db_url: str) -> tuple[list[str], list[str]]:
    engine = get_engine(db_url)
    ensure_tables(engine)
    with engine.connect() as conn:
        terms_df = pd.read_sql(
            text("SELECT DISTINCT term FROM imported_records WHERE term IS NOT NULL ORDER BY term"),
            conn,
        )
        classes_df = pd.read_sql(
            text("SELECT DISTINCT class_name FROM imported_records WHERE class_name IS NOT NULL ORDER BY class_name"),
            conn,
        )
    terms = [str(x) for x in terms_df["term"].dropna().tolist()]
    classes = [str(x) for x in classes_df["class_name"].dropna().tolist()]
    return terms, classes


def save_prediction_results(df: pd.DataFrame, db_url: str, source_label: str, run_id: str | None = None) -> str:
    engine = get_engine(db_url)
    ensure_tables(engine)
    rid = run_id or uuid4().hex[:16]

    work = df.copy()
    for col in PRED_COLUMNS:
        if col not in work.columns:
            work[col] = None
    payload = work[PRED_COLUMNS].rename(columns=PRED_RENAME)
    payload["run_id"] = rid
    payload["source_label"] = source_label
    payload["created_at"] = datetime.utcnow()
    payload = payload.where(pd.notnull(payload), None)
    payload.to_sql("prediction_runs", con=engine, if_exists="append", index=False)
    return rid
