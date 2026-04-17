from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sap.db import (
    fetch_imported_records,
    get_db_filter_options,
    save_imported_records,
    save_prediction_results,
    test_connection,
)
from sap.features import build_term_features, clean_data
from sap.io import load_table, prepare
from sap.model import infer
from sap.reporting import export_csv, export_pdf


st.set_page_config(page_title="AI Performance Predictor", layout="wide")

st.markdown(
    """
<style>
.stApp {
  background: radial-gradient(circle at top left, #1b1e2b 0%, #0f1117 45%);
  color: #f3f4f8;
}
.stDeployButton { display: none !important; }
[data-testid="stDeployButton"] { display: none !important; }
button[kind="deployButton"] { display: none !important; }
footer { visibility: hidden; }
.block-container { max-width: 1180px; }
.hero {
  background: linear-gradient(120deg, #2b1055, #5b2aa8);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  color: white;
  margin-bottom: .8rem;
}
div[data-testid="stMetric"] {
  background: #171a23;
  border: 1px solid #30364a;
  border-radius: 10px;
  padding: .5rem .75rem;
}
/* Add label to sidebar toggle */
[data-testid="collapsedControl"]::before {
  content: "DATABASE INTEGRATION";
  font-size: 11px;
  color: #888;
  margin-right: 4px;
  font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h2 style="margin:0;">AI-Powered Student Academic Performance Predictor</h2>
  <p style="margin:.35rem 0 0 0;opacity:.9;">Admin mode: upload intra-semester records and get risk predictions instantly.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.info("Model is pre-trained by the project team. Admins only need to upload prediction data.")


if "db_url" not in st.session_state:
    secret_url = ""
    try:
        secret_url = st.secrets.get("MYSQL_URL", "")
    except Exception:
        secret_url = ""
    st.session_state["db_url"] = secret_url or os.getenv("MYSQL_URL", "")
if "auto_save_mysql" not in st.session_state:
    st.session_state["auto_save_mysql"] = True


with st.sidebar:
    st.markdown("### DATABASE INTEGRATION")
    db_url = st.text_input(
        "MySQL SQLAlchemy URL",
        value=st.session_state["db_url"],
        type="password",
        placeholder="mysql+pymysql://user:password@host:3306/database",
        help="Used to save uploaded records and prediction outputs, and to run predictions from DB data.",
    )
    st.session_state["db_url"] = db_url.strip()
    st.session_state["auto_save_mysql"] = st.checkbox(
        "Auto-save upload/manual records + prediction runs",
        value=st.session_state["auto_save_mysql"],
    )
    if st.button("Test MySQL Connection"):
        if not st.session_state["db_url"]:
            st.warning("Please enter a MySQL URL first.")
        else:
            try:
                test_connection(st.session_state["db_url"])
                st.success("MySQL is reachable and tables are ready.")
            except Exception as e:
                st.error(f"MySQL connection failed: {e}")


def add_issue_reason(df: pd.DataFrame) -> pd.DataFrame:
    def reason(r: pd.Series) -> str:
        scored_issues = []

        ca_pct = float(r.get("avg_ca_pct", 100))
        att_pct = float(r.get("avg_attendance", 100))
        overall_pct = float(r.get("avg_overall_pct", 100))
        fail_count = int(r.get("subject_failure_count", 0))

        # Score each risk factor by severity so the most critical appears first.
        if ca_pct < 55:
            scored_issues.append(("Low CA", 55 - ca_pct))
        if att_pct < 75:
            scored_issues.append(("Low attendance", 75 - att_pct))
        if overall_pct < 50:
            scored_issues.append(("Low performance", 50 - overall_pct))
        if fail_count >= 3:
            scored_issues.append(("Many failed subjects", (fail_count - 2) * 5))

        if not scored_issues:
            return "No major risk"

        scored_issues.sort(key=lambda x: x[1], reverse=True)
        top_labels = [label for label, _ in scored_issues[:2]]
        return ", ".join(top_labels)

    out = df.copy()
    out["main_issue"] = out.apply(reason, axis=1)
    return out


def present(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Student Name" not in out.columns:
        out["Student Name"] = out["Student ID"].astype(str)
    if "avg_overall_pct" not in out.columns:
        out["avg_overall_pct"] = out.get("avg_ca_pct", 0.0)
    out["Likelihood of Passing (%)"] = (out["pass_probability"] * 100).round(1)
    out["Current CA Progress (%)"] = out["avg_ca_pct"].round(1)
    out["Current Performance (%)"] = out["avg_overall_pct"].round(1)
    out["Risk Level"] = out["risk_tier"]
    out["Predicted Outcome"] = out["predicted_outcome"]
    out["Historical Failed Subjects"] = out["subject_failure_count"].astype(int)
    return out[
        [
            "Student ID",
            "Student Name",
            "Term",
            "Class",
            "Predicted Outcome",
            "Likelihood of Passing (%)",
            "Risk Level",
            "Current CA Progress (%)",
            "Current Performance (%)",
            "Historical Failed Subjects",
            "main_issue",
        ]
    ]


def maybe_save_to_mysql(raw_df: pd.DataFrame | None, preds: pd.DataFrame, source_label: str, save_raw: bool) -> None:
    db_url = st.session_state.get("db_url", "").strip()
    auto_save = st.session_state.get("auto_save_mysql", False)
    if not db_url or not auto_save:
        return

    try:
        if save_raw and raw_df is not None and len(raw_df) > 0:
            saved_rows = save_imported_records(raw_df, db_url, source_label=source_label)
            st.caption(f"Saved {saved_rows:,} imported records to MySQL.")
        run_id = save_prediction_results(preds, db_url, source_label=source_label)
        st.caption(f"Saved prediction run to MySQL (run_id={run_id}).")
    except Exception as e:
        st.warning(f"MySQL save skipped due to error: {e}")


def run_prediction_pipeline(raw_df: pd.DataFrame, source_label: str, save_raw: bool) -> None:
    low_min = 0.75
    med_min = 0.50

    prepared = prepare(raw_df, mode="predict")
    clean = clean_data(prepared)
    term_df = build_term_features(clean)
    preds = infer(term_df, thresholds={"low": low_min, "medium": med_min})
    preds = add_issue_reason(preds)
    st.session_state["preds"] = preds
    st.session_state["pred_source"] = source_label
    
    # Show which models were used
    models_used = preds["prediction_models"].iloc[0] if len(preds) > 0 and "prediction_models" in preds.columns else "Ensemble"
    st.success(f"Generated predictions for {len(preds):,} student-term records using {models_used}")

    maybe_save_to_mysql(raw_df=raw_df, preds=preds, source_label=source_label, save_raw=save_raw)


st.markdown("### Step 1: Run Intra-semester Predictions (no Final Outcome required)")
st.caption("Predictions use an ensemble of all trained models: Random Forest + XGBoost + LSTM for maximum accuracy")
pred_file = st.file_uploader("Upload Prediction File (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="pred")

st.markdown("#### Or Enter Student Records Manually")
st.caption("Use this if you do not have a CSV/Excel file. Add one row per student-subject record.")
manual_seed = pd.DataFrame(
    [
        {
            "Student ID": "STU_1",
            "Student Name": "Mishael Edegwa",
            "Gender": "Male",
            "Term": "2026_T1",
            "Class": "SS2",
            "Subject": "Mathematics",
            "CA Score": 35,
            "CA Total Score": 40,
            "Classes Held": 30,
            "Classes Attended": 27,
            "Behavioral Rating": "Average",
        }
    ]
)
manual_df = st.data_editor(
    manual_seed,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Behavioral Rating": st.column_config.SelectboxColumn(
            "Behavioral Rating",
            options=["Poor", "Average", "Good", "Excellent"],
            required=True,
        ),
    },
    key="manual_entry_table",
)

if pred_file is not None and st.button("Run Predictions"):
    suffix = "." + pred_file.name.split(".")[-1].lower()
    temp = Path(f"_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}")
    with open(temp, "wb") as f:
        f.write(pred_file.getbuffer())

    try:
        df = load_table(temp)
        run_prediction_pipeline(df, source_label="upload", save_raw=True)
    except Exception as e:
        st.error(
            "Prediction failed. Ensure the model is pre-trained and saved. "
            "If needed, run `python train.py --input <training_file>` once."
        )
        st.caption(f"Technical detail: {e}")
    finally:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass

if st.button("Run Predictions From Manual Entry"):
    try:
        entry_df = manual_df.copy()
        entry_df = entry_df.dropna(how="all")
        if len(entry_df) == 0:
            st.warning("Please enter at least one row before running predictions.")
        else:
            # Drop blank student IDs to avoid accidental empty rows from dynamic editor.
            entry_df["Student ID"] = entry_df["Student ID"].astype(str).str.strip()
            entry_df = entry_df[entry_df["Student ID"] != ""]
            if len(entry_df) == 0:
                st.warning("Please provide Student ID values for manual entries.")
            else:
                run_prediction_pipeline(entry_df, source_label="manual", save_raw=True)
    except Exception as e:
        st.error("Manual prediction failed. Please check required fields in the table.")
        st.caption(f"Technical detail: {e}")

st.markdown("### Step 1B: Run Predictions From MySQL Records")
if not st.session_state.get("db_url", "").strip():
    st.caption("Add a MySQL URL in the sidebar, then run predictions directly from DB records.")
else:
    try:
        terms, classes = get_db_filter_options(st.session_state["db_url"])
        c1, c2, c3 = st.columns(3)
        chosen_term = c1.selectbox("DB Term", ["All"] + terms)
        chosen_class = c2.selectbox("DB Class", ["All"] + classes)
        db_limit = int(c3.number_input("DB Rows To Read", min_value=100, max_value=20000, value=5000, step=100))

        if st.button("Run Predictions From MySQL"):
            db_df = fetch_imported_records(
                st.session_state["db_url"],
                limit=db_limit,
                term=None if chosen_term == "All" else chosen_term,
                class_name=None if chosen_class == "All" else chosen_class,
            )
            if db_df.empty:
                st.warning("No records found in MySQL for the selected filters.")
            else:
                run_prediction_pipeline(db_df, source_label="mysql_records", save_raw=False)

        with st.expander("Preview Latest Imported Records In MySQL"):
            preview_df = fetch_imported_records(st.session_state["db_url"], limit=100)
            st.dataframe(preview_df, use_container_width=True)
            st.caption("This preview is read live from MySQL, so edits made in the DB are reflected on refresh.")
    except Exception as e:
        st.warning(f"MySQL section unavailable: {e}")

if "preds" in st.session_state:
    preds = st.session_state["preds"].copy()

    st.markdown("### Step 2: Review Risk List")
    source_label = st.session_state.get("pred_source", "current session")
    st.caption(f"Prediction source: {source_label}")

    quick = st.selectbox(
        "Quick Filter",
        [
            "All Students",
            "Students At Risk of Failing",
            "Students At Risk of Passing",
        ],
    )

    c1, c2, c3 = st.columns(3)
    class_filter = c1.multiselect("Class", sorted(preds["Class"].dropna().unique()))
    term_filter = c2.multiselect("Term", sorted(preds["Term"].dropna().unique()))
    sid_filter = c3.text_input("Student ID")

    filt = preds.copy()
    if quick == "Students At Risk of Failing":
        filt = filt[(filt["predicted_outcome"] == "Fail") | (filt["risk_tier"] == "High")]
    elif quick == "Students At Risk of Passing":
        filt = filt[(filt["predicted_outcome"] == "Pass") & (filt["risk_tier"].isin(["Medium", "High"]))]

    if class_filter:
        filt = filt[filt["Class"].isin(class_filter)]
    if term_filter:
        filt = filt[filt["Term"].isin(term_filter)]
    if sid_filter:
        filt = filt[filt["Student ID"].str.contains(sid_filter, case=False, na=False)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("High Risk", int((filt["risk_tier"] == "High").sum()))
    m2.metric("Medium Risk", int((filt["risk_tier"] == "Medium").sum()))
    m3.metric("Low Risk", int((filt["risk_tier"] == "Low").sum()))
    m4.metric("Avg Passing Likelihood", f"{(filt['pass_probability'].mean() * 100 if len(filt) else 0):.1f}%")

    view = present(filt)
    st.dataframe(view, use_container_width=True)

    report_name = f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = export_csv(view, report_name)
    pdf_path = export_pdf(view, report_name)

    d1, d2 = st.columns(2)
    with d1:
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV Report", data=f, file_name=csv_path.name, mime="text/csv")
    with d2:
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", data=f, file_name=pdf_path.name, mime="application/pdf")
