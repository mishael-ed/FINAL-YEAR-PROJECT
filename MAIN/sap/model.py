from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import DEFAULT_THRESHOLDS, MODEL_DIR
from .features import RF_FEATURES, build_rf_xy


@dataclass
class TrainResult:
    metrics: dict
    feature_importance: pd.DataFrame


def train_rf(term_df: pd.DataFrame) -> TrainResult:
    train_df = term_df.copy()
    if "target_next_term_risk" in train_df.columns:
        train_df = train_df[train_df["target_next_term_risk"].notna()].copy()

    x, y = build_rf_xy(train_df)
    groups = train_df["Student ID"] if "Student ID" in train_df.columns else None

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
    )
    warnings: list[str] = []

    # out-of-fold grouped CV  for robust metrics
    if groups is not None and groups.nunique() >= 4 and y.nunique() > 1:
        n_splits = min(5, int(groups.nunique()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(x, y, groups=groups))
        y_pred = cross_val_predict(model, x, y, cv=splits, method="predict")
        y_prob_fail = cross_val_predict(model, x, y, cv=splits, method="predict_proba")[:, 1]
        y_eval = y
        eval_method = f"stratified_group_{n_splits}fold_cv"
    else:
        # fallback for smaller datasets.
        if groups is not None and groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(x, y, groups=groups))
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if y.nunique() > 1 else None,
            )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob_fail = model.predict_proba(x_test)[:, 1]
        y_eval = y_test
        eval_method = "holdout_split"
        if y_eval.nunique() < 2:
            warnings.append("Evaluation set had one class only; AUC is not defined.")

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_eval, y_prob_fail)) if len(np.unique(y_eval)) > 1 else None,
        "evaluation_method": eval_method,
        "warnings": warnings,
    }

    # Final model fit on full data for inference after evaluation.
    model.fit(x, y)

    fi = pd.DataFrame({"feature": RF_FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    joblib.dump(model, MODEL_DIR / "rf_model.joblib")
    return TrainResult(metrics=metrics, feature_importance=fi)


def load_rf() -> RandomForestClassifier:
    return joblib.load(MODEL_DIR / "rf_model.joblib")


def classify_risk(pass_prob: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if pass_prob >= t["low"]:
        return "Low"
    if pass_prob >= t["medium"]:
        return "Medium"
    return "High"


def infer(term_df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    model = load_rf()
    x = term_df[RF_FEATURES].copy()
    fail_prob = model.predict_proba(x)[:, 1]
    pass_prob = 1.0 - fail_prob

    work = term_df.copy()
    if "Student Name" not in work.columns:
        work["Student Name"] = work["Student ID"].astype(str)
    if "avg_overall_pct" not in work.columns:
        work["avg_overall_pct"] = work.get("avg_ca_pct", 0.0)

    out = work[
        [
            "Student ID",
            "Student Name",
            "Term",
            "Class",
            "avg_ca_pct",
            "avg_overall_pct",
            "avg_attendance",
            "subject_failure_count",
        ]
    ].copy()
    out["pass_probability"] = pass_prob
    out["predicted_outcome"] = np.where(out["pass_probability"] >= 0.5, "Pass", "Fail")
    out["risk_tier"] = out["pass_probability"].apply(lambda p: classify_risk(float(p), thresholds))
    return out
