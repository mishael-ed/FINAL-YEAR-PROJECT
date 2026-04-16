from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Suppress TensorFlow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .config import DEFAULT_THRESHOLDS, MODEL_DIR
from .features import RF_FEATURES, build_rf_xy


@dataclass
class TrainResult:
    metrics: dict
    feature_importance: pd.DataFrame


def create_sequences(term_df: pd.DataFrame, seq_length: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences of student data for LSTM training.
    
    Groups by student, creates sliding windows of consecutive terms.
    Returns X (sequences) and y (target for next term).
    """
    sequences = []
    targets = []
    
    # Sort by student and term
    df = term_df.sort_values(["Student ID", "Term Num"]).copy()
    
    for student_id, group in df.groupby("Student ID"):
        # Skip if not enough terms for sequence
        if len(group) < seq_length + 1:
            continue
        
        group_reset = group.reset_index(drop=True)
        
        # Create sliding windows
        for i in range(len(group_reset) - seq_length):
            # Input: seq_length consecutive terms
            seq = group_reset.iloc[i:i+seq_length][RF_FEATURES].values
            sequences.append(seq)
            
            # Target: next term outcome
            target = group_reset.iloc[i+seq_length]["target_next_term_risk"]
            if pd.notna(target):
                targets.append(int(target))
    
    if not sequences:
        # Fallback for small datasets
        return np.array([]), np.array([])
    
    X = np.array(sequences)  # Shape: (samples, seq_length, features)
    y = np.array(targets)
    return X, y


def train_lstm(term_df: pd.DataFrame) -> dict:
    """Train LSTM model for time-series prediction."""
    if not TENSORFLOW_AVAILABLE:
        return {"error": "TensorFlow not installed. Install: pip install tensorflow"}
    
    train_df = term_df.copy()
    if "target_next_term_risk" in train_df.columns:
        train_df = train_df[train_df["target_next_term_risk"].notna()].copy()
    
    X, y = create_sequences(train_df, seq_length=3)
    
    if len(X) == 0:
        return {"error": "Not enough data for LSTM training", "samples": 0}
    
    # Normalize features
    X_shape = X.shape
    X_flat = X.reshape(-1, X_shape[-1])
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(X_shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_pred_prob)) if len(np.unique(y_test)) > 1 else None,
        "samples": len(X),
        "evaluation_method": "lstm_sequential"
    }
    
    # Save model and scaler
    model.save(MODEL_DIR / "lstm_model.keras")
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
    
    return metrics


def train_xgboost(term_df: pd.DataFrame) -> dict:
    """Train XGBoost model (gradient boosting ensemble)."""
    train_df = term_df.copy()
    if "target_next_term_risk" in train_df.columns:
        train_df = train_df[train_df["target_next_term_risk"].notna()].copy()
    
    x, y = build_rf_xy(train_df)
    groups = train_df["Student ID"] if "Student ID" in train_df.columns else None
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )
    
    warnings: list[str] = []
    
    if groups is not None and groups.nunique() >= 4 and y.nunique() > 1:
        n_splits = min(5, int(groups.nunique()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(x, y, groups=groups))
        y_pred = cross_val_predict(model, x, y, cv=splits, method="predict")
        y_prob_fail = cross_val_predict(model, x, y, cv=splits, method="predict_proba")[:, 1]
        y_eval = y
    else:
        if groups is not None and groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(x, y, groups=groups))
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
            )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob_fail = model.predict_proba(x_test)[:, 1]
        y_eval = y_test
    
    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_eval, y_prob_fail)) if len(np.unique(y_eval)) > 1 else None,
        "evaluation_method": "xgboost_cv"
    }
    
    model.fit(x, y)
    joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")
    
    return metrics


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


def load_lstm() -> tuple:
    """Load LSTM model and scaler."""
    if not TENSORFLOW_AVAILABLE:
        return None, None
    try:
        model = load_model(MODEL_DIR / "lstm_model.keras")
        scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
        return model, scaler
    except:
        return None, None


def load_xgboost() -> GradientBoostingClassifier:
    """Load XGBoost model."""
    try:
        return joblib.load(MODEL_DIR / "xgboost_model.joblib")
    except:
        return None


def classify_risk(pass_prob: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if pass_prob >= t["low"]:
        return "Low"
    if pass_prob >= t["medium"]:
        return "Medium"
    return "High"


def infer(term_df: pd.DataFrame, thresholds: dict | None = None, model_type: str = "ensemble") -> pd.DataFrame:
    """Make predictions using ensemble of all models (RF + XGBoost + LSTM).
    
    Args:
        term_df: Student data with features
        thresholds: Risk tier thresholds
        model_type: Currently always uses ensemble (all available models combined)
    """
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
    
    # Always use ensemble of all available models
    predictions = []
    x = term_df[RF_FEATURES].copy()
    
    # Try Random Forest
    try:
        rf_model = load_rf()
        rf_prob = rf_model.predict_proba(x)[:, 1]  # Probability of FAIL
        rf_pass_prob = 1.0 - rf_prob
        predictions.append(("RF", rf_pass_prob))
    except Exception as e:
        pass
    
    # Try XGBoost
    try:
        xgb_model = load_xgboost()
        if xgb_model is not None:
            xgb_prob = xgb_model.predict_proba(x)[:, 1]
            xgb_pass_prob = 1.0 - xgb_prob
            predictions.append(("XGBoost", xgb_pass_prob))
    except Exception as e:
        pass
    
    # Try LSTM
    try:
        if TENSORFLOW_AVAILABLE:
            lstm_model, scaler = load_lstm()
            if lstm_model is not None and scaler is not None and len(x) > 0:
                x_scaled = scaler.transform(x.values)
                x_lstm = x_scaled.reshape(x_scaled.shape[0], 1, x_scaled.shape[1])
                lstm_prob_fail = lstm_model.predict(x_lstm, verbose=0).flatten()
                lstm_pass_prob = 1.0 - lstm_prob_fail
                predictions.append(("LSTM", lstm_pass_prob))
    except Exception as e:
        pass
    
    # Average predictions from all models
    if len(predictions) > 0:
        model_names = [name for name, _ in predictions]
        prob_arrays = [probs for _, probs in predictions]
        pass_prob = np.mean(prob_arrays, axis=0)
        out["prediction_models"] = f"Ensemble({', '.join(model_names)})"
    else:
        # Fallback: RF only
        rf_model = load_rf()
        rf_prob = rf_model.predict_proba(x)[:, 1]
        pass_prob = 1.0 - rf_prob
        out["prediction_models"] = "RF (fallback)"
    
    out["pass_probability"] = pass_prob
    out["predicted_outcome"] = np.where(out["pass_probability"] >= 0.5, "Pass", "Fail")
    out["risk_tier"] = out["pass_probability"].apply(lambda p: classify_risk(float(p), thresholds))
    
    return out
