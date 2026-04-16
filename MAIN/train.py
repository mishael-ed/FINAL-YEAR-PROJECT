from __future__ import annotations

import argparse
from sap.io import load_table, prepare
from sap.features import clean_data, build_term_features
from sap.model import train_rf, train_lstm, train_xgboost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = load_table(args.input)
    prepared = prepare(df, mode="train")
    clean = clean_data(prepared)
    term_df = build_term_features(clean)
    
    print(f"Training ensemble models on {len(term_df)} student-term records\n")
    
    # Train Random Forest
    print("=" * 60)
    print("1. RANDOM FOREST MODEL")
    print("=" * 60)
    try:
        rf_result = train_rf(term_df)
        print("Random Forest - Training Complete")
        for k, v in rf_result.metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error training RF: {e}")
    print()
    
    # Train XGBoost
    print("=" * 60)
    print("2. XGBOOST MODEL (Gradient Boosting)")
    print("=" * 60)
    try:
        xgb_result = train_xgboost(term_df)
        print("XGBoost - Training Complete")
        for k, v in xgb_result.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error training XGBoost: {e}")
    print()
    
    # Train LSTM
    print("=" * 60)
    print("3. LSTM MODEL (Deep Learning)")
    print("=" * 60)
    try:
        lstm_result = train_lstm(term_df)
        print("LSTM - Training Complete")
        for k, v in lstm_result.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error training LSTM: {e}")
    print()
    
    print("=" * 60)
    print("All models trained and saved successfully!")
    print("Predictions will use ensemble of all 3 models")
    print("=" * 60)


if __name__ == "__main__":
    main()
