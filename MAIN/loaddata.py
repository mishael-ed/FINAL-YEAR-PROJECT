from __future__ import annotations

import argparse
from sap.io import load_table, prepare
from sap.features import clean_data, build_term_features
from sap.model import infer, train_rf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", default="student_data.xlsx")
    parser.add_argument("--predict-input", default="student_data.xlsx")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    train_df = load_table(args.train_input)
    train_prepared = prepare(train_df, mode="train")
    train_term = build_term_features(clean_data(train_prepared))
    train_rf(train_term)

    pred_df = load_table(args.predict_input)
    pred_prepared = prepare(pred_df, mode="predict")
    pred_term = build_term_features(clean_data(pred_prepared))
    preds = infer(pred_term)
    preds.to_csv(args.output, index=False)

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
