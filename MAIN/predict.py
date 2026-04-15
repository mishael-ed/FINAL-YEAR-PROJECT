from __future__ import annotations

import argparse
from sap.io import load_table, prepare
from sap.features import clean_data, build_term_features
from sap.model import infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    df = load_table(args.input)
    prepared = prepare(df, mode="predict")
    clean = clean_data(prepared)
    term_df = build_term_features(clean)
    preds = infer(term_df)
    preds.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(preds.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
