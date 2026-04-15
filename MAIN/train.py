from __future__ import annotations

import argparse
from sap.io import load_table, prepare
from sap.features import clean_data, build_term_features
from sap.model import train_rf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = load_table(args.input)
    prepared = prepare(df, mode="train")
    clean = clean_data(prepared)
    term_df = build_term_features(clean)
    res = train_rf(term_df)

    print("Training complete")
    for k, v in res.metrics.items():
        if isinstance(v, (int, float)) and v is not None:
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
