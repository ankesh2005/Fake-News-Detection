from pathlib import Path

import pandas as pd

try:  # Supports both `python src/preprocessing.py` and package imports.
    from .text_processing import clean_text
except ImportError:
    from text_processing import clean_text


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"


def main() -> None:
    fake_df = pd.read_csv(DATASET_DIR / "Fake.csv")
    true_df = pd.read_csv(DATASET_DIR / "True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)

    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    df = df.dropna(subset=["title", "text"]).copy()
    df["content"] = (df["title"].astype(str) + " " + df["text"].astype(str)).map(clean_text)
    df = df[df["content"].str.strip().ne("")][["content", "label"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = DATASET_DIR / "processed_news.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} cleaned articles to {output_path}")
    print(df["label"].value_counts().sort_index().rename(index={0: "Fake", 1: "Real"}))


if __name__ == "__main__":
    main()
