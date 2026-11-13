"""
Week 1 Internship Task: Python for Machine Learning & Data Preprocessing

Run:
    python preprocessing.py
    python preprocessing.py --input path/to/your_dataset.csv --output cleaned_dataset.csv

If --input is omitted, the script downloads the Titanic training dataset from a
public GitHub mirror. An internet connection is required for this first run.
The output is a model-ready, fully numeric CSV with the target column retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
TARGET_COLUMN = "Survived"


def load_data(input_path: str | None) -> pd.DataFrame:
    """Load a user-provided CSV, or download Titanic when no path is supplied."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV was not found: {path}")
        print(f"\nLoading user dataset: {path.resolve()}")
        return pd.read_csv(path)

    print("\nNo input file supplied. Loading the Titanic training dataset...")
    return pd.read_csv(TITANIC_URL)


def explore_data(df: pd.DataFrame) -> None:
    """Print a compact exploratory data analysis (EDA) report."""
    print("\n" + "=" * 65)
    print("1. INITIAL EXPLORATORY DATA ANALYSIS")
    print("=" * 65)
    print(f"Dataset shape (rows, columns): {df.shape}")
    print("\nFirst five rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nSummary statistics (including categorical columns):")
    print(df.describe(include="all").transpose().to_string())
    print("\nMissing values by column:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].to_string() if missing.any() else "No missing values.")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def clean_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records and columns that are identifiers, free text,
    or excessively incomplete in this Titanic example.

    PassengerId is an identifier; Name and Ticket are high-cardinality fields;
    Cabin has substantial missingness. These are excluded from this baseline.
    The target Survived is deliberately retained.
    """
    print("\n" + "=" * 65)
    print("2. DATA CLEANING & FEATURE SELECTION")
    print("=" * 65)

    cleaned = df.copy()
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    print(f"Duplicate rows removed: {before - len(cleaned)}")

    columns_to_drop = [
        col for col in ["PassengerId", "Name", "Ticket", "Cabin"]
        if col in cleaned.columns
    ]
    cleaned = cleaned.drop(columns=columns_to_drop)
    print(f"Dropped columns: {columns_to_drop or 'None'}")

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' was not found. "
            "For another dataset, update TARGET_COLUMN or adapt the script."
        )

    # Rows without a target cannot be used as labelled examples.
    target_missing = int(cleaned[TARGET_COLUMN].isna().sum())
    if target_missing:
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    print(f"Rows removed because target was missing: {target_missing}")
    print(f"Shape after cleaning/selection: {cleaned.shape}")
    return cleaned


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build separate numeric and categorical preprocessing pipelines."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            # Median is robust to skew and outliers compared with mean imputation.
            ("imputer", SimpleImputer(strategy="median")),
            # StandardScaler centers at 0 and scales to unit standard deviation.
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Most-frequent imputation preserves a valid observed category.
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # One-hot encoding avoids imposing a false order on nominal categories.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    print(f"\nNumeric input features: {numeric_features}")
    print(f"Categorical input features: {categorical_features}")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_and_save(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Impute, encode, scale features and save the transformed dataset."""
    print("\n" + "=" * 65)
    print("3. IMPUTATION, ENCODING & SCALING")
    print("=" * 65)

    y = df[TARGET_COLUMN].reset_index(drop=True)
    X = df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    # OneHotEncoder returns a sparse matrix by default; convert it for CSV output.
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(transformed, columns=feature_names)

    # Keep the target separate from feature scaling/encoding.
    final_df = X_processed.copy()
    final_df[TARGET_COLUMN] = y.to_numpy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output, index=False)

    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Missing values remaining: {int(final_df.isna().sum().sum())}")
    print("\nFinal data types:")
    print(final_df.dtypes.to_string())
    print("\nPreview of processed data:")
    print(final_df.head().to_string(index=False))
    print(f"\nSaved processed dataset to: {output.resolve()}")
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and preprocess a tabular CSV for machine learning."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input CSV. If omitted, the Titanic dataset is downloaded.",
    )
    parser.add_argument(
        "--output",
        default="cleaned_dataset.csv",
        help="Path for the processed CSV (default: cleaned_dataset.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        raw_df = load_data(args.input)
        if raw_df.empty:
            raise ValueError("The input dataset is empty.")
        explore_data(raw_df)
        cleaned_df = clean_and_select_features(raw_df)
        preprocess_and_save(cleaned_df, args.output)
        print("\nPREPROCESSING COMPLETED SUCCESSFULLY.")
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
