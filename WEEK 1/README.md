# Week 1 – Python for Machine Learning & Data Preprocessing

A modular, commented Python project that explores and preprocesses a real-world
Titanic passenger dataset. It demonstrates data inspection, missing-value
handling, feature selection, categorical encoding, numerical standardization,
and CSV export using pandas, NumPy, and scikit-learn.

## Project structure

```text
week1-ml-preprocessing/
├── preprocessing.py
├── README.md
└── cleaned_dataset.csv       # generated after running the script
```

## Requirements

- Python 3.9+
- pandas
- NumPy
- scikit-learn

Install dependencies:

```bash
python -m pip install pandas numpy scikit-learn
```

## Run

Use the built-in Titanic source (internet required for download):

```bash
python preprocessing.py
```

Or provide your own CSV:

```bash
python preprocessing.py --input data/my_dataset.csv --output outputs/cleaned_dataset.csv
```

For a different dataset, update `TARGET_COLUMN` near the top of the script and
review the feature-selection list in `clean_and_select_features()`. The target
column must exist and should contain the label to predict.

## Preprocessing decisions

| Step | Technique | Reason |
|---|---|---|
| Duplicate handling | Remove exact duplicate rows | Prevent repeated records from distorting analysis |
| Feature selection | Drop `PassengerId`, `Name`, `Ticket`, `Cabin` | Identifier/high-cardinality fields add little value to this baseline; `Cabin` is heavily incomplete |
| Numeric missing values | Median imputation | Robust to skew and outliers |
| Categorical missing values | Most-frequent imputation | Replaces blanks with an observed valid category |
| Nominal categories | One-hot encoding | Does not imply an artificial ranking between categories |
| Numeric features | StandardScaler | Centers numeric features and scales them to unit variance |
| Target | Retain `Survived` unchanged | Keeps the labelled outcome available for supervised learning |

## Output

The script writes `cleaned_dataset.csv` (or the path supplied with `--output`).
It prints the input shape, data types, descriptive statistics, missing-value
counts, duplicate count, selected columns, final shape, remaining missing values,
and a preview of the transformed data.

## Important modelling note

This script fits preprocessing on the full dataset to create the requested
standalone cleaned CSV. For a predictive modelling experiment, split the data
into training and test sets first, then fit imputers, encoders, and scalers on
the training set only (preferably in a scikit-learn `Pipeline`). Apply the fitted
pipeline to the test set. This prevents data leakage and gives a more reliable
evaluation. The exported CSV is a preprocessing artifact, not a model evaluation.

## Reproducibility

The source CSV is read from the public URL in `preprocessing.py` when no local
input path is supplied. Save a local copy of the source data if you need a fully
offline, version-controlled run. Record the dataset source and retrieval date
in your internship submission.
