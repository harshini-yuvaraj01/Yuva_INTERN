
"""
WEEK 2: SUPERVISED MACHINE LEARNING
Algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)

Dataset: Breast Cancer Wisconsin Dataset
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# Create output folder
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

print("\nLOADING DATASET...")

dataset = load_breast_cancer(as_frame=True)

X = dataset.data

# Original target: 0 = malignant, 1 = benign
# Convert to: 1 = malignant, 0 = benign
y = (dataset.target == 0).astype(int)

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

print("\nFirst five rows:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 2. SPLIT DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            p=2
        ))
    ])
}


# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
confusion_matrices = {}

for name, model in models.items():

    print("\n" + "=" * 50)
    print("MODEL:", name)
    print("=" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

    # Display results
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# --------------------------------------------------
# 5. SAVE METRICS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/classification_metrics.csv",
    index=False
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 6. IDENTIFY BEST MODEL USING F1 SCORE
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBEST MODEL:", best_model["Model"])
print("Best F1 Score:", round(best_model["F1 Score"], 4))


# Save metrics as JSON
with open("outputs/model_results.json", "w") as file:
    json.dump(results, file, indent=4)


# --------------------------------------------------
# 7. MODEL COMPARISON GRAPH
# --------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title("Comparison of Supervised Learning Models")
plt.ylabel("Score")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("outputs/model_comparison.png")
plt.show()


# --------------------------------------------------
# 8. CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, cm) in zip(axes.flatten(), confusion_matrices.items()):

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)

plt.suptitle("Confusion Matrices of Classification Models")
plt.tight_layout()

plt.savefig("outputs/confusion_matrices.png")
plt.show()


print("\nALL MODELS TRAINED AND EVALUATED SUCCESSFULLY!")
print("Check the outputs folder for generated results.")