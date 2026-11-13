
import json
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 1. Load Dataset
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("Dataset Loaded Successfully!")
print("Dataset Shape:", X.shape)


# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Create Machine Learning Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])


# 4. Train Model
print("\nTraining Model...")

pipeline.fit(X_train, y_train)


# 5. Make Predictions
y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]


# 6. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))


# 7. Save Model and Metadata
artifact = {
    "pipeline": pipeline,
    "feature_names": list(X.columns),
    "target_names": list(data.target_names),
    "defaults": {
        col: float(X_train[col].median())
        for col in X.columns
    }
}

joblib.dump(artifact, "model.pkl")

print("\nModel saved successfully as model.pkl")


# 8. Save Evaluation Metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": confusion_matrix(
        y_test, y_pred
    ).tolist()
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully!")

print("\n========== TRAINING COMPLETED ==========")
