
"""
WEEK 3: UNSUPERVISED LEARNING & MODEL EVALUATION

Part 1:
- Wine dataset
- StandardScaler
- K-Means clustering
- Elbow method
- Silhouette score
- Agglomerative clustering
- Dendrogram
- PCA visualization

Part 2:
- Breast Cancer Wisconsin dataset
- Random Forest classifier
- 5-Fold Cross-Validation
- GridSearchCV hyperparameter tuning
- Confusion Matrix
- Precision, Recall, F1-score
- ROC Curve and ROC-AUC

All results are saved in the outputs/ folder.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def save_plot(filename):
    """Save the current Matplotlib figure in the outputs folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.show()
    plt.close()


# ==================================================
# PART 1: UNSUPERVISED LEARNING
# ==================================================

def run_clustering():
    print("\n" + "=" * 60)
    print("PART 1: UNSUPERVISED LEARNING & DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # 1. Load Wine dataset
    wine = load_wine(as_frame=True)
    X = wine.data

    print("\nWine dataset loaded successfully.")
    print("Dataset shape:", X.shape)
    print("\nFirst five rows:")
    print(X.head())

    # 2. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFeature scaling completed using StandardScaler.")

    # --------------------------------------------------
    # 3. K-MEANS: ELBOW METHOD AND SILHOUETTE SCORE
    # --------------------------------------------------

    print("\n--- K-MEANS CLUSTERING ---")

    k_values = range(2, 9)
    inertias = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

        print(
            f"K={k} | Inertia={kmeans.inertia_:.3f} "
            f"| Silhouette Score={score:.4f}"
        )

    # Select K with the highest silhouette score
    best_k = list(k_values)[int(np.argmax(silhouette_scores))]

    print("\nSelected optimal K:", best_k)

    # Save K evaluation values
    k_results = pd.DataFrame({
        "K": list(k_values),
        "Inertia": inertias,
        "Silhouette Score": silhouette_scores
    })

    k_results.to_csv(
        os.path.join(OUTPUT_DIR, "kmeans_evaluation.csv"),
        index=False
    )

    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True)
    save_plot("elbow_method.png")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o"
    )
    plt.title("Silhouette Score for Different K Values")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    save_plot("silhouette_scores.png")

    # Train final K-Means model
    final_kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    kmeans_labels = final_kmeans.fit_predict(X_scaled)

    print("Final K-Means clustering completed.")

    # --------------------------------------------------
    # 4. AGGLOMERATIVE HIERARCHICAL CLUSTERING
    # --------------------------------------------------

    print("\n--- AGGLOMERATIVE HIERARCHICAL CLUSTERING ---")

    # Ward linkage minimizes within-cluster variance.
    # Linkage matrix is used to construct the dendrogram.
    linkage_matrix = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=9
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points / Merged Clusters")
    plt.ylabel("Ward Distance")
    save_plot("dendrogram.png")

    hierarchical = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )

    hierarchical_labels = hierarchical.fit_predict(X_scaled)

    print("Agglomerative clustering completed.")
    print(
        "Hierarchical Silhouette Score:",
        round(silhouette_score(X_scaled, hierarchical_labels), 4)
    )

    # --------------------------------------------------
    # 5. PRINCIPAL COMPONENT ANALYSIS
    # --------------------------------------------------

    print("\n--- PRINCIPAL COMPONENT ANALYSIS (PCA) ---")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    print("Explained variance ratio:")
    print(explained_variance)

    print(
        "Total variance explained by two components:",
        round(sum(explained_variance) * 100, 2),
        "%"
    )

    # Save PCA data and cluster labels
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "KMeans_Cluster": kmeans_labels,
        "Hierarchical_Cluster": hierarchical_labels
    })

    pca_df.to_csv(
        os.path.join(OUTPUT_DIR, "pca_cluster_results.csv"),
        index=False
    )

    # Plot K-Means clusters in PCA space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        s=50
    )

    plt.title("K-Means Clusters Visualized Using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    save_plot("pca_kmeans_clusters.png")

    # Plot hierarchical clusters in PCA space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=hierarchical_labels,
        cmap="plasma",
        s=50
    )

    plt.title("Hierarchical Clusters Visualized Using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    save_plot("pca_hierarchical_clusters.png")

    return {
        "selected_k": int(best_k),
        "best_silhouette": float(max(silhouette_scores)),
        "hierarchical_silhouette": float(
            silhouette_score(X_scaled, hierarchical_labels)
        ),
        "pca_variance": explained_variance.tolist(),
        "pca_total_variance": float(sum(explained_variance))
    }


# ==================================================
# PART 2: MODEL EVALUATION & HYPERPARAMETER TUNING
# ==================================================

def run_model_evaluation():
    print("\n" + "=" * 60)
    print("PART 2: MODEL EVALUATION & HYPERPARAMETER TUNING")
    print("=" * 60)

    # 1. Load Breast Cancer dataset
    cancer = load_breast_cancer(as_frame=True)

    X = cancer.data

    # Original target: 0 = malignant, 1 = benign
    # Convert to: 1 = malignant, 0 = benign
    y = (cancer.target == 0).astype(int)

    print("\nBreast Cancer dataset loaded successfully.")
    print("Dataset shape:", X.shape)
    print("Target: 1 = Malignant, 0 = Benign")

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Create a pipeline
    # Imputation and model fitting are kept together.
    # Random Forest does not require feature scaling.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # --------------------------------------------------
    # 4. BASELINE MODEL AND 5-FOLD CROSS-VALIDATION
    # --------------------------------------------------

    print("\n--- BASELINE RANDOM FOREST ---")

    baseline_model = pipeline.set_params(
        classifier__n_estimators=100,
        classifier__max_depth=None,
        classifier__min_samples_split=2,
        classifier__min_samples_leaf=1
    )

    cv_scores = cross_val_score(
        baseline_model,
        X_train,
        y_train,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    print("5-Fold F1 Scores:", np.round(cv_scores, 4))
    print("Mean CV F1 Score:", round(cv_scores.mean(), 4))
    print("Standard Deviation:", round(cv_scores.std(), 4))

    # Fit baseline on full training set
    baseline_model.fit(X_train, y_train)

    baseline_predictions = baseline_model.predict(X_test)

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
        zero_division=0
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    # --------------------------------------------------
    # 5. HYPERPARAMETER TUNING USING GRIDSEARCHCV
    # --------------------------------------------------

    print("\n--- GRID SEARCH HYPERPARAMETER TUNING ---")

    parameter_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(round(grid_search.best_score_, 4))

    # --------------------------------------------------
    # 6. BASELINE VS TUNED MODEL COMPARISON
    # --------------------------------------------------

    tuned_predictions = best_model.predict(X_test)

    tuned_accuracy = accuracy_score(
        y_test,
        tuned_predictions
    )

    tuned_precision = precision_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    tuned_recall = recall_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    tuned_f1 = f1_score(
        y_test,
        tuned_predictions,
        zero_division=0
    )

    comparison = pd.DataFrame({
        "Model": ["Baseline Random Forest", "Tuned Random Forest"],
        "Accuracy": [
            baseline_accuracy,
            tuned_accuracy
        ],
        "F1 Score": [
            baseline_f1,
            tuned_f1
        ],
        "Cross Validation F1": [
            cv_scores.mean(),
            grid_search.best_score_
        ]
    })

    print("\nBASELINE VS TUNED MODEL:")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        index=False
    )

    # --------------------------------------------------
    # 7. FINAL MODEL EVALUATION
    # --------------------------------------------------

    print("\n--- TUNED MODEL EVALUATION ---")

    print("Accuracy :", round(tuned_accuracy, 4))
    print("Precision:", round(tuned_precision, 4))
    print("Recall   :", round(tuned_recall, 4))
    print("F1 Score :", round(tuned_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            tuned_predictions,
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, tuned_predictions)

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Tuned Random Forest Confusion Matrix")
    save_plot("confusion_matrix.png")

    # --------------------------------------------------
    # 8. ROC CURVE AND ROC-AUC
    # --------------------------------------------------

    print("\n--- ROC CURVE AND ROC-AUC ---")

    # Probability of positive class (malignant = 1)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("ROC-AUC Score:", round(roc_auc, 4))

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title("ROC Curve - Tuned Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_plot("roc_curve.png")

    # --------------------------------------------------
    # 9. SAVE FINAL RESULTS
    # --------------------------------------------------

    final_results = {
        "baseline_accuracy": float(baseline_accuracy),
        "baseline_test_f1": float(baseline_f1),
        "baseline_cv_mean_f1": float(cv_scores.mean()),
        "baseline_cv_std_f1": float(cv_scores.std()),
        "tuned_accuracy": float(tuned_accuracy),
        "tuned_precision": float(tuned_precision),
        "tuned_recall": float(tuned_recall),
        "tuned_f1": float(tuned_f1),
        "tuned_cv_f1": float(grid_search.best_score_),
        "roc_auc": float(roc_auc),
        "best_parameters": grid_search.best_params_,
        "confusion_matrix": cm.tolist()
    }

    with open(
        os.path.join(OUTPUT_DIR, "evaluation_results.json"),
        "w"
    ) as file:
        json.dump(final_results, file, indent=4)

    print("\nAll evaluation results saved successfully.")

    return final_results


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    print("\nWEEK 3 MACHINE LEARNING PROJECT")
    print("Unsupervised Learning & Model Evaluation")

    clustering_results = run_clustering()
    evaluation_results = run_model_evaluation()

    # Save combined summary
    summary = {
        "clustering": clustering_results,
        "model_evaluation": evaluation_results
    }

    with open(
        os.path.join(OUTPUT_DIR, "week3_summary.json"),
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("WEEK 3 PROJECT COMPLETED SUCCESSFULLY!")
    print("Check the outputs folder for all results and plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()