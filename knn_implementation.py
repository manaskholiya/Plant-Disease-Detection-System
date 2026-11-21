"""KNN implementation for Lab 08 using sklearn's Breast Cancer dataset.

This file completes the KNN portion only; it does not implement Naive Bayes.
Run with: python knn_implementation.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def main():
    # sklearn target: 0 = malignant, 1 = benign. Keep this mapping explicit
    # so the report and plotted labels have the same meaning.
    cancer = load_breast_cancer(as_frame=True)
    data = cancer.data[["mean radius", "mean texture", "mean perimeter"]].copy()
    data.columns = ["Radius", "Texture", "Perimeter"]
    data["Diagnosis"] = pd.Series(cancer.target, index=data.index).map(
        {0: "Malignant", 1: "Benign"}
    )

    print("First five rows of the Breast Cancer dataset:")
    print(data.head())

    print("\nMissing values before handling:")
    print(data.isna().sum())

    feature_names = ["Radius", "Texture", "Perimeter"]
    X = data[feature_names]
    y = data["Diagnosis"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Radius", y="Perimeter", hue="Diagnosis")
    plt.title("Breast Cancer: Radius vs Perimeter")
    plt.tight_layout()
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Fit imputation and scaling only on training data to avoid test leakage.
    # The sklearn dataset has no missing values, but this handles them if present.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print("\nMissing feature values after handling:")
    print(pd.DataFrame(X_train, columns=feature_names).isna().sum())

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    malignant_probability = model.predict_proba(X_test_scaled)[
        :, list(model.classes_).index("Malignant")
    ]

    labels = ["Malignant", "Benign"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=labels, columns=labels))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted diagnosis")
    plt.ylabel("Actual diagnosis")
    plt.title("KNN Confusion Matrix (k=7)")
    plt.tight_layout()
    plt.show()

    y_test_malignant = (y_test == "Malignant").astype(int)
    auc = roc_auc_score(y_test_malignant, malignant_probability)
    print(f"\nROC-AUC score (Malignant as positive class): {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
