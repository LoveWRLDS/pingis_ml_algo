"""
train_rf.py — Phase 1: Random Forest classifier for hit detection

Loads data/processed/dataset.csv, trains a Random Forest,
evaluates on a held-out test set, and saves the model.

Usage:
    python skills/pingis-stroke-detection/scripts/train_rf.py

Output:
    data/models/rf_classifier.pkl   — trained model
    data/models/feature_scaler.pkl  — fitted StandardScaler (required at inference)
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATASET = Path("data/processed/dataset.csv")
MODEL_DIR = Path("data/models")

# Labels to train on. Remove "idle" if you don't have idle samples yet.
TARGET_LABELS = ["hit", "swing_miss"]   # idle kräver ~50 samples — lägg till när du har mer data


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}")
        print("Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(DATASET)

    # Filter to known labels only
    df = df[df["label"].isin(TARGET_LABELS)]

    if len(df) < 10:
        print(f"Only {len(df)} rows after filtering. Need more training data.")
        sys.exit(1)

    print(f"\nDataset: {len(df)} rows")
    print(df["label"].value_counts().to_string())

    feature_cols = [c for c in df.columns if c not in ("label", "stroke_type", "player_name", "handedness")]
    X = df[feature_cols].values.astype(float)
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return X, y, feature_cols, le


def train_and_evaluate(X, y, feature_cols, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=min(5, len(y_train) // 5), scoring="f1_macro")
    print(f"\nCross-validation F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Test set evaluation
    y_pred = clf.predict(X_test_s)
    print("\n── Test Set Results ──────────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix (test set)")

    # Feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # top 20 features
    axes[1].barh(
        range(len(indices)),
        importances[indices],
        align="center",
        color="steelblue",
    )
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title("Top 20 Features")

    plt.tight_layout()
    plt.show()

    print("\nTop 10 most important features:")
    for i in indices[:10]:
        print(f"  {feature_cols[i]:<35} {importances[i]:.4f}")

    return clf, scaler


def save_model(clf, scaler, le):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "rf_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "feature_scaler.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    print(f"\n✓ Saved model to {MODEL_DIR}/rf_classifier.pkl")
    print(f"✓ Saved scaler to {MODEL_DIR}/feature_scaler.pkl")
    print(f"✓ Saved label encoder to {MODEL_DIR}/label_encoder.pkl")
    print("\nIMPORTANT: The scaler must be loaded and applied at inference time.")


def main():
    X, y, feature_cols, le = load_dataset()
    clf, scaler = train_and_evaluate(X, y, feature_cols, le)
    save_model(clf, scaler, le)


if __name__ == "__main__":
    main()
