#!/usr/bin/env python3
"""
train_model_contact.py
----------------------
Trains a baseline binary classifier for contact prediction.

Usage:
    python -m src.models.train_model_contact \
        --input data_proc/training_contact.parquet \
        --output_dir models/contact
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target, perform train/test split.
    """
    # Identify columns to exclude from features
    exclude_cols = ['is_contact', 'game_date', 'game_pk', 'pitcher', 'batter']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df['is_contact']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Training set:   {len(X_train):,} samples")
    print(f"  Test set:       {len(X_test):,} samples")
    print(f"  Positive rate (train): {y_train.mean():.3f}")
    print(f"  Positive rate (test):  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, feature_cols


def build_preprocessor(X: pd.DataFrame):
    """
    Create preprocessing pipeline for numeric and categorical features.
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    # Build transformers
    transformers = []

    if numeric_features:
        # Pipeline for numeric: impute then scale
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_pipeline, numeric_features))

    if categorical_features:
        # Pipeline for categorical: impute then one-hot
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    return preprocessor


def train_baseline(X_train, X_test, y_train, y_test):
    """
    Train baseline logistic regression model.
    """
    print("\n[3/5] Building preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)

    print("\n[4/5] Training logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Fit
    pipeline.fit(X_train, y_train)
    print("  ✓ Training complete")

    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    return pipeline, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def evaluate_model(y_train, y_test, y_pred_train, y_pred_test, y_proba_train, y_proba_test):
    """
    Compute evaluation metrics.
    """
    # Train metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_proba_train)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_pred_train, average='binary', zero_division=0
    )

    # Test metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='binary', zero_division=0
    )

    metrics = {
        'train': {
            'accuracy': float(train_acc),
            'roc_auc': float(train_auc),
            'precision': float(train_prec),
            'recall': float(train_rec),
            'f1_score': float(train_f1)
        },
        'test': {
            'accuracy': float(test_acc),
            'roc_auc': float(test_auc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1_score': float(test_f1)
        }
    }

    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    metrics['test']['confusion_matrix'] = cm_test.tolist()

    return metrics


def print_metrics(metrics):
    """
    Pretty print evaluation metrics.
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for split in ['train', 'test']:
        print(f"\n{split.upper()} SET:")
        m = metrics[split]
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1_score']:.4f}")

        if 'confusion_matrix' in m:
            print(f"\n  Confusion Matrix:")
            cm = np.array(m['confusion_matrix'])
            print(f"    [[TN={cm[0,0]:,}, FP={cm[0,1]:,}],")
            print(f"     [FN={cm[1,0]:,}, TP={cm[1,1]:,}]]")

    print("\n" + "=" * 80)


def main(args):
    print("=" * 80)
    print("Training Contact Prediction Model (Baseline)")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading training data...")
    df = pd.read_parquet(args.input)
    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Prepare data
    print("\n[2/5] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(
        df, test_size=args.test_size, random_state=args.random_state
    )

    # Train model
    pipeline, y_pred_train, y_pred_test, y_proba_train, y_proba_test = train_baseline(
        X_train, X_test, y_train, y_test
    )

    # Evaluate
    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(
        y_train, y_test, y_pred_train, y_pred_test, y_proba_train, y_proba_test
    )

    # Print results
    print_metrics(metrics)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "baseline.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Model saved → {model_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {metrics_path}")

    # Save feature names (for reference)
    feature_info = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols)
    }
    features_path = output_dir / "feature_info.json"
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✅ Feature info saved → {features_path}")

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to training_contact.parquet")
    ap.add_argument("--output_dir", default="models/contact",
                    help="Directory to save model and metrics")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Test set proportion (default: 0.2)")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()
    main(args)
