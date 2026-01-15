#!/usr/bin/env python3
"""
train_model_bip.py
------------------
Trains a multi-class classifier for ball-in-play (BIP) outcome prediction.

This model predicts the outcome type (OUT, 1B, 2B, 3B, HR, ROE) for balls in play.
Optionally can also predict spray location (sector_bin, ring_bin, spray_bin).

Usage:
    python -m src.models.train_model_bip \
        --input data_proc/training_bip.parquet \
        --output_dir models/bip \
        --target outcome
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    log_loss
)
import joblib


def prepare_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and target, perform train/test split.
    """
    # Identify columns to exclude from features
    exclude_cols = [
        target_col, 'outcome', 'sector_bin', 'ring_bin', 'spray_bin',
        'game_date', 'game_pk', 'pitcher', 'batter',
        'hc_x', 'hc_y', 'spray_angle_deg', 'spray_distance_ft',
        'events', 'description'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data. Available: {df.columns.tolist()}")

    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Training set:   {len(X_train):,} samples")
    print(f"  Test set:       {len(X_test):,} samples")
    print(f"  Class distribution (train):")
    print(y_train.value_counts(normalize=True).round(3).to_string())

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
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    return preprocessor


def train_model(X_train, X_test, y_train, y_test, model_type: str = 'random_forest'):
    """
    Train multi-class classifier for BIP outcome prediction.

    Args:
        model_type: 'random_forest' or 'logistic'
    """
    print(f"\n[3/5] Building preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)

    print(f"\n[4/5] Training {model_type} model...")

    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
    elif model_type == 'logistic':
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Fit
    pipeline.fit(X_train, y_train)
    print("  ✓ Training complete")

    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_proba_train = pipeline.predict_proba(X_train)
    y_proba_test = pipeline.predict_proba(X_test)

    return pipeline, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def evaluate_model(y_train, y_test, y_pred_train, y_pred_test, y_proba_train, y_proba_test):
    """
    Compute evaluation metrics for multi-class classification.
    """
    # Get class labels
    classes = sorted(list(set(y_train) | set(y_test)))

    # Train metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_proba_train, labels=classes)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_pred_train, average='weighted', zero_division=0
    )

    # Test metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_proba_test, labels=classes)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=0
    )

    # Per-class metrics (test only)
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test, y_pred_test, average=None, zero_division=0, labels=classes
    )

    metrics = {
        'train': {
            'accuracy': float(train_acc),
            'log_loss': float(train_logloss),
            'precision_weighted': float(train_prec),
            'recall_weighted': float(train_rec),
            'f1_weighted': float(train_f1)
        },
        'test': {
            'accuracy': float(test_acc),
            'log_loss': float(test_logloss),
            'precision_weighted': float(test_prec),
            'recall_weighted': float(test_rec),
            'f1_weighted': float(test_f1)
        }
    }

    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test, labels=classes)
    metrics['test']['confusion_matrix'] = cm_test.tolist()
    metrics['test']['class_labels'] = classes

    # Per-class performance
    metrics['test']['per_class'] = {}
    for i, cls in enumerate(classes):
        metrics['test']['per_class'][str(cls)] = {
            'precision': float(per_class_prec[i]),
            'recall': float(per_class_rec[i]),
            'f1_score': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }

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
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  Log Loss:          {m['log_loss']:.4f}")
        print(f"  Precision (wtd):   {m['precision_weighted']:.4f}")
        print(f"  Recall (wtd):      {m['recall_weighted']:.4f}")
        print(f"  F1-Score (wtd):    {m['f1_weighted']:.4f}")

        if 'confusion_matrix' in m:
            print(f"\n  Confusion Matrix:")
            cm = np.array(m['confusion_matrix'])
            classes = m['class_labels']

            # Print header
            header = "       " + "  ".join([f"{c:>5}" for c in classes])
            print(header)
            print("       " + "-" * (len(header) - 7))

            # Print rows
            for i, cls in enumerate(classes):
                row = f"{cls:>5} |" + "  ".join([f"{cm[i,j]:>5}" for j in range(len(classes))])
                print(row)

        if 'per_class' in m:
            print(f"\n  Per-Class Performance:")
            print(f"  {'Class':<6} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for cls, perf in m['per_class'].items():
                print(f"  {cls:<6} {perf['precision']:>10.4f} {perf['recall']:>10.4f} "
                      f"{perf['f1_score']:>10.4f} {perf['support']:>10}")

    print("\n" + "=" * 80)


def main(args):
    print("=" * 80)
    print("Training Ball-in-Play (BIP) Outcome Model")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading training data...")
    df = pd.read_parquet(args.input)
    print(f"  ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Prepare data
    print(f"\n[2/5] Preparing train/test split (target: {args.target})...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(
        df, target_col=args.target, test_size=args.test_size, random_state=args.random_state
    )

    # Train model
    pipeline, y_pred_train, y_pred_test, y_proba_train, y_proba_test = train_model(
        X_train, X_test, y_train, y_test, model_type=args.model_type
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
    model_path = output_dir / f"{args.model_type}_{args.target}.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Model saved → {model_path}")

    # Save metrics
    metrics_path = output_dir / f"metrics_{args.target}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {metrics_path}")

    # Save feature names (for reference)
    feature_info = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'target': args.target,
        'model_type': args.model_type
    }
    features_path = output_dir / f"feature_info_{args.target}.json"
    with open(features_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✅ Feature info saved → {features_path}")

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to training_bip.parquet")
    ap.add_argument("--output_dir", default="models/bip",
                    help="Directory to save model and metrics")
    ap.add_argument("--target", default="outcome",
                    choices=['outcome', 'sector_bin', 'ring_bin', 'spray_bin'],
                    help="Target variable to predict")
    ap.add_argument("--model_type", default="random_forest",
                    choices=['random_forest', 'logistic'],
                    help="Type of classifier to use")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Test set proportion (default: 0.2)")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()
    main(args)
