#!/usr/bin/env python3
"""
evaluate.py
-----------
Comprehensive model evaluation script for contact and BIP models.

Generates evaluation reports including:
- Performance metrics (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves (for binary classification)
- Feature importance (for tree-based models)
- Calibration plots
- Spray charts (for BIP models)

Usage:
    # Evaluate contact model
    python -m src.models.evaluate \
        --model models/contact/baseline.pkl \
        --data data_proc/training_contact.parquet \
        --target is_contact \
        --output_dir reports/contact \
        --task binary

    # Evaluate BIP outcome model
    python -m src.models.evaluate \
        --model models/bip/random_forest_outcome.pkl \
        --data data_proc/training_bip.parquet \
        --target outcome \
        --output_dir reports/bip \
        --task multiclass
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import joblib


def load_model_and_data(model_path: str, data_path: str, target_col: str, test_size: float = 0.2):
    """
    Load trained model and prepare test data.
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Identify columns to exclude from features
    exclude_cols = [
        target_col, 'outcome', 'sector_bin', 'ring_bin', 'spray_bin',
        'game_date', 'game_pk', 'pitcher', 'batter', 'is_contact',
        'hc_x', 'hc_y', 'spray_angle_deg', 'spray_distance_ft',
        'events', 'description'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    # Train/test split (use same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")

    return model, X_test, y_test


def evaluate_binary(model, X_test, y_test, output_dir: Path):
    """
    Evaluate binary classification model (e.g., contact prediction).
    """
    print("\n" + "=" * 80)
    print("BINARY CLASSIFICATION EVALUATION")
    print("=" * 80)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    ap = average_precision_score(y_test, y_proba)

    metrics = {
        'accuracy': float(acc),
        'roc_auc': float(auc),
        'average_precision': float(ap),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Print metrics
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  Average Precision:  {ap:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1-Score:           {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:,}, FP={cm[0,1]:,}],")
    print(f"   [FN={cm[1,0]:,}, TP={cm[1,1]:,}]]")

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved → {output_dir / 'eval_metrics.json'}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Contact', 'Contact']).plot(ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].grid(False)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 0].plot(recall, precision, linewidth=2, label=f'PR (AP = {ap:.3f})')
    axes[1, 0].set_xlabel('Recall', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
    axes[1, 1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    axes[1, 1].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[1, 1].set_ylabel('Fraction of Positives', fontsize=12)
    axes[1, 1].set_title('Calibration Curve', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    print(f"✅ Evaluation plots saved → {output_dir / 'evaluation_plots.png'}")
    plt.close()

    return metrics


def evaluate_multiclass(model, X_test, y_test, output_dir: Path):
    """
    Evaluate multi-class classification model (e.g., BIP outcome prediction).
    """
    print("\n" + "=" * 80)
    print("MULTI-CLASS CLASSIFICATION EVALUATION")
    print("=" * 80)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Get class labels
    classes = sorted(list(set(y_test)))

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba, labels=classes)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=classes
    )

    metrics = {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'class_labels': classes,
        'per_class': {}
    }

    for i, cls in enumerate(classes):
        metrics['per_class'][str(cls)] = {
            'precision': float(per_class_prec[i]),
            'recall': float(per_class_rec[i]),
            'f1_score': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    metrics['confusion_matrix'] = cm.tolist()

    # Print metrics
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Log Loss:          {logloss:.4f}")
    print(f"  Precision (wtd):   {prec:.4f}")
    print(f"  Recall (wtd):      {rec:.4f}")
    print(f"  F1-Score (wtd):    {f1:.4f}")

    print(f"\nPer-Class Performance:")
    print(f"  {'Class':<6} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cls, perf in metrics['per_class'].items():
        print(f"  {cls:<6} {perf['precision']:>10.4f} {perf['recall']:>10.4f} "
              f"{perf['f1_score']:>10.4f} {perf['support']:>10}")

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved → {output_dir / 'eval_metrics.json'}")

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].grid(False)

    # 2. Per-Class Performance
    class_names = [str(c) for c in classes]
    x_pos = np.arange(len(class_names))

    width = 0.25
    axes[1].bar(x_pos - width, per_class_prec, width, label='Precision', alpha=0.8)
    axes[1].bar(x_pos, per_class_rec, width, label='Recall', alpha=0.8)
    axes[1].bar(x_pos + width, per_class_f1, width, label='F1-Score', alpha=0.8)

    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Per-Class Performance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=150, bbox_inches='tight')
    print(f"✅ Evaluation plots saved → {output_dir / 'evaluation_plots.png'}")
    plt.close()

    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Normalized Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix saved → {output_dir / 'confusion_matrix_normalized.png'}")
    plt.close()

    return metrics


def extract_feature_importance(model, output_dir: Path, top_n: int = 20):
    """
    Extract and visualize feature importance for tree-based models.
    """
    try:
        # Try to get feature importances from the classifier
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_

            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()

            # Create dataframe and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Save to CSV
            importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
            print(f"✅ Feature importance saved → {output_dir / 'feature_importance.csv'}")

            # Plot top N features
            top_features = importance_df.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_features)), top_features['importance'].values, alpha=0.8)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'].values)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            print(f"✅ Feature importance plot saved → {output_dir / 'feature_importance.png'}")
            plt.close()

        else:
            print("  ⚠ Model does not have feature_importances_ attribute (not a tree-based model)")

    except Exception as e:
        print(f"  ⚠ Could not extract feature importance: {e}")


def main(args):
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Load model and data
    model, X_test, y_test = load_model_and_data(
        args.model, args.data, args.target, test_size=args.test_size
    )

    output_dir = Path(args.output_dir)

    # Evaluate based on task type
    if args.task == 'binary':
        metrics = evaluate_binary(model, X_test, y_test, output_dir)
    elif args.task == 'multiclass':
        metrics = evaluate_multiclass(model, X_test, y_test, output_dir)
    else:
        raise ValueError(f"Unknown task type: {args.task}")

    # Extract feature importance (if available)
    if args.feature_importance:
        print("\nExtracting feature importance...")
        extract_feature_importance(model, output_dir, top_n=args.top_n)

    print("\n" + "=" * 80)
    print("✅ Evaluation complete!")
    print(f"   Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to trained model (.pkl file)")
    ap.add_argument("--data", required=True,
                    help="Path to training data (.parquet file)")
    ap.add_argument("--target", required=True,
                    help="Target column name")
    ap.add_argument("--task", required=True,
                    choices=['binary', 'multiclass'],
                    help="Classification task type")
    ap.add_argument("--output_dir", default="reports/evaluation",
                    help="Directory to save evaluation results")
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Test set proportion (default: 0.2)")
    ap.add_argument("--feature_importance", action='store_true',
                    help="Extract and visualize feature importance (tree-based models only)")
    ap.add_argument("--top_n", type=int, default=20,
                    help="Number of top features to display (default: 20)")
    args = ap.parse_args()
    main(args)
