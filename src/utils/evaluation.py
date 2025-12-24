"""
Model evaluation utilities for insurance claims classification.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    fbeta_score
)
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Fitted sklearn model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model (for display)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        dict with metrics: roc_auc, pr_auc, accuracy, precision, recall, f1, f2
    
    Example:
        >>> results = evaluate_model(model, X_test, y_test, 'LogReg')
        >>> print(f"ROC-AUC: {results['roc_auc']:.3f}")
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'model': model_name,
        'accuracy': (y_pred == y_test).mean()
    }
    
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
        results['pr_auc'] = average_precision_score(y_test, y_proba)
        
        # Apply threshold for binary classification
        y_pred_thresh = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        
        tn, fp, fn, tp = cm.ravel()
        results['threshold'] = threshold
        results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['f1'] = (2 * results['precision'] * results['recall'] / 
                        (results['precision'] + results['recall']) 
                        if (results['precision'] + results['recall']) > 0 else 0)
        results['f2'] = fbeta_score(y_test, y_pred_thresh, beta=2)
        results['tp'] = int(tp)
        results['fp'] = int(fp)
        results['fn'] = int(fn)
        results['tn'] = int(tn)
        results['support_1'] = int((y_test == 1).sum())
        results['support_0'] = int((y_test == 0).sum())
    
    return results


def compare_models(results_list, sort_by='roc_auc'):
    """
    Create comparison DataFrame from list of result dicts.
    
    Args:
        results_list: List of dicts from evaluate_model()
        sort_by: Column to sort by (default: 'roc_auc')
    
    Returns:
        DataFrame sorted by specified metric
    
    Example:
        >>> results = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]
        >>> df = compare_models(results)
    """
    df = pd.DataFrame(results_list)
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    return df


def plot_roc_pr_curves(models_dict, X_test, y_test, figsize=(14, 5)):
    """
    Plot ROC and PR curves for multiple models.
    
    Args:
        models_dict: {model_name: fitted_model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size (default: (14, 5))
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> models = {'LogReg': model1, 'RF': model2}
        >>> fig = plot_roc_pr_curves(models, X_test, y_test)
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for name, model in models_dict.items():
        if not hasattr(model, 'predict_proba'):
            print(f"Skipping {name}: no predict_proba method")
            continue
            
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax2.plot(recall, precision, label=f'{name} (AP={pr_auc:.3f})', linewidth=2)
    
    # ROC Curve formatting
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Precision-Recall Curve formatting
    baseline = y_test.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', 
                label=f'Baseline (y={baseline:.2%})', linewidth=1.5)
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    return fig


def best_f1_threshold(y_true, y_proba):
    """
    Find threshold that maximizes F1 score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
    
    Returns:
        float: Optimal threshold
    
    Example:
        >>> threshold = best_f1_threshold(y_test, y_proba)
        >>> print(f"Best F1 threshold: {threshold:.3f}")
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / np.clip(precision + recall, 1e-9, None)
    
    idx = np.nanargmax(f1_scores)
    best_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    
    return best_threshold
