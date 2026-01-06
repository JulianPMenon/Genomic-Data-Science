"""
Metrics calculation utilities
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob=None, average='binary'):
    """
    Calculate various classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_prob (array-like, optional): Predicted probabilities for ROC-AUC
        average (str): Averaging strategy for multi-class ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {}
    
    # Convert to numpy if torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            if average == 'binary':
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average=average, multi_class='ovr')
        except ValueError:
            # Handle case where only one class is present
            metrics['roc_auc'] = None
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics, logger=None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics (dict): Metrics dictionary from calculate_metrics
        logger (logging.Logger, optional): Logger to use for output
    """
    print_fn = logger.info if logger else print
    
    print_fn("=" * 50)
    print_fn("Metrics:")
    print_fn(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print_fn(f"  Precision: {metrics['precision']:.4f}")
    print_fn(f"  Recall:    {metrics['recall']:.4f}")
    print_fn(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print_fn(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print_fn("\nConfusion Matrix:")
        print_fn(metrics['confusion_matrix'])
    
    print_fn("=" * 50)
