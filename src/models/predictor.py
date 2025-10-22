"""Prediction functions for trained models.

This module provides unified prediction functions for:
- Binary classification (with probability outputs)
- Regression (with rank clipping)
- Batch prediction across multiple test sets
- Confidence-based filtering
- Model ensemble methods

All functions support both single and batch prediction modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional

from ..logger import log_info, log_debug, log_warning
from ..config import CONFIG


def make_predictions(
    X_test: Union[pd.DataFrame, np.ndarray],
    model_result: Dict,
    task_type: str = 'binary',
    ensemble_method: str = 'auto_best'
) -> Dict:
    """Execute predictions on test data.

    Args:
        X_test: Test features (DataFrame or ndarray)
        model_result: Model result dictionary from training (must contain 'model' and 'scaler')
        task_type: Task type ('binary' or 'regression')
        ensemble_method: Ensemble method ('auto_best', 'ensemble', 'manual')

    Returns:
        Dictionary containing:
            - predictions: Predicted values
            - probabilities: Prediction probabilities (binary only)
            - confidence: Confidence scores
            - method_used: Method used for prediction
            - Additional task-specific metrics
    """
    model = model_result['model']
    scaler = model_result['scaler']

    # =========================================
    # 1. Data scaling
    # =========================================
    if scaler is not None:
        X_test_fit = scaler.transform(X_test)
    else:
        X_test_fit = X_test

    # =========================================
    # 2. Execute predictions
    # =========================================
    log_info("Prediction started")
    log_info(f"  Model: {model_result['model_name']}")
    log_info(f"  Samples: {len(X_test_fit)}")

    predictions = model.predict(X_test_fit)

    # =========================================
    # 3. Format outputs
    # =========================================
    if task_type == 'binary':
        # Binary classification
        # Get probability values
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test_fit)[:, 1]
        else:
            probabilities = None

        # Calculate confidence
        if probabilities is not None:
            confidence = np.abs(probabilities - 0.5) * 2  # Normalize to 0-1
        else:
            confidence = np.ones(len(predictions))

        # Filter high-confidence predictions
        high_conf_mask = confidence >= CONFIG['PREDICTION_CONFIDENCE_THRESHOLD']

        result = {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'high_confidence_mask': high_conf_mask,
            'method_used': 'binary_classification',
            'n_high_confidence': high_conf_mask.sum()
        }

        if probabilities is not None:
            log_info(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        log_info(f"  High confidence predictions: {high_conf_mask.sum()}/{len(predictions)}")

    else:
        # Regression (rank learning)
        # Clip predictions to valid rank range
        predictions_clipped = np.clip(predictions, CONFIG['MIN_RANK'], CONFIG['MAX_RANK'])

        # Confidence based on proximity to integers
        fractional_part = np.abs(predictions - np.round(predictions))
        confidence = 1.0 - fractional_part  # Closer to integer = higher confidence

        result = {
            'predictions': predictions_clipped,
            'predictions_raw': predictions,
            'confidence': confidence,
            'method_used': 'regression',
        }

        log_info(f"  Prediction range: [{predictions_clipped.min():.1f}, {predictions_clipped.max():.1f}]")
        log_info(f"  Average predicted rank: {predictions_clipped.mean():.2f}")

    return result


def make_predictions_batch(
    X_test_list: List[Union[pd.DataFrame, np.ndarray]],
    model_result: Dict,
    task_type: str = 'binary'
) -> List[Dict]:
    """Execute batch predictions on multiple test sets.

    Args:
        X_test_list: List of test feature sets
        model_result: Model result dictionary from training
        task_type: Task type ('binary' or 'regression')

    Returns:
        List of prediction result dictionaries
    """
    predictions_list = []

    for i, X_test in enumerate(X_test_list):
        log_debug(f"  [{i+1}/{len(X_test_list)}] Predicting...")
        pred_result = make_predictions(X_test, model_result, task_type)
        predictions_list.append(pred_result)

    log_info(f"  Batch prediction completed ({len(X_test_list)} sets)")

    return predictions_list


def apply_prediction_filter(
    pred_result: Dict,
    task_type: str = 'binary',
    min_confidence: Optional[float] = None
) -> Dict:
    """Filter predictions by confidence threshold.

    Args:
        pred_result: Prediction result dictionary from make_predictions
        task_type: Task type ('binary' or 'regression')
        min_confidence: Minimum confidence threshold (uses CONFIG value if None)

    Returns:
        Dictionary containing filtered predictions and metadata
    """
    if min_confidence is None:
        min_confidence = CONFIG['PREDICTION_CONFIDENCE_THRESHOLD']

    confidence = pred_result['confidence']
    mask = confidence >= min_confidence

    if task_type == 'binary':
        predictions = pred_result['predictions']
        probabilities = pred_result.get('probabilities', None)

        filtered = {
            'predictions': predictions[mask],
            'probabilities': probabilities[mask] if probabilities is not None else None,
            'confidence': confidence[mask],
            'indices': np.where(mask)[0],
            'n_filtered': mask.sum(),
            'filter_ratio': mask.sum() / len(mask)
        }
    else:
        predictions = pred_result['predictions']

        filtered = {
            'predictions': predictions[mask],
            'confidence': confidence[mask],
            'indices': np.where(mask)[0],
            'n_filtered': mask.sum(),
            'filter_ratio': mask.sum() / len(mask)
        }

    return filtered


def ensemble_predictions(
    pred_results_list: List[Dict],
    task_type: str = 'binary',
    weights: Optional[List[float]] = None
) -> Dict:
    """Ensemble predictions from multiple models.

    Args:
        pred_results_list: List of prediction result dictionaries
        task_type: Task type ('binary' or 'regression')
        weights: Model weights (equal weights if None)

    Returns:
        Dictionary containing ensembled predictions
    """
    n_models = len(pred_results_list)

    if weights is None:
        weights = np.ones(n_models) / n_models
    else:
        weights = np.array(weights) / np.sum(weights)

    if task_type == 'binary':
        # Weighted average of probabilities
        proba_ensemble = np.zeros_like(pred_results_list[0]['probabilities'])

        for pred_result, w in zip(pred_results_list, weights):
            if pred_result['probabilities'] is not None:
                proba_ensemble += pred_result['probabilities'] * w

        # Convert to hard labels
        predictions_ensemble = (proba_ensemble >= 0.5).astype(int)

        # Confidence: Agreement ratio across models
        agreement = 0
        for pred_result in pred_results_list:
            agreement += (pred_result['predictions'] == predictions_ensemble).astype(int)
        agreement = agreement / n_models

        result = {
            'predictions': predictions_ensemble,
            'probabilities': proba_ensemble,
            'confidence': agreement,
            'method': 'ensemble_voting',
            'n_models': n_models
        }

    else:
        # Weighted average of predictions
        pred_ensemble = np.zeros_like(pred_results_list[0]['predictions'])

        for pred_result, w in zip(pred_results_list, weights):
            pred_ensemble += pred_result['predictions'] * w

        # Clip to valid range
        pred_ensemble = np.clip(pred_ensemble, CONFIG['MIN_RANK'], CONFIG['MAX_RANK'])

        # Confidence: Inverse of prediction variance (lower variance = higher confidence)
        pred_variance = np.var([pr['predictions'] for pr in pred_results_list], axis=0)
        confidence = 1.0 / (1.0 + pred_variance)

        result = {
            'predictions': pred_ensemble,
            'confidence': confidence,
            'method': 'ensemble_averaging',
            'n_models': n_models,
            'variance': pred_variance
        }

    return result


def predict_with_threshold(
    X_test: Union[pd.DataFrame, np.ndarray],
    model_result: Dict,
    task_type: str = 'binary',
    threshold: float = 0.5,
    return_probabilities: bool = True
) -> Dict:
    """Execute predictions with custom decision threshold.

    Args:
        X_test: Test features
        model_result: Model result dictionary from training
        task_type: Task type ('binary' or 'regression')
        threshold: Decision threshold for binary classification
        return_probabilities: Whether to return probability values

    Returns:
        Dictionary containing predictions with custom threshold
    """
    # Get base predictions
    pred_result = make_predictions(X_test, model_result, task_type)

    if task_type == 'binary' and pred_result['probabilities'] is not None:
        # Apply custom threshold
        predictions_custom = (pred_result['probabilities'] >= threshold).astype(int)

        result = {
            'predictions': predictions_custom,
            'probabilities': pred_result['probabilities'] if return_probabilities else None,
            'confidence': pred_result['confidence'],
            'threshold': threshold,
            'method_used': f'binary_classification_threshold_{threshold}'
        }

        log_info(f"  Custom threshold applied: {threshold}")
        log_info(f"  Positive predictions: {predictions_custom.sum()}/{len(predictions_custom)}")

        return result
    else:
        log_warning(f"  Custom threshold not applicable for task_type={task_type}")
        return pred_result


def evaluate_prediction_quality(
    pred_result: Dict,
    y_true: np.ndarray,
    task_type: str = 'binary'
) -> Dict:
    """Evaluate prediction quality metrics.

    Args:
        pred_result: Prediction result dictionary
        y_true: True labels
        task_type: Task type ('binary' or 'regression')

    Returns:
        Dictionary containing quality metrics
    """
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

    predictions = pred_result['predictions']

    if task_type == 'binary':
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
        }

        if pred_result.get('probabilities') is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, pred_result['probabilities'])
            except:
                metrics['roc_auc'] = np.nan

    else:
        metrics = {
            'mae': mean_absolute_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
        }

    return metrics
