"""Feature selection functions using multiple methods.

This module provides robust feature selection by combining:
- Correlation-based filtering
- Lasso regularization
- F-test and mutual information
- Tree-based feature importance
- Ensemble voting across methods

All methods are unified for both binary classification and regression tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..logger import log_info, log_debug, log_warning
from ..config import CONFIG


def select_features_unified(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    task_type: str = 'binary',
    method: str = 'ensemble'
) -> Dict:
    """Select features using unified format across multiple methods.

    This function combines multiple feature selection techniques to ensure
    robust feature selection. The process includes:
    1. Correlation-based filtering (remove highly correlated features)
    2. Multiple scoring methods (Lasso, F-test, Tree-based)
    3. Voting-based ensemble selection

    Args:
        X_train: Training features (DataFrame or ndarray)
        y_train: Training labels (Series or ndarray)
        X_test: Test features (DataFrame or ndarray)
        task_type: Task type ('binary' for classification, 'regression' for regression)
        method: Selection method ('lasso', 'f_test', 'mutual_info', 'ensemble')

    Returns:
        Dictionary containing:
            - X_train_filtered: Selected training features
            - X_test_filtered: Selected test features
            - selected_features: List of selected feature names
            - feature_importance: Feature importance scores
            - n_selected: Number of selected features
    """
    # Extract feature names
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_array = X_train.values
        X_test_array = X_test.values
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_array = X_train
        X_test_array = X_test

    log_info(f"Feature selection started (method: {method})")
    log_info(f"  Initial features: {len(feature_names)}")

    # =========================================
    # Phase 1: Remove highly correlated features
    # =========================================
    log_info(f"  Phase 1: Removing correlated features (threshold: {CONFIG.get('CORRELATION_THRESHOLD', 0.95)})...")

    corr_matrix = pd.DataFrame(X_train_array, columns=feature_names).corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_corr_features = set()
    for column in upper_triangle.columns:
        high_corr_cols = upper_triangle[column][
            upper_triangle[column] > CONFIG.get('CORRELATION_THRESHOLD', 0.95)
        ].index
        high_corr_features.update(high_corr_cols)

    features_after_corr = [f for f in feature_names if f not in high_corr_features]
    log_debug(f"    After correlation removal: {len(features_after_corr)} features (removed: {len(high_corr_features)})")

    # Reconstruct feature indices
    feature_indices = [i for i, f in enumerate(feature_names) if f in features_after_corr]
    X_train_filtered = X_train_array[:, feature_indices]
    X_test_filtered = X_test_array[:, feature_indices]
    feature_names = features_after_corr

    # =========================================
    # Phase 2: Score features using multiple methods
    # =========================================
    log_info("  Phase 2: Scoring features...")

    feature_votes = {f: 0 for f in feature_names}

    # Method 1: Lasso regularization
    try:
        if task_type == 'binary':
            model_lasso = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            model_lasso.fit(X_train_filtered, y_train)
            lasso_coef = np.abs(model_lasso.coef_[0])
        else:
            lasso_model = LassoCV(cv=3, max_iter=10000, random_state=42)
            lasso_model.fit(X_train_filtered, y_train)
            lasso_coef = np.abs(lasso_model.coef_)

        lasso_features = [
            f for f, c in zip(feature_names, lasso_coef)
            if c > np.percentile(lasso_coef, 50)
        ]
        for f in lasso_features:
            feature_votes[f] += 1
        log_debug(f"    Lasso: {len(lasso_features)} features recommended")
    except Exception as e:
        log_debug(f"    Lasso: Skipped ({str(e)[:30]})")

    # Method 2: F-test
    try:
        if task_type == 'binary':
            selector = SelectKBest(
                f_classif,
                k=min(len(feature_names) // 2, CONFIG.get('MAX_FEATURES', 80))
            )
        else:
            selector = SelectKBest(
                f_regression,
                k=min(len(feature_names) // 2, CONFIG.get('MAX_FEATURES', 80))
            )

        selector.fit(X_train_filtered, y_train)
        f_test_features = [f for f, s in zip(feature_names, selector.get_support()) if s]
        for f in f_test_features:
            feature_votes[f] += 1
        log_debug(f"    F-test: {len(f_test_features)} features recommended")
    except Exception as e:
        log_debug(f"    F-test: Skipped ({str(e)[:30]})")

    # Method 3: Tree-based importance
    try:
        tree_model = (
            RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            if task_type == 'binary'
            else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        )
        tree_model.fit(X_train_filtered, y_train)
        tree_features = [
            f for f, imp in zip(feature_names, tree_model.feature_importances_)
            if imp > np.percentile(tree_model.feature_importances_, 50)
        ]
        for f in tree_features:
            feature_votes[f] += 1
        log_debug(f"    Tree: {len(tree_features)} features recommended")
    except Exception as e:
        log_debug(f"    Tree: Skipped ({str(e)[:30]})")

    # =========================================
    # Phase 3: Voting-based integration
    # =========================================
    log_info("  Phase 3: Voting integration...")

    # Select features recommended by at least one method
    selected_features = [f for f, votes in feature_votes.items() if votes >= 1]

    # Apply feature count constraints
    min_features = CONFIG.get('MIN_FEATURES', 20)
    max_features = CONFIG.get('MAX_FEATURES', 80)

    if len(selected_features) < min_features:
        selected_features = list(feature_votes.keys())[:min_features]
    elif len(selected_features) > max_features:
        selected_features = sorted(
            selected_features,
            key=lambda f: feature_votes[f],
            reverse=True
        )[:max_features]

    log_debug(f"    Final selection: {len(selected_features)} features (min: {min_features}, max: {max_features})")

    # =========================================
    # Phase 4: Apply filtering
    # =========================================
    selected_indices = [i for i, f in enumerate(feature_names) if f in selected_features]
    X_train_final = X_train_filtered[:, selected_indices]
    X_test_final = X_test_filtered[:, selected_indices]
    selected_features_final = [feature_names[i] for i in selected_indices]

    # Feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': selected_features_final,
        'votes': [feature_votes.get(f, 0) for f in selected_features_final]
    }).sort_values('votes', ascending=False)

    result = {
        'X_train_filtered': X_train_final,
        'X_test_filtered': X_test_final,
        'selected_features': selected_features_final,
        'feature_importance': feature_importance,
        'n_selected': len(selected_features_final)
    }

    log_info(f"  Feature selection completed: {result['n_selected']} features selected")

    return result


def filter_features_by_pvalue(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    task_type: str = 'binary',
    alpha: float = 0.05
) -> Dict:
    """Filter features by statistical significance (p-value).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        task_type: Task type ('binary' or 'regression')
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary containing filtered features and p-values
    """
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    # Select scoring function
    score_func = f_classif if task_type == 'binary' else f_regression

    # Calculate p-values
    selector = SelectKBest(score_func, k='all')
    selector.fit(X_train, y_train)

    # Get p-values
    pvalues = selector.pvalues_
    feature_names = X_train.columns.tolist()

    # Filter by significance
    significant_features = [
        f for f, p in zip(feature_names, pvalues)
        if p < alpha
    ]

    log_info(f"P-value filtering: {len(significant_features)}/{len(feature_names)} features (alpha={alpha})")

    # Apply filter
    X_train_filtered = X_train[significant_features]
    X_test_filtered = X_test[significant_features]

    return {
        'X_train_filtered': X_train_filtered,
        'X_test_filtered': X_test_filtered,
        'selected_features': significant_features,
        'pvalues': dict(zip(feature_names, pvalues)),
        'n_selected': len(significant_features)
    }
