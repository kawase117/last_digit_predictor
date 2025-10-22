"""Label creation functions for binary classification and regression tasks.

This module provides functions to create labels for:
- Binary classification tasks (TOP1/TOP2)
- Regression tasks (rank learning)
- Unified data preparation with train/test splits
- Data validation

All functions support time-series cross-validation and prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from ..logger import log_info, log_debug, log_warning
from ..config import CONFIG


def create_top_labels(df: pd.DataFrame, event: str, rank: int = 1) -> pd.Series:
    """Create binary labels for TOP1/TOP2 classification.

    Args:
        df: Merged dataframe containing last_digit_rank_diff column
        event: Event name (e.g., '1day', '2day')
        rank: Rank threshold (1 for TOP1, 2 for TOP2)

    Returns:
        Binary labels (1 if rank <= threshold, 0 otherwise)

    Raises:
        ValueError: If last_digit_rank_diff column is not found
    """
    label_col = 'last_digit_rank_diff'

    if label_col not in df.columns:
        raise ValueError(f"{label_col} column not found in dataframe")

    # Rank <= threshold means high performance (label = 1)
    # Rank > threshold means low performance (label = 0)
    labels = (df[label_col] <= rank).astype(int)

    return labels


def create_rank_labels(df: pd.DataFrame) -> pd.Series:
    """Create rank labels for regression tasks.

    Args:
        df: Merged dataframe containing last_digit_rank_diff column

    Returns:
        Rank labels (1-11 range)
    """
    labels = df['last_digit_rank_diff'].copy()

    # Fill NaN with median value
    labels = labels.fillna(6.0)

    # Clip to valid range (1-11)
    labels = np.clip(labels, CONFIG['MIN_RANK'], CONFIG['MAX_RANK'])

    return labels


def prepare_unified_data(
    df: pd.DataFrame,
    event: str,
    task_type: str = 'binary',
    rank: int = 1
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, list]:
    """Prepare data in unified format for training and testing.

    Args:
        df: Merged dataframe with all features
        event: Event name (e.g., '1day', '2day')
        task_type: Task type ('binary' for TOP1/TOP2, 'regression' for rank learning)
        rank: Rank threshold for binary classification (only used when task_type='binary')

    Returns:
        Tuple containing:
            - X_train: Training features
            - y_train: Training labels
            - X_test: Test features
            - y_test: Test labels
            - test_data: Full test dataframe (for profit calculation)
            - feature_cols: List of feature column names

    Raises:
        ValueError: If event column not found or event data is empty
    """
    # Extract event-specific data
    flag_col = f'is_{event}'

    if flag_col not in df.columns:
        raise ValueError(f"Event column '{flag_col}' not found in dataframe")

    event_data = df[df[flag_col] == 1].copy().reset_index(drop=True)

    if len(event_data) == 0:
        raise ValueError(f"No data found for event '{event}'")

    # Generate labels
    if task_type == 'binary':
        labels = create_top_labels(event_data, event, rank)
        task_name = f'TOP{rank}'
    elif task_type == 'regression':
        labels = create_rank_labels(event_data)
        task_name = 'Rank_Learning'
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Auto-detect feature columns
    exclude_patterns = [
        'date', 'event', 'target', 'label', 'current_diff',
        'last_digit_rank', 'digit_num', 'last_digit', 'is_'
    ]

    feature_cols = []
    for col in event_data.columns:
        # Skip columns matching exclude patterns
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue

        # Only numeric columns
        if pd.api.types.is_numeric_dtype(event_data[col]):
            feature_cols.append(col)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")

    log_info(f"Task: {task_name}")
    log_info(f"  Features: {len(feature_cols)} columns")
    log_info(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # Time-series split
    if 'date' in event_data.columns:
        event_data = event_data.sort_values('date').reset_index(drop=True)
        unique_dates = event_data['date'].unique()
    else:
        unique_dates = np.arange(len(event_data))

    n_dates = len(unique_dates)
    n_test = max(1, int(n_dates * CONFIG['TEST_SIZE']))
    n_train = n_dates - n_test

    # Split by time (future data goes to test set)
    if 'date' in event_data.columns:
        train_dates = set(unique_dates[:n_train])
        train_mask = event_data['date'].isin(train_dates)
    else:
        train_mask = np.arange(len(event_data)) < int(n_train * len(event_data))

    test_mask = ~train_mask

    # Extract features and labels
    X = event_data[feature_cols].copy()
    y = labels.reset_index(drop=True)

    # Handle NaN values
    X = X.fillna(X.mean())

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    # Split into train and test
    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    # Keep detailed test data for profit calculation
    test_data = event_data[test_mask].reset_index(drop=True)

    # Print statistics
    log_info(f"  Training samples: {len(X_train)}")
    log_info(f"  Test samples: {len(X_test)}")

    if task_type == 'binary':
        log_info(f"  Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")
        log_info(f"  Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")

        # Check for class imbalance
        label_counts = pd.Series(y_train).value_counts()
        if len(label_counts) == 2:
            ratio = label_counts.iloc[1] / label_counts.iloc[0]
            if ratio < 0.2 or ratio > 5:
                log_warning(f"  Class imbalance: {ratio:.2f}x (potential data insufficiency)")
    else:
        log_info(f"  Train label stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
        log_info(f"  Test label stats: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

    return X_train, y_train, X_test, y_test, test_data, feature_cols


def validate_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = 'binary'
) -> bool:
    """Validate data quality and consistency.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        task_type: Task type ('binary' or 'regression')

    Returns:
        True if data is valid, False otherwise
    """
    errors = []

    # Check 1: Sample size
    if len(X_train) < 10:
        errors.append(f"Training data too small: {len(X_train)} < 10")

    if len(X_test) < 5:
        errors.append(f"Test data too small: {len(X_test)} < 5")

    # Check 2: Features
    if X_train.shape[1] == 0:
        errors.append("No features found")

    # Check 3: NaN values
    if X_train.isnull().sum().sum() > 0:
        errors.append(f"NaN values in training data: {X_train.isnull().sum().sum()}")

    if y_train.isnull().sum() > 0:
        errors.append(f"NaN values in training labels: {y_train.isnull().sum()}")

    # Check 4: Infinite values
    if np.isinf(X_train.values).sum() > 0:
        errors.append("Infinite values in training data")

    # Check 5: Task-specific checks
    if task_type == 'binary':
        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            errors.append(f"Binary classification requires 2 classes, found: {unique_labels}")

    elif task_type == 'regression':
        if y_train.min() < CONFIG['MIN_RANK'] or y_train.max() > CONFIG['MAX_RANK']:
            errors.append(f"Labels out of range: [{y_train.min()}, {y_train.max()}]")

    # Report errors
    if errors:
        log_warning("Data validation errors:")
        for error in errors:
            log_warning(f"  - {error}")
        return False
    else:
        log_info("Data validation: OK")
        return True
