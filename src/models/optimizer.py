"""Optuna-based hyperparameter optimization for models.

This module provides unified optimization functions for:
- Binary classification tasks (LogReg, RandomForest, XGBoost, LightGBM)
- Regression tasks (Ridge, RandomForest, XGBoost, LightGBM)
- Cross-validation with pruning
- Task-specific objective functions

The optimization uses TPE sampler and median pruner for efficient search.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Callable
import warnings

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..logger import log_info, log_debug, log_warning
from ..config import CONFIG

warnings.filterwarnings('ignore')


def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'binary',
    ranking_mode: Optional[str] = None,
    n_trials: int = 20,
    cv_folds: int = 3
) -> optuna.study.Study:
    """Run Optuna hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (currently unused, kept for API consistency)
        y_test: Test labels (currently unused, kept for API consistency)
        task_type: Task type ('binary' or 'regression')
        ranking_mode: Ranking mode ('baseline' or 'top3_focus') for regression
        n_trials: Number of Optuna trials
        cv_folds: Number of cross-validation folds

    Returns:
        Optuna study object with best parameters in study.best_params
    """
    log_info("Optuna optimization started")
    log_info(f"  Task type: {task_type}")
    if ranking_mode:
        log_info(f"  Ranking mode: {ranking_mode}")
    log_info(f"  Trials: {n_trials}")

    def objective(trial):
        """Optuna objective function."""
        # Select model type
        model_name = trial.suggest_categorical(
            'model_name',
            ['RandomForest', 'Ridge', 'LightGBM']
        )

        if task_type == 'binary':
            # Binary classification
            if model_name == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )

            elif model_name == 'Ridge':
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 10, log=True),
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )

            elif model_name == 'LightGBM':
                from lightgbm import LGBMClassifier
                pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
                    scale_pos_weight=pos_weight,
                    random_state=42,
                    verbose=-1
                )

            # Cross-validation with F1 score
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='f1_weighted'
            )
            score = cv_scores.mean()

        else:  # Regression
            if model_name == 'RandomForest':
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    random_state=42,
                    n_jobs=-1
                )

            elif model_name == 'Ridge':
                model = Ridge(
                    alpha=trial.suggest_float('alpha', 0.1, 100, log=True),
                    random_state=42
                )

            elif model_name == 'LightGBM':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1),
                    random_state=42,
                    verbose=-1
                )

            # Cross-validation with MAE
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_absolute_error'
            )
            score = -cv_scores.mean()  # Negate for minimization

        return score

    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize' if task_type == 'binary' else 'minimize',
        pruner=MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log_info(f"  Best score: {study.best_value:.4f}")
    log_info(f"  Best parameters: {study.best_params}")

    return study


def create_objective_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Callable:
    """Create objective function for binary classification.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for validation)
        y_test: Test labels (for validation)

    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        model_name = trial.suggest_categorical(
            'model_name',
            ['LogReg', 'RF', 'XGB']
        )

        try:
            if model_name == 'LogReg':
                model = LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 100, log=True),
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == 'RF':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    random_state=42,
                    n_jobs=-1
                )
            else:  # XGB
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=42,
                    verbosity=0
                )

            # Cross-validation
            cv = StratifiedKFold(
                n_splits=CONFIG.get('CV_FOLDS', 3),
                shuffle=False
            )
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='f1_weighted'
            )
            score = cv_scores.mean()

            return score
        except Exception as e:
            log_debug(f"Trial failed: {str(e)[:50]}")
            return 0.0

    return objective


def create_objective_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Callable:
    """Create objective function for regression.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for validation)
        y_test: Test labels (for validation)

    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        model_name = trial.suggest_categorical(
            'model_name',
            ['Ridge', 'RF', 'XGB']
        )

        try:
            if model_name == 'Ridge':
                model = Ridge(
                    alpha=trial.suggest_float('alpha', 0.1, 100, log=True),
                    random_state=42
                )
            elif model_name == 'RF':
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    random_state=42,
                    n_jobs=-1
                )
            else:  # XGB
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=42,
                    verbosity=0
                )

            # Cross-validation with R² score
            cv = KFold(n_splits=CONFIG.get('CV_FOLDS', 3), shuffle=False)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='r2'
            )
            score = cv_scores.mean()

            return score
        except Exception as e:
            log_debug(f"Trial failed: {str(e)[:50]}")
            return -np.inf

    return objective


def optimize_by_task(
    df_merged: pd.DataFrame,
    cv_feature_results: Dict,
    test_events: list,
    n_trials: int = 20
) -> Dict:
    """Optimize hyperparameters for each event and task combination.

    Args:
        df_merged: Merged dataframe with all features
        cv_feature_results: Dictionary of selected features per event/task
        test_events: List of event names to optimize
        n_trials: Number of optimization trials per task

    Returns:
        Dictionary containing optimization results for each event and task
    """
    optuna_results = {
        'top1': {},
        'top2': {},
        'rank': {}
    }

    for event in test_events:
        log_info(f"Event: {event}")

        # Get event data
        event_col = f'is_{event}'
        if event_col not in df_merged.columns:
            log_warning(f"  Column '{event_col}' not found")
            continue

        event_data = df_merged[df_merged[event_col] == 1].copy().reset_index(drop=True)

        if len(event_data) < CONFIG.get('MIN_EVENT_DAYS', 8):
            log_warning(f"  Insufficient data: {len(event_data)} days")
            continue

        # TOP1: Binary classification
        if event in cv_feature_results and 'top1' in cv_feature_results[event]:
            log_info("  TOP1 (binary) optimization...")

            selected_features = cv_feature_results[event]['top1']

            if len(selected_features) == 0:
                log_warning("    No features selected")
            else:
                X = event_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).values
                y_top1 = (event_data['last_digit_rank_diff'].values <= 1).astype(int)

                # Train/test split
                split_idx = int(len(X) * 0.75)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y_top1[:split_idx], y_top1[split_idx:]

                try:
                    # Optuna optimization
                    sampler = TPESampler(seed=42)
                    study = optuna.create_study(sampler=sampler, direction='maximize')
                    objective = create_objective_binary(X_train, y_train, X_test, y_test)

                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                    optuna_results['top1'][event] = {
                        'best_params': study.best_params,
                        'best_score': study.best_value,
                        'n_trials': len(study.trials),
                        'selected_features': selected_features
                    }

                    log_info(f"    Optimization completed")
                    log_info(f"      Best model: {study.best_params.get('model_name', 'N/A')}")
                    log_info(f"      F1 score: {study.best_value:.4f}")

                except Exception as e:
                    log_warning(f"    Error: {str(e)[:50]}")

        # TOP2: Binary classification
        if event in cv_feature_results and 'top2' in cv_feature_results[event]:
            log_info("  TOP2 (binary) optimization...")

            selected_features = cv_feature_results[event]['top2']

            if len(selected_features) == 0:
                log_warning("    No features selected")
            else:
                X = event_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).values
                y_top2 = (event_data['last_digit_rank_diff'].values <= 2).astype(int)

                split_idx = int(len(X) * 0.75)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y_top2[:split_idx], y_top2[split_idx:]

                try:
                    sampler = TPESampler(seed=42)
                    study = optuna.create_study(sampler=sampler, direction='maximize')
                    objective = create_objective_binary(X_train, y_train, X_test, y_test)

                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                    optuna_results['top2'][event] = {
                        'best_params': study.best_params,
                        'best_score': study.best_value,
                        'n_trials': len(study.trials),
                        'selected_features': selected_features
                    }

                    log_info(f"    Optimization completed")
                    log_info(f"      Best model: {study.best_params.get('model_name', 'N/A')}")
                    log_info(f"      F1 score: {study.best_value:.4f}")

                except Exception as e:
                    log_warning(f"    Error: {str(e)[:50]}")

        # RANK: Regression
        if event in cv_feature_results and 'rank' in cv_feature_results[event]:
            log_info("  RANK (regression) optimization...")

            selected_features = cv_feature_results[event]['rank']

            if len(selected_features) == 0:
                log_warning("    No features selected")
            else:
                X = event_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0).values
                y_rank = event_data['last_digit_rank_diff'].values

                split_idx = int(len(X) * 0.75)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y_rank[:split_idx], y_rank[split_idx:]

                try:
                    sampler = TPESampler(seed=42)
                    study = optuna.create_study(sampler=sampler, direction='maximize')
                    objective = create_objective_regression(X_train, y_train, X_test, y_test)

                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                    optuna_results['rank'][event] = {
                        'best_params': study.best_params,
                        'best_score': study.best_value,
                        'n_trials': len(study.trials),
                        'selected_features': selected_features
                    }

                    log_info(f"    Optimization completed")
                    log_info(f"      Best model: {study.best_params.get('model_name', 'N/A')}")
                    log_info(f"      R² score: {study.best_value:.4f}")

                except Exception as e:
                    log_warning(f"    Error: {str(e)[:50]}")

    # Print summary
    log_info("Optuna optimization completed")
    log_info("Optimization results summary:")

    for task in ['top1', 'top2', 'rank']:
        log_info(f"\n{task.upper()}:")
        for event in test_events:
            if event in optuna_results[task]:
                result = optuna_results[task][event]
                log_info(
                    f"  {event:8s}: {result['best_params'].get('model_name', 'N/A'):8s} "
                    f"| Score: {result['best_score']:.4f}"
                )

    return optuna_results
