"""
モデルモジュール

このパッケージは、機械学習モデルの訓練・予測・評価機能を提供します。

モジュール構成:
- label_creator: ラベル作成（TOP1/TOP2/ランク）
- feature_selector: 特徴量選択（Lasso/F-test/Tree-based）
- optimizer: Optuna最適化
- predictor: 予測・アンサンブル
"""

from .label_creator import (
    create_top_labels,
    create_rank_labels,
    prepare_unified_data,
    validate_data,
)

from .feature_selector import (
    select_features_unified,
    filter_features_by_pvalue,
)

from .optimizer import (
    run_optuna_optimization,
    create_objective_binary,
    create_objective_regression,
)

from .predictor import (
    make_predictions,
    make_predictions_batch,
    apply_prediction_filter,
    ensemble_predictions,
)

__all__ = [
    # label_creator
    'create_top_labels',
    'create_rank_labels',
    'prepare_unified_data',
    'validate_data',
    # feature_selector
    'select_features_unified',
    'filter_features_by_pvalue',
    # optimizer
    'run_optuna_optimization',
    'create_objective_binary',
    'create_objective_regression',
    # predictor
    'make_predictions',
    'make_predictions_batch',
    'apply_prediction_filter',
    'ensemble_predictions',
]
