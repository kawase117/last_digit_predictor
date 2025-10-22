"""
Last Digit Predictor - パチスロ末尾予測システム

このパッケージは、パチスロ店舗の末尾データを分析し、
次回イベント時の出玉を予測する機械学習システムを提供します。

主な機能:
- データ読み込み・整形
- 特徴量エンジニアリング（prev系、ラグ、変化量、補助特徴量）
- モデル訓練・最適化（Optuna）
- 予測・評価
- 可視化

基本的な使い方:
```python
from src import load_data_with_events
from src.features import create_lag_features
from src.models import prepare_unified_data, run_optuna_optimization

# データ読み込み
df = load_data_with_events('pachinko_analysis_マルハンメガシティ柏.db')

# 特徴量生成
df = create_lag_features(df, ['avg_diff_coins'], [1, 2, 3])

# モデル訓練
X_train, y_train, X_test, y_test = prepare_unified_data(df, '5day', 'binary')
results = run_optuna_optimization(X_train, y_train, X_test, y_test, 'binary')
```
"""

__version__ = '1.0.0'

# 設定・ロガー
from .config import (
    CONFIG,
    EVENT_DEFINITIONS,
    DIGIT_ORDER,
    get_config,
    update_config,
    check_library_availability,
)

from .logger import (
    setup_logger,
    get_logger,
    log_info,
    log_debug,
    log_warning,
    log_error,
    log_success,
    log_section,
)

# データ読み込み
from .data_loader import (
    get_available_tables,
    load_last_digit_data,
    load_data_with_events,
    load_multiple_tables,
)

# 特徴量
from .features import (
    # prev
    build_event_history,
    create_prev_basic_features,
    create_prev_change_features,
    create_prev_stat_features,
    create_prev_trend_features,
    # lag
    create_lag_features,
    create_moving_avg_std_features,
    # change
    create_change_features,
    create_rank_change_features,
    create_prev_features,
    # auxiliary
    create_auxiliary_features,
)

# モデル
from .models import (
    # label
    create_top_labels,
    create_rank_labels,
    prepare_unified_data,
    validate_data,
    # feature selection
    select_features_unified,
    filter_features_by_pvalue,
    # optimization
    run_optuna_optimization,
    create_objective_binary,
    create_objective_regression,
    # prediction
    make_predictions,
    make_predictions_batch,
    apply_prediction_filter,
    ensemble_predictions,
)

# 評価
from .evaluation import (
    evaluate_unified_metrics,
    log_unified_metrics,
    get_best_metric,
)

# ユーティリティ
from .utils import (
    validate_event,
    print_progress,
    log_error_with_context,
    print_config,
    diagnose_error,
    compare_metrics,
)

# 可視化
from .visualization import (
    plot_model_performance_comparison,
    plot_cumulative_profit,
    plot_confusion_matrix,
    plot_event_comparison_dashboard,
    plot_profit_heatmap,
    plot_feature_importance,
    save_figure,
    create_comparison_table,
)


__all__ = [
    # version
    '__version__',
    # config
    'CONFIG',
    'EVENT_DEFINITIONS',
    'DIGIT_ORDER',
    'get_config',
    'update_config',
    'check_library_availability',
    # logger
    'setup_logger',
    'get_logger',
    'log_info',
    'log_debug',
    'log_warning',
    'log_error',
    'log_success',
    'log_section',
    # data_loader
    'get_available_tables',
    'load_last_digit_data',
    'load_data_with_events',
    'load_multiple_tables',
    # features - prev
    'build_event_history',
    'create_prev_basic_features',
    'create_prev_change_features',
    'create_prev_stat_features',
    'create_prev_trend_features',
    # features - lag
    'create_lag_features',
    'create_moving_avg_std_features',
    # features - change
    'create_change_features',
    'create_rank_change_features',
    'create_prev_features',
    # features - auxiliary
    'create_auxiliary_features',
    # models - label
    'create_top_labels',
    'create_rank_labels',
    'prepare_unified_data',
    'validate_data',
    # models - feature selection
    'select_features_unified',
    'filter_features_by_pvalue',
    # models - optimization
    'run_optuna_optimization',
    'create_objective_binary',
    'create_objective_regression',
    # models - prediction
    'make_predictions',
    'make_predictions_batch',
    'apply_prediction_filter',
    'ensemble_predictions',
    # evaluation
    'evaluate_unified_metrics',
    'log_unified_metrics',
    'get_best_metric',
    # utils
    'validate_event',
    'print_progress',
    'log_error_with_context',
    'print_config',
    'diagnose_error',
    'compare_metrics',
    # visualization
    'plot_model_performance_comparison',
    'plot_cumulative_profit',
    'plot_confusion_matrix',
    'plot_event_comparison_dashboard',
    'plot_profit_heatmap',
    'plot_feature_importance',
    'save_figure',
    'create_comparison_table',
]
