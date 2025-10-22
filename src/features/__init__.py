"""
特徴量エンジニアリングモジュール

このパッケージは、パチスロ分析用の特徴量生成機能を提供します。

モジュール構成:
- prev_features: イベント履歴に基づく特徴量（リーク防止済み）
- lag_features: ラグ特徴量と移動平均
- change_features: 変化量・差分特徴量
- auxiliary_features: 補助特徴量（時系列・曜日・距離など）
"""

from .prev_features import (
    build_event_history,
    create_prev_basic_features,
    create_prev_change_features,
    create_prev_stat_features,
    create_prev_trend_features,
)

from .lag_features import (
    create_lag_features,
    create_moving_avg_std_features,
)

from .change_features import (
    create_change_features,
    create_rank_change_features,
    create_prev_features,
)

from .auxiliary_features import (
    create_auxiliary_features,
)

__all__ = [
    # prev_features
    'build_event_history',
    'create_prev_basic_features',
    'create_prev_change_features',
    'create_prev_stat_features',
    'create_prev_trend_features',
    # lag_features
    'create_lag_features',
    'create_moving_avg_std_features',
    # change_features
    'create_change_features',
    'create_rank_change_features',
    'create_prev_features',
    # auxiliary_features
    'create_auxiliary_features',
]
