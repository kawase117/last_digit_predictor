"""
ユーティリティ関数モジュール - データ検証・診断・比較機能

このモジュールは、データの妥当性チェック、エラー診断、設定表示、
メトリクス比較など、分析作業を支援するユーティリティ関数を提供します。

主要機能:
    - イベントデータの妥当性検証
    - 進捗表示
    - エラーログ記録（コンテキスト付き）
    - CONFIG設定の表示
    - データ・特徴量・モデルのエラー診断
    - 複数イベントの評価指標比較
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

from .logger import log_info, log_warning, log_error, log_debug


def validate_event(
    df: pd.DataFrame,
    event: str,
    task_type: str = 'binary',
    rank: int = 1
) -> bool:
    """
    イベントの妥当性チェック

    Args:
        df (pd.DataFrame): マージ済みデータ
        event (str): イベント名
        task_type (str): 'binary' or 'regression'
        rank (int): TOP順位（二値分類の場合）

    Returns:
        bool: 妥当な場合True、問題がある場合False
    """

    log_info(f"\nイベント検証: {event}")

    errors = []
    warnings = []

    # チェック1: イベントが存在するか
    if 'event_type' in df.columns:
        event_count = (df['event_type'] == event).sum()
        if event_count == 0:
            errors.append(f"イベントが存在しません: {event}")
        else:
            log_info(f"   イベント存在: {event_count}行")

    # チェック2: ラベルの分布
    if 'last_digit_rank_diff' in df.columns:
        if task_type == 'binary':
            labels = (df['last_digit_rank_diff'] <= rank).astype(int)
            label_dist = labels.value_counts().to_dict()

            if 0 not in label_dist or 1 not in label_dist:
                errors.append(f"ラベル分布が偏っています: {label_dist}")
            else:
                ratio = label_dist[1] / len(labels) * 100
                if ratio < 10 or ratio > 90:
                    warnings.append(f"ラベル不均衡: {ratio:.1f}%")
                log_info(f"   ラベル分布: {label_dist}")

        else:
            rank_stats = df['last_digit_rank_diff'].describe()
            log_info(f"   ランク統計: mean={rank_stats['mean']:.2f}, std={rank_stats['std']:.2f}")

    # チェック3: 特徴量の量
    exclude_patterns = [
        'date', 'event', 'target', 'label', 'current_diff',
        'last_digit_rank', 'digit_num', 'last_digit'
    ]

    feature_count = sum(
        1 for col in df.columns
        if not any(p in col.lower() for p in exclude_patterns)
        and df[col].dtype in ['int64', 'float64']
    )

    if feature_count < 20:
        warnings.append(f"特徴量が少なめ: {feature_count}個")
    else:
        log_info(f"   特徴量数: {feature_count}個")

    # 出力
    if errors:
        log_error("\n   エラー:")
        for err in errors:
            log_error(f"      • {err}")
        return False

    if warnings:
        log_warning("\n   警告:")
        for warn in warnings:
            log_warning(f"      • {warn}")

    return True


def print_progress(
    current: int,
    total: int,
    task_name: str = ''
) -> None:
    """
    進捗表示（プログレスバー付き）

    Args:
        current (int): 現在の処理番号
        total (int): 総処理数
        task_name (str): タスク名
    """

    progress = current / total * 100
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r[{bar}] {progress:.1f}% ({current}/{total}) {task_name}", end='')

    if current == total:
        print()  # 改行


def log_error_with_context(
    error_type: str,
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    エラーログ記録（コンテキスト情報付き）

    Args:
        error_type (str): エラー種類
        message (str): エラーメッセージ
        context (Optional[Dict[str, Any]]): コンテキスト情報
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log_error(f"\n[{timestamp}] {error_type}")
    log_error(f"   メッセージ: {message}")

    if context:
        log_error(f"   コンテキスト:")
        for key, value in context.items():
            log_error(f"      • {key}: {value}")


def print_config(config: Dict[str, Any]) -> None:
    """
    CONFIG設定値を見やすく表示

    Args:
        config (Dict[str, Any]): CONFIG辞書
    """

    log_info("\nCONFIG設定一覧")
    log_info("=" * 60)

    for key, value in config.items():
        if isinstance(value, dict):
            log_info(f"\n{key}:")
            for k, v in value.items():
                log_info(f"  • {k}: {v}")
        elif isinstance(value, list):
            log_info(f"\n{key}:")
            for item in value:
                log_info(f"  • {item}")
        else:
            log_info(f"{key}: {value}")

    log_info("=" * 60)


def diagnose_error(
    error_type: str,
    df: Optional[pd.DataFrame] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[np.ndarray] = None
) -> None:
    """
    エラーの原因を診断

    Args:
        error_type (str): エラー種類 ('data', 'features', 'model', etc.)
        df (Optional[pd.DataFrame]): データフレーム
        X_train (Optional[pd.DataFrame]): 訓練特徴量
        y_train (Optional[np.ndarray]): 訓練ラベル
    """

    log_info(f"\nエラー診断: {error_type}")
    log_info("=" * 60)

    if error_type == 'data' and df is not None:
        log_info("\n【データ診断】")
        log_info(f"  形状: {df.shape}")
        log_info(f"  カラム数: {len(df.columns)}")
        log_info(f"  行数: {len(df)}")
        log_info(f"  メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        log_info("\n  NaN値:")
        nan_info = df.isnull().sum()
        if nan_info.sum() > 0:
            log_info(f"    • {nan_info[nan_info > 0].to_dict()}")
        else:
            log_info(f"    • なし")

        log_info("\n  データ型:")
        for dtype in df.dtypes.unique():
            count = (df.dtypes == dtype).sum()
            log_info(f"    • {dtype}: {count}列")

    elif error_type == 'features' and X_train is not None:
        log_info("\n【特徴量診断】")
        log_info(f"  形状: {X_train.shape}")
        log_info(f"  特徴量数: {X_train.shape[1]}")
        log_info(f"  サンプル数: {X_train.shape[0]}")

        log_info("\n  特徴量の統計:")
        if isinstance(X_train, pd.DataFrame):
            stats_df = X_train.describe().T[['mean', 'std', 'min', 'max']]
            log_info(f"\n{stats_df.to_string()}")
        else:
            log_info(f"    • 平均: {X_train.mean(axis=0)}")
            log_info(f"    • 標準偏差: {X_train.std(axis=0)}")

    elif error_type == 'model' and y_train is not None:
        log_info("\n【ラベル診断】")
        log_info(f"  サンプル数: {len(y_train)}")
        log_info(f"  ユニーク値: {len(np.unique(y_train))}")
        log_info(f"  分布: {pd.Series(y_train).value_counts().to_dict()}")

    log_info("\n" + "=" * 60)


def compare_metrics(
    metrics_dict: Dict[str, Dict[str, float]],
    event_names: Optional[List[str]] = None
) -> None:
    """
    複数イベントの評価指標を比較表示

    Args:
        metrics_dict (Dict[str, Dict[str, float]]): {event: metrics} の形式
        event_names (Optional[List[str]]): 表示順序（Noneの場合は辞書順）
    """

    if event_names is None:
        event_names = list(metrics_dict.keys())

    log_info("\n評価指標比較")
    log_info("=" * 80)

    # 共通キーを抽出
    all_keys = set()
    for metrics in metrics_dict.values():
        all_keys.update(metrics.keys())

    # 比較表を作成
    comparison_data = []
    for event in event_names:
        if event not in metrics_dict:
            continue

        metrics = metrics_dict[event]
        row = {'Event': event}

        # 主要指標のみ表示
        for key in ['mae', 'rmse', 'f1', 'accuracy', 'top3_hit_rate', 'spearman_corr']:
            if key in metrics:
                row[key] = metrics[key]

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # フォーマット表示
    log_info(comparison_df.to_string(index=False))
    log_info("=" * 80)
