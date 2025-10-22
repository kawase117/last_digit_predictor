"""
ラグ特徴量モジュール - 時系列ラグと移動統計量の生成

このモジュールは、時系列データにおけるラグ特徴量（過去N日の値）、
移動平均、移動標準偏差を生成します。データリーク防止のため、
当日のデータは除外して過去データのみを使用します。
"""

import pandas as pd
import numpy as np
from ..logger import log_info, log_debug


def create_lag_features(df, target_cols, lag_days=[1, 2, 3, 4, 7, 14, 21, 28]):
    """末尾ごとのラグ特徴量を生成（当日を除く過去データのみ）

    Args:
        df (pd.DataFrame): 日付でソートされたデータ（1日11行）
        target_cols (list): ラグ対象カラムリスト
        lag_days (list, optional): ラグ日数. Defaults to [1,2,3,4,7,14,21,28].

    Returns:
        pd.DataFrame: allday_lagX_* カラムを追加したデータ
    """

    df_out = df.copy()
    lag_feature_count = 0

    for target_col in target_cols:
        for lag_day in lag_days:
            # 1日 = 11行なので、lag_day日前 = lag_day * 11行前
            shift_amount = lag_day * 11

            df_out[f'allday_lag{lag_day}_{target_col}'] = (
                df_out.groupby('digit_num')[target_col]
                .shift(shift_amount)
                .values
            )
            lag_feature_count += 1

    log_info(f"ラグ特徴量: {lag_feature_count}個")
    return df_out


def create_moving_avg_std_features(
    df, target_cols,
    window_sizes=[1, 2, 3, 4, 7, 14, 21, 28]
):
    """末尾ごとの移動平均・標準偏差を生成（当日を除く過去データのみ）

    Args:
        df (pd.DataFrame): 日付でソートされたデータ（1日11行）
        target_cols (list): 対象カラムリスト
        window_sizes (list, optional): ウィンドウサイズ. Defaults to [1,2,3,4,7,14,21,28].

    Returns:
        pd.DataFrame: allday_ma/std_* カラムを追加したデータ
    """

    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    ma_feature_count = 0
    std_feature_count = 0

    for target_col in target_cols:
        for window in window_sizes:
            # shift(1)で当日データを除外してから移動平均を計算
            df_out[f'allday_ma{window}_{target_col}'] = (
                df_out.groupby('digit_num')[target_col]
                .shift(1)  # 当日を除外
                .rolling(window=window, min_periods=1)
                .mean()
                .values
            )
            ma_feature_count += 1

            # 標準偏差
            df_out[f'allday_std{window}_{target_col}'] = (
                df_out.groupby('digit_num')[target_col]
                .shift(1)  # 当日を除外
                .rolling(window=window, min_periods=1)
                .std()
                .values
            )
            std_feature_count += 1

    log_info(f"移動平均特徴量: {ma_feature_count}個")
    log_info(f"標準偏差特徴量: {std_feature_count}個")
    return df_out
