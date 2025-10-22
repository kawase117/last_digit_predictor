"""
変化量特徴量モジュール - 差分、変化率、ランク変化、prev系特徴量の生成

このモジュールは、時系列データにおける変化量（差分・変化率）、
ランク変化特徴量、およびイベント履歴に基づくprev系特徴量を生成します。
データリーク防止のため、当日のデータは除外して過去データのみを使用します。
"""

import pandas as pd
import numpy as np
from ..logger import log_info, log_debug


def create_change_features(
    df, target_cols,
    change_lags=[1, 7, 14]
):
    """lag_day日前との比較で変化量を生成（差分・変化率）

    Args:
        df (pd.DataFrame): ラグ特徴量が既に追加されているデータ
        target_cols (list): 対象カラムリスト
        change_lags (list, optional): 比較ラグ日数. Defaults to [1, 7, 14].

    Returns:
        pd.DataFrame: allday_lagX_*_diff/pct カラムを追加したデータ
    """

    df_out = df.copy()
    change_feature_count = 0

    for target_col in target_cols:
        for lag_day in change_lags:
            lag_col = f'allday_lag{lag_day}_{target_col}'

            if lag_col in df_out.columns:
                # 差分: 当日値 - lag日前値
                df_out[f'allday_lag{lag_day}_{target_col}_diff'] = (
                    df_out[target_col] - df_out[lag_col]
                )
                change_feature_count += 1

                # 変化率: (当日値 - lag日前値) / |lag日前値|
                df_out[f'allday_lag{lag_day}_{target_col}_pct'] = (
                    (df_out[target_col] - df_out[lag_col]) /
                    (df_out[lag_col].abs() + 1e-10)  # ゼロ除算対策
                )
                change_feature_count += 1

    log_info(f"変化量特徴量: {change_feature_count}個")
    return df_out


def create_rank_change_features(
    df, rank_col='last_digit_rank_diff',
    change_lags=[1, 7, 14],
    stat_windows=[7, 14, 28]
):
    """ランクカラムの変化量・統計量を生成

    Args:
        df (pd.DataFrame): ラグ特徴量が既に追加されているデータ
        rank_col (str, optional): ランクカラム名. Defaults to 'last_digit_rank_diff'.
        change_lags (list, optional): 比較ラグ日数. Defaults to [1, 7, 14].
        stat_windows (list, optional): ウィンドウサイズ. Defaults to [7, 14, 28].

    Returns:
        pd.DataFrame: allday_rank_change_*, allday_rank_max/min/std_* を追加したデータ
    """

    df_out = df.copy()
    rank_feature_count = 0

    if rank_col not in df_out.columns:
        log_info(f"{rank_col} が見つかりません")
        return df_out

    # ランク変化量
    for lag_day in change_lags:
        lag_col = f'allday_lag{lag_day}_{rank_col}'

        if lag_col in df_out.columns:
            # ランク差: 当日ランク - lag日前ランク
            # (ランクが小さい方が良い)
            df_out[f'allday_rank_change{lag_day}'] = (
                df_out[rank_col] - df_out[lag_col]
            )
            rank_feature_count += 1

    # ランク統計量
    for window in stat_windows:
        df_out[f'allday_rank_max{window}'] = (
            df_out.groupby('digit_num')[rank_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .max()
            .values
        )
        rank_feature_count += 1

        df_out[f'allday_rank_min{window}'] = (
            df_out.groupby('digit_num')[rank_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .min()
            .values
        )
        rank_feature_count += 1

        df_out[f'allday_rank_std{window}'] = (
            df_out.groupby('digit_num')[rank_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .std()
            .values
        )
        rank_feature_count += 1

    log_info(f"ランク変化特徴量: {rank_feature_count}個")
    return df_out


def create_prev_features(
    df, available_events,
    metric_cols=['avg_diff_coins', 'avg_games', 'win_rate',
                 'high_profit_rate', 'last_digit_rank_diff',
                 'last_digit_rank_games'],
    exclude_cols=['max_games', 'min_games']  # リーク防止
):
    """イベント履歴から過去データのprev_*特徴量を生成

    当日値は除外してリークを防止します。

    Args:
        df (pd.DataFrame): イベントフラグ(is_*)が含まれるデータ
        available_events (list): イベント名リスト
        metric_cols (list, optional): 特徴量対象カラム
        exclude_cols (list, optional): リーク防止のため最初から生成しないカラム. Defaults to ['max_games', 'min_games'].

    Returns:
        pd.DataFrame: prev_* カラムを追加したデータ
    """

    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    # リーク防止: 対象カラムから除外カラムを削除
    metric_cols = [col for col in metric_cols if col not in exclude_cols]

    event_history = {}
    prev_feature_count = 0

    # イベント履歴の構築
    for idx, row in df_out.iterrows():
        current_date = row['date']
        current_digit = row['digit_num']

        for event in available_events:
            flag_col = f'is_{event}'

            if flag_col in df_out.columns and row[flag_col] == 1:
                key = (event, current_digit)

                if key not in event_history:
                    event_history[key] = []

                event_record = {'date': current_date}
                for metric in metric_cols:
                    if metric in df_out.columns:
                        event_record[metric] = row[metric]

                event_history[key].append(event_record)

    # prev_1, prev_2, prev_3 の基本特徴量生成
    feature_rows = []

    for idx, row in df_out.iterrows():
        current_digit = row['digit_num']
        row_features = {}

        for event in available_events:
            key = (event, current_digit)

            # prev_1, prev_2, prev_3 (前回、前々回、前々々回)
            for prev_n in [1, 2, 3]:
                if key in event_history and len(event_history[key]) > prev_n:
                    prev_record = event_history[key][-prev_n]

                    for metric in metric_cols:
                        if metric in prev_record:
                            feature_name = f'prev_{prev_n}_{metric}'
                            row_features[feature_name] = prev_record[metric]
                            prev_feature_count = prev_feature_count + 1 if feature_name not in row_features else prev_feature_count

        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    df_out = pd.concat([df_out.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    log_info(f"prev基本特徴量: {len(features_df.columns)}個")
    return df_out
