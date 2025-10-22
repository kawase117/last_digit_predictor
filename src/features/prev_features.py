"""
prev系特徴量生成（リーク防止済み）

イベント履歴に基づく特徴量を生成します。
- prev_1, prev_2, prev_3: 前回、前々回、前々々回の値
- prev変化量: prev_1とprev_2の差
- prev統計量: 過去N回の最大/最小/平均/標準偏差
- prevトレンド: 過去3回の値のトレンド判定
"""

import pandas as pd
import numpy as np
from ..logger import log_info, log_debug


def build_event_history(df, available_events, metric_cols):
    """
    イベント発生時の履歴を辞書で構築

    Args:
        df (pd.DataFrame): date, digit_num, is_* フラグを含むデータ
        available_events (list): イベント名リスト
        metric_cols (list): 記録対象のメトリクスカラム

    Returns:
        dict: event_history = {(event, digit_num): [履歴リスト]}
    """
    event_history = {}

    for idx, row in df.iterrows():
        current_date = row['date']
        current_digit = row['digit_num']

        for event in available_events:
            flag_col = f'is_{event}'

            if flag_col in df.columns and row[flag_col] == 1:
                key = (event, current_digit)

                if key not in event_history:
                    event_history[key] = []

                # イベント発生時のメトリクスを記録
                event_record = {'date': current_date}
                for metric in metric_cols:
                    if metric in df.columns:
                        event_record[metric] = row[metric]

                event_history[key].append(event_record)

    return event_history


def create_prev_basic_features(df, available_events, metric_cols):
    """
    前回、前々回、前々々回のイベント履歴特徴量を生成

    Args:
        df (pd.DataFrame): イベントフラグを含むデータ
        available_events (list): イベント名リスト
        metric_cols (list): 記録対象のメトリクスカラム

    Returns:
        pd.DataFrame: prev_1_*, prev_2_*, prev_3_* を追加したデータ
    """
    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    # イベント履歴の構築
    event_history = build_event_history(df_out, available_events, metric_cols)

    # prev_1, prev_2, prev_3 の生成
    feature_rows = []

    for idx, row in df_out.iterrows():
        current_digit = row['digit_num']
        row_features = {}

        for event in available_events:
            key = (event, current_digit)

            # 前回(1), 前々回(2), 前々々回(3)
            for prev_n in [1, 2, 3]:
                if key in event_history and len(event_history[key]) > prev_n:
                    prev_record = event_history[key][-prev_n]

                    for metric in metric_cols:
                        if metric in prev_record:
                            feature_name = f'prev_{prev_n}_{metric}'
                            row_features[feature_name] = prev_record[metric]

        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    df_out = pd.concat([df_out.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    log_info(f"prev基本特徴量: {len(features_df.columns)}個")
    return df_out


def create_prev_change_features(df, available_events, metric_cols):
    """
    prev_1 と prev_2 の差分から変化量特徴量を生成

    Args:
        df (pd.DataFrame): prev基本特徴量を含むデータ
        available_events (list): イベント名リスト
        metric_cols (list): 対象メトリクスカラム

    Returns:
        pd.DataFrame: prev_*_change カラムを追加したデータ
    """
    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    event_history = build_event_history(df_out, available_events, metric_cols)

    feature_rows = []

    for idx, row in df_out.iterrows():
        current_digit = row['digit_num']
        row_features = {}

        for event in available_events:
            key = (event, current_digit)

            if key in event_history and len(event_history[key]) >= 2:
                # prev_1 と prev_2 の差を計算
                prev_1 = event_history[key][-1]
                prev_2 = event_history[key][-2]

                for metric in metric_cols:
                    if metric in prev_1 and metric in prev_2:
                        change = prev_1[metric] - prev_2[metric]
                        feature_name = f'prev_1_{metric}_change'
                        row_features[feature_name] = change

        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    df_out = pd.concat([df_out.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    log_info(f"prev変化量特徴量: {len(features_df.columns)}個")
    return df_out


def create_prev_stat_features(df, available_events, metric_cols, windows=[3, 5]):
    """
    過去N回の最大値・最小値・平均・標準偏差を生成

    注意: metric_cols は過去イベント時の値のみ（当日値は除外）

    Args:
        df (pd.DataFrame): イベント履歴を含むデータ
        available_events (list): イベント名リスト
        metric_cols (list): 対象メトリクスカラム（ランク系のみ推奨）
        windows (list): 集計ウィンドウ（デフォルト: [3, 5]）

    Returns:
        pd.DataFrame: prev_max/min/avg/std_* カラムを追加したデータ
    """
    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    event_history = build_event_history(df_out, available_events, metric_cols)

    feature_rows = []

    for idx, row in df_out.iterrows():
        current_digit = row['digit_num']
        row_features = {}

        for event in available_events:
            key = (event, current_digit)

            for window in windows:
                if key in event_history and len(event_history[key]) >= window:
                    recent_records = event_history[key][-window:]

                    for metric in metric_cols:
                        values = [r[metric] for r in recent_records if metric in r]

                        if len(values) > 0:
                            # 最大値
                            feature_name = f'prev_max{window}_{metric}'
                            row_features[feature_name] = max(values)

                            # 最小値
                            feature_name = f'prev_min{window}_{metric}'
                            row_features[feature_name] = min(values)

                            # 平均値
                            feature_name = f'prev_avg{window}_{metric}'
                            row_features[feature_name] = np.mean(values)

                            # 標準偏差
                            feature_name = f'prev_std{window}_{metric}'
                            row_features[feature_name] = np.std(values)

        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    df_out = pd.concat([df_out.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    log_info(f"prev統計量特徴量: {len(features_df.columns)}個")
    return df_out


def create_prev_trend_features(df, available_events, metric_cols=['avg_diff_coins', 'last_digit_rank_diff']):
    """
    過去3回の差枚・ランク改善トレンドを生成

    Args:
        df (pd.DataFrame): イベント履歴を含むデータ
        available_events (list): イベント名リスト
        metric_cols (list): トレンド対象メトリクス

    Returns:
        pd.DataFrame: prev_*_trend_3 カラムを追加したデータ
    """
    df_out = df.copy()
    df_out = df_out.sort_values(['date', 'digit_num']).reset_index(drop=True)

    event_history = build_event_history(df_out, available_events, metric_cols)

    feature_rows = []

    for idx, row in df_out.iterrows():
        current_digit = row['digit_num']
        row_features = {}

        for event in available_events:
            key = (event, current_digit)

            # 差枚トレンド（過去3回）
            if key in event_history and len(event_history[key]) >= 3:
                recent_diff = [r['avg_diff_coins'] for r in event_history[key][-3:]
                              if 'avg_diff_coins' in r]

                if len(recent_diff) == 3:
                    if recent_diff[2] > recent_diff[1] > recent_diff[0]:
                        trend = 1  # 上昇
                    elif recent_diff[2] < recent_diff[1] < recent_diff[0]:
                        trend = -1  # 下降
                    else:
                        trend = 0  # 横ばい

                    row_features[f'prev_diff_trend_3'] = trend

            # ランク改善トレンド（過去3回）
            if key in event_history and len(event_history[key]) >= 3:
                recent_ranks = [r['last_digit_rank_diff'] for r in event_history[key][-3:]
                               if 'last_digit_rank_diff' in r]

                if len(recent_ranks) == 3 and all(isinstance(r, (int, float)) for r in recent_ranks):
                    # ランク改善: 数値が小さくなっている
                    if recent_ranks[2] < recent_ranks[1] < recent_ranks[0]:
                        row_features[f'prev_rank_improving_trend_3'] = 1
                    elif recent_ranks[2] > recent_ranks[1] > recent_ranks[0]:
                        row_features[f'prev_rank_declining_trend_3'] = 1

        feature_rows.append(row_features)

    features_df = pd.DataFrame(feature_rows)
    df_out = pd.concat([df_out.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    log_info(f"prevトレンド特徴量: {len(features_df.columns)}個")
    return df_out
