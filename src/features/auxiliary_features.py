"""
補助特徴量モジュール - 時系列位置、曜日、距離、イベントマッチング特徴量の生成

このモジュールは、時系列位置（データセット内の日数）、曜日フラグ、
ターゲット末尾との距離、イベントマッチング（イベントと末尾の組み合わせ）
などの補助的な特徴量を生成します。
"""

import pandas as pd
import numpy as np
from ..logger import log_info, log_debug


def create_auxiliary_features(df, available_events):
    """時系列位置系、曜日系、距離系、イベントマッチング系を一括生成

    Args:
        df (pd.DataFrame): 基本データ
        available_events (list): イベント名リスト

    Returns:
        pd.DataFrame: days_since_start, weekday_*, distance_*, match_* を追加したデータ
    """

    df_out = df.copy()
    auxiliary_feature_count = 0

    # ============================================================
    # ステップ1: 時系列位置系特徴量
    # ============================================================

    log_debug("時系列位置系特徴量生成中...")

    df_out['date_temp'] = pd.to_datetime(df_out['date'], format='%Y%m%d')
    min_date = df_out['date_temp'].min()
    max_date = df_out['date_temp'].max()

    df_out['days_since_start'] = (df_out['date_temp'] - min_date).dt.days
    df_out['days_to_end'] = (max_date - df_out['date_temp']).dt.days
    df_out['day_of_month'] = df_out['date_temp'].dt.day

    time_position_cols = ['days_since_start', 'days_to_end', 'day_of_month']
    auxiliary_feature_count += len(time_position_cols)

    # ============================================================
    # ステップ2: 曜日系特徴量
    # ============================================================

    log_debug("曜日系特徴量生成中...")

    weekday_names = {0: 'monday', 1: 'tuesday', 2: 'wednesday',
                     3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}
    df_out['weekday_num'] = df_out['date_temp'].dt.dayofweek

    weekday_cols = []
    for day_num, day_name in weekday_names.items():
        col_name = f'is_weekday_{day_name}'
        df_out[col_name] = (df_out['weekday_num'] == day_num).astype(int)
        weekday_cols.append(col_name)
        auxiliary_feature_count += 1

    # ============================================================
    # ステップ3: 距離系特徴量
    # ============================================================

    log_debug("距離系特徴量生成中...")

    distance_cols = []
    for target_digit in range(11):  # 0-9, 10=ゾロ目
        df_out[f'distance_from_{target_digit}'] = (
            df_out['digit_num'] - target_digit
        ).abs()
        distance_cols.append(f'distance_from_{target_digit}')
        auxiliary_feature_count += 1

    # ============================================================
    # ステップ4: イベントマッチング系特徴量
    # ============================================================

    log_debug("イベントマッチング系特徴量生成中...")

    match_cols = []
    for event in available_events:
        flag_col = f'is_{event}'

        if flag_col in df_out.columns:
            if event.endswith('day') and event != 'zorome':
                # 通常イベント（1day～9day）
                try:
                    day_num = int(event[0])
                    df_out[f'match_{event}'] = (
                        (df_out[flag_col] == 1) &
                        (df_out['digit_num'] == day_num)
                    ).astype(int)
                except:
                    pass

            elif event == '39day':
                # 3と9の複合イベント
                df_out[f'match_{event}'] = (
                    (df_out[flag_col] == 1) &
                    ((df_out['digit_num'] == 3) | (df_out['digit_num'] == 9))
                ).astype(int)

            elif event == '40day':
                # 4と0の複合イベント
                df_out[f'match_{event}'] = (
                    (df_out[flag_col] == 1) &
                    ((df_out['digit_num'] == 4) | (df_out['digit_num'] == 0))
                ).astype(int)

            elif event == 'zorome':
                # ゾロ目イベント（末尾が10=ゾロ目）
                df_out[f'match_{event}'] = (
                    (df_out[flag_col] == 1) &
                    (df_out['digit_num'] == 10)
                ).astype(int)

        match_feature_count = len([e for e in available_events
                                   if 'day' in e or e == 'zorome'])
        auxiliary_feature_count += match_feature_count

    # ============================================================
    # 一時カラムと文字列カラムの削除
    # ============================================================

    # date_temp を削除
    df_out = df_out.drop('date_temp', axis=1)

    # その他のobject型（文字列）カラムを確認・削除
    object_cols = df_out.select_dtypes(include=['object']).columns.tolist()
    if 'date' in object_cols:
        object_cols.remove('date')  # dateは後で削除する（dateカラムは保持）

    if object_cols:
        log_info(f"文字列カラムが存在: {object_cols}")
        log_info("削除していません（後のセルで処理予定）")

    log_info(f"補助特徴量: {auxiliary_feature_count}個")
    return df_out
