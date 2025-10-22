"""
データローダー - データベースからのデータ読み込みと整形

このモジュールは、SQLiteデータベースからデータを読み込み、
分析に必要な形式に整形する機能を提供します。
"""

import pandas as pd
import sqlite3
from typing import List, Optional
from .logger import log_info, log_success, log_error, log_warning, log_section
from .config import DIGIT_ORDER


def get_available_tables(db_path: str) -> List[str]:
    """
    データベース内のテーブル一覧を取得

    Args:
        db_path (str): データベースファイルパス

    Returns:
        List[str]: テーブル名リスト
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # テーブル一覧を取得
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )

        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        return tables
    except Exception as e:
        log_error(f"テーブル一覧の取得に失敗: {str(e)}")
        return []


def load_last_digit_data(
    db_path: str,
    table_name: str = 'last_digit_summary_all'
) -> pd.DataFrame:
    """
    last_digit_summaryテーブルをデータベースから読み込み

    Args:
        db_path (str): データベースファイルパス
        table_name (str): テーブル名

    Returns:
        pd.DataFrame: 読み込み済みデータ

    Raises:
        FileNotFoundError: データベースファイルが見つからない場合
        ValueError: テーブルの読み込みに失敗した場合
    """
    try:
        # DB接続
        conn = sqlite3.connect(db_path)

        # データ読込
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} ORDER BY date, last_digit",
                conn
            )
            log_success(f"{table_name}読込: {len(df)}行")
        except Exception as e:
            log_error(f"テーブル '{table_name}' の読込に失敗")
            log_error(f"詳細: {str(e)[:100]}")
            raise ValueError(f"テーブル読み込みエラー: {str(e)}")

        conn.close()

        # データ型の確認と変換
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            except:
                df['date'] = pd.to_datetime(df['date'])

        # 日付範囲
        if 'date' in df.columns:
            log_info(f"日付範囲: {df['date'].min()} ～ {df['date'].max()}")

        if 'last_digit' in df.columns:
            log_info(f"末尾種類: {df['last_digit'].nunique()}種")

        return df

    except FileNotFoundError:
        log_error("データベースが見つかりません")
        log_error(f"ファイル: {db_path}")
        raise
    except Exception as e:
        log_error(f"データベース読込失敗: {str(e)}")
        raise


def find_last_digit_table(db_path: str) -> Optional[str]:
    """
    データベースから末尾データテーブルを自動検出

    Args:
        db_path (str): データベースファイルパス

    Returns:
        Optional[str]: テーブル名（見つからない場合はNone）
    """
    available_tables = get_available_tables(db_path)

    if not available_tables:
        log_warning("利用可能なテーブルが見つかりません")
        return None

    # 優先順位: last_digit_summary_all > last_digit_* > その他
    for priority_pattern in ['last_digit_summary_all', 'last_digit_summary', 'last_digit']:
        for table in available_tables:
            if priority_pattern in table:
                return table

    # フォールバック: 最初のテーブル
    return available_tables[0] if available_tables else None


def load_data_with_events(
    db_path: str,
    table_name: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    メインデータとイベントカレンダーを統合して読み込み

    Args:
        db_path (str): データベースファイルパス
        table_name (str, optional): テーブル名。Noneの場合は自動検出
        validate (bool): データ検証を実行するか

    Returns:
        pd.DataFrame: イベント情報を含むデータ
    """
    log_section("データ読み込み")

    # テーブル名の決定
    if table_name is None:
        table_name = find_last_digit_table(db_path)
        if table_name is None:
            raise ValueError("末尾データテーブルが見つかりません")
        log_info(f"自動検出されたテーブル: {table_name}")

    # テーブル一覧を表示
    available_tables = get_available_tables(db_path)
    log_info(f"利用可能なテーブル ({len(available_tables)}個): {', '.join(available_tables)}")

    # メインデータ読み込み
    df = load_last_digit_data(db_path, table_name)

    # イベントカレンダーと統合
    try:
        conn = sqlite3.connect(db_path)
        df_events = pd.read_sql_query("SELECT * FROM event_calendar ORDER BY date", conn)
        conn.close()

        # dateカラムの型を統一
        if 'date' in df_events.columns:
            df_events['date'] = pd.to_datetime(df_events['date'], format='%Y%m%d', errors='coerce')

        # マージ
        df = df.merge(df_events, on='date', how='left')
        log_success(f"イベントカレンダーと統合完了 ({len(df_events)}日分)")

    except Exception as e:
        log_warning(f"イベントカレンダーの読み込みをスキップ: {str(e)}")

    # データ検証
    if validate:
        _validate_dataframe(df)

    # last_digitをカテゴリ化
    if 'last_digit' in df.columns:
        df['last_digit'] = pd.Categorical(df['last_digit'], categories=DIGIT_ORDER, ordered=True)

    log_success("データ読み込み完了")
    return df


def _validate_dataframe(df: pd.DataFrame):
    """
    データフレームの検証

    Args:
        df (pd.DataFrame): 検証対象のデータフレーム
    """
    log_info("\n【データ検証】")

    # 形状
    log_info(f"データ形状: {df.shape[0]}行 × {df.shape[1]}列")

    # 重要列のチェック
    important_cols = ['last_digit_rank_diff', 'current_diff', 'avg_diff_coins', 'date', 'last_digit']
    log_info("\n重要列チェック:")
    for col in important_cols:
        exists = col in df.columns
        status = '✅' if exists else '❌'
        log_info(f"  {status} {col}")

    # イベントフラグのチェック
    event_flags = [col for col in df.columns if col.startswith('is_')]
    if event_flags:
        log_info(f"\nイベントフラグ: {len(event_flags)}個")
        log_info(f"  例: {', '.join(event_flags[:5])}")

    # データ統計
    log_info("\nデータ統計:")
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        days = (date_max - date_min).days
        log_info(f"  日付範囲: {date_min} ～ {date_max}")
        log_info(f"  日数: {days}日")

    if 'last_digit' in df.columns:
        log_info(f"  末尾種類: {df['last_digit'].nunique()}種")
        log_info(f"  末尾例: {df['last_digit'].unique()[:5].tolist()}")

    if 'avg_diff_coins' in df.columns:
        log_info(f"  差枚平均: {df['avg_diff_coins'].mean():.1f}枚")
        log_info(f"  差枚範囲: {df['avg_diff_coins'].min():.1f} ～ {df['avg_diff_coins'].max():.1f}枚")


def load_multiple_tables(
    db_path: str,
    table_patterns: List[str] = ['last_digit_summary_all', 'last_digit_summary_jug', 'last_digit_summary_other']
) -> dict:
    """
    複数のテーブルを一括読み込み

    Args:
        db_path (str): データベースファイルパス
        table_patterns (List[str]): 読み込むテーブル名のリスト

    Returns:
        dict: {テーブル名: DataFrame}の辞書
    """
    available_tables = get_available_tables(db_path)
    result = {}

    for pattern in table_patterns:
        # パターンに一致するテーブルを検索
        matched_table = None
        for table in available_tables:
            if pattern in table or table in pattern:
                matched_table = table
                break

        if matched_table:
            try:
                df = load_data_with_events(db_path, matched_table, validate=False)
                result[pattern] = df
                log_success(f"{pattern}: 読み込み完了")
            except Exception as e:
                log_warning(f"{pattern}: 読み込みスキップ ({str(e)})")
                result[pattern] = pd.DataFrame()
        else:
            log_warning(f"{pattern}: テーブルが見つかりません")
            result[pattern] = pd.DataFrame()

    return result
