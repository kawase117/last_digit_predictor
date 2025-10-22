"""
ロギングシステム - 全てのprint文をloggingモジュールで置き換え

このモジュールは、print文の代わりに使用する統一的なロギングシステムを提供します。
ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）を使い分けて、
適切な情報を適切なタイミングで出力します。
"""

import logging
import sys
from typing import Optional


# ============================================================
# ロガーの設定
# ============================================================

def setup_logger(
    name: str = 'last_digit_predictor',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    ロガーをセットアップ

    Args:
        name (str): ロガー名
        level (int): ログレベル（logging.DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file (str, optional): ログファイルのパス。Noneの場合はコンソール出力のみ
        format_string (str, optional): ログフォーマット。Noneの場合はデフォルト

    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 既存のハンドラをクリア（重複防止）
    if logger.handlers:
        logger.handlers.clear()

    # フォーマッタの設定
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラ（オプション）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# デフォルトロガー
# ============================================================

# シンプルなフォーマット（Jupyter Notebook用）
_default_logger = setup_logger(
    name='last_digit_predictor',
    level=logging.INFO,
    format_string='[%(levelname)s] %(message)s'
)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    ロガーを取得

    Args:
        name (str, optional): ロガー名。Noneの場合はデフォルトロガー

    Returns:
        logging.Logger: ロガー
    """
    if name is None:
        return _default_logger
    return logging.getLogger(name)


# ============================================================
# 便利な関数（print文の代替）
# ============================================================

def log_info(message: str, logger: Optional[logging.Logger] = None):
    """
    INFO レベルのログを出力（print の代替）

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.info(message)


def log_debug(message: str, logger: Optional[logging.Logger] = None):
    """
    DEBUG レベルのログを出力（デバッグ用print の代替）

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.debug(message)


def log_warning(message: str, logger: Optional[logging.Logger] = None):
    """
    WARNING レベルのログを出力

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.warning(message)


def log_error(message: str, logger: Optional[logging.Logger] = None, exc_info: bool = False):
    """
    ERROR レベルのログを出力

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
        exc_info (bool): 例外情報を含めるか
    """
    if logger is None:
        logger = _default_logger
    logger.error(message, exc_info=exc_info)


def log_critical(message: str, logger: Optional[logging.Logger] = None):
    """
    CRITICAL レベルのログを出力

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.critical(message)


def log_section(title: str, logger: Optional[logging.Logger] = None, width: int = 70):
    """
    セクションヘッダーを出力（元のprint("="*70)の代替）

    Args:
        title (str): セクションタイトル
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
        width (int): セパレータの幅
    """
    if logger is None:
        logger = _default_logger
    logger.info("=" * width)
    logger.info(f"【{title}】")
    logger.info("=" * width)


def log_progress(current: int, total: int, task_name: str = '', logger: Optional[logging.Logger] = None):
    """
    進捗状況を出力

    Args:
        current (int): 現在の進捗
        total (int): 全体数
        task_name (str): タスク名
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger

    percentage = (current / total * 100) if total > 0 else 0
    task_label = f"[{task_name}] " if task_name else ""
    logger.info(f"{task_label}進捗: {current}/{total} ({percentage:.1f}%)")


def log_success(message: str, logger: Optional[logging.Logger] = None):
    """
    成功メッセージを出力（元の print("✅ ...")の代替）

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.info(f"✅ {message}")


def log_failure(message: str, logger: Optional[logging.Logger] = None):
    """
    失敗メッセージを出力（元の print("❌ ...")の代替）

    Args:
        message (str): メッセージ
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.error(f"❌ {message}")


def set_log_level(level: int, logger: Optional[logging.Logger] = None):
    """
    ログレベルを変更

    Args:
        level (int): ログレベル（logging.DEBUG, INFO, WARNING, ERROR, CRITICAL）
        logger (logging.Logger, optional): ロガー。Noneの場合はデフォルト
    """
    if logger is None:
        logger = _default_logger
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# ============================================================
# 使用例（コメントとして残す）
# ============================================================

"""
# 基本的な使い方
from src.logger import log_info, log_warning, log_error, log_success, log_section

# INFO（通常のprint代替）
log_info("データ読み込み開始")

# セクションヘッダー
log_section("データ準備フェーズ")

# 成功
log_success("データ読み込み完了")

# 警告
log_warning("一部のデータが欠損しています")

# エラー
log_error("ファイルが見つかりません")

# デバッグ（デフォルトでは表示されない）
log_debug("変数の値: x=10, y=20")

# ログレベルを変更してデバッグを表示
set_log_level(logging.DEBUG)
log_debug("この情報が表示されます")
"""
