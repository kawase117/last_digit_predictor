"""
評価指標モジュール - 統一的な評価関数群

このモジュールは、二値分類と回帰（ランク学習）の両方に対応した
統一的な評価指標の計算と表示を提供します。

主要機能:
    - 二値分類指標（Accuracy, F1, Precision, Recall, ROC-AUC）
    - 回帰指標（MAE, RMSE, Spearman相関）
    - ランク学習特化指標（TOP3命中率）
    - 利益指標（平均利益、利益効率）
    - 総合スコア計算（モデル比較用）
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from scipy.stats import spearmanr

from .logger import log_info, log_warning, log_debug


def evaluate_unified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_data: pd.DataFrame,
    task_type: str = 'binary',
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    統一フォーマットの評価指標を計算

    Args:
        y_true (np.ndarray): 真実ラベル
        y_pred (np.ndarray): 予測値
        test_data (pd.DataFrame): テストデータ（利益計算用）
        task_type (str): 'binary' (二値分類) or 'regression' (回帰)
        config (Optional[Dict[str, Any]]): CONFIG辞書（MIN_RANK, MAX_RANKの取得用）

    Returns:
        Dict[str, float]: 統一フォーマット評価指標
            - mae: 平均絶対誤差
            - rmse: 二乗平均平方根誤差
            - spearman_corr: Spearman相関係数
            - spearman_pval: Spearman相関のp値
            - (二値分類の場合) accuracy, f1, precision, recall, roc_auc
            - (回帰の場合) top3_hit_rate, spearman_top3
            - (利益データがある場合) avg_predicted_profit, avg_correct_profit, profit_loss_rate
    """

    # CONFIG のデフォルト値
    if config is None:
        config = {'MIN_RANK': 1, 'MAX_RANK': 11}

    metrics = {}

    # =====================================
    # 1. 共通指標（全タスク）
    # =====================================

    # MAE（平均絶対誤差）
    metrics['mae'] = mean_absolute_error(y_true, y_pred)

    # RMSE（二乗平均平方根誤差）
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

    # Spearman相関係数（ランク相関）
    spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
    metrics['spearman_corr'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    metrics['spearman_pval'] = spearman_pval if not np.isnan(spearman_pval) else 1.0

    # =====================================
    # 2. タスク別指標
    # =====================================

    if task_type == 'binary':
        # ===== 二値分類指標 =====

        # 予測が確率の場合と硬いラベルの場合に対応
        if y_pred.min() >= 0 and y_pred.max() <= 1 and len(np.unique(y_pred)) > 2:
            # 確率値と判断
            y_pred_binary = (y_pred >= 0.5).astype(int)
            y_pred_proba = y_pred
        else:
            # ハードラベル
            y_pred_binary = y_pred.astype(int)
            y_pred_proba = None

        # 精度（Accuracy）
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)

        # F1スコア
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)

        # 適合率（Precision）
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)

        # 再現率（Recall）
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)

        # ROC-AUC（確率値がある場合）
        try:
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_binary)
        except Exception:
            metrics['roc_auc'] = np.nan

    elif task_type == 'regression':
        # ===== 回帰指標 =====

        # クリップ（ランク学習は1-11の範囲）
        y_pred_clipped = np.clip(y_pred, config['MIN_RANK'], config['MAX_RANK'])

        # TOP3命中率（予測ランク<=3）
        top3_pred = (y_pred_clipped <= 3).astype(int)
        top3_true = (y_true <= 3).astype(int)
        metrics['top3_hit_rate'] = accuracy_score(top3_true, top3_pred)

        # TOP3のSpearman相関
        top3_mask = (y_true <= 3) | (y_pred_clipped <= 3)
        if top3_mask.sum() >= 3:
            spearman_top3, _ = spearmanr(y_true[top3_mask], y_pred_clipped[top3_mask])
            metrics['spearman_top3'] = spearman_top3 if not np.isnan(spearman_top3) else 0.0
        else:
            metrics['spearman_top3'] = 0.0

    # =====================================
    # 3. 利益指標（共通）
    # =====================================

    if 'current_diff' in test_data.columns:
        try:
            # 予測正解時の差枚
            if task_type == 'binary':
                y_pred_binary = (y_pred >= 0.5).astype(int) if y_pred.min() >= 0 and y_pred.max() <= 1 else y_pred.astype(int)
                correct_mask = (y_pred_binary == y_true)
            else:
                # 回帰の場合、予測ランクと真実ランクが3以内なら正解
                y_pred_clipped = np.clip(y_pred, config['MIN_RANK'], config['MAX_RANK'])
                correct_mask = np.abs(y_pred_clipped - y_true) <= 3

            # 平均利益
            if correct_mask.sum() > 0:
                metrics['avg_predicted_profit'] = test_data.loc[correct_mask, 'current_diff'].mean()
                metrics['avg_correct_profit'] = test_data.loc[correct_mask, 'current_diff'].mean()
            else:
                metrics['avg_predicted_profit'] = 0.0
                metrics['avg_correct_profit'] = 0.0

            # 利益効率（実現利益 / 期待利益）
            total_profit = test_data['current_diff'].sum()
            correct_profit = test_data.loc[correct_mask, 'current_diff'].sum()

            if total_profit > 0:
                metrics['profit_loss_rate'] = correct_profit / total_profit
            else:
                metrics['profit_loss_rate'] = 0.0
        except Exception as e:
            log_warning(f"利益指標の計算でエラーが発生しました: {e}")
            metrics['avg_predicted_profit'] = np.nan
            metrics['avg_correct_profit'] = np.nan
            metrics['profit_loss_rate'] = np.nan

    return metrics


def log_unified_metrics(
    metrics: Dict[str, float],
    event_name: str = '',
    task_name: str = ''
) -> None:
    """
    統一フォーマット評価指標を見やすくログ出力

    Args:
        metrics (Dict[str, float]): 評価指標辞書
        event_name (str): イベント名
        task_name (str): タスク名
    """

    log_info("=" * 70)
    log_info(f"評価結果: {event_name} - {task_name}")
    log_info("=" * 70)

    # 共通指標
    log_info("\n【共通指標】")
    log_info(f"  MAE (平均絶対誤差):      {metrics.get('mae', np.nan):.4f}")
    log_info(f"  RMSE (二乗平均平方根):  {metrics.get('rmse', np.nan):.4f}")
    log_info(f"  Spearman相関:          {metrics.get('spearman_corr', np.nan):.4f}")

    # タスク別指標
    if 'accuracy' in metrics:
        log_info("\n【二値分類指標】")
        log_info(f"  Accuracy (精度):       {metrics.get('accuracy', np.nan):.4f}")
        log_info(f"  F1スコア:              {metrics.get('f1', np.nan):.4f}")
        log_info(f"  Precision (適合率):    {metrics.get('precision', np.nan):.4f}")
        log_info(f"  Recall (再現率):       {metrics.get('recall', np.nan):.4f}")
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            log_info(f"  ROC-AUC:              {metrics.get('roc_auc', np.nan):.4f}")

    if 'top3_hit_rate' in metrics:
        log_info("\n【回帰指標（ランク学習）】")
        log_info(f"  TOP3命中率:            {metrics.get('top3_hit_rate', np.nan):.4f}")
        log_info(f"  TOP3 Spearman相関:     {metrics.get('spearman_top3', np.nan):.4f}")

    # 利益指標
    if 'avg_predicted_profit' in metrics:
        log_info("\n【利益指標】")
        log_info(f"  平均利益:              {metrics.get('avg_predicted_profit', np.nan):.1f} 枚")
        log_info(f"  利益効率:              {metrics.get('profit_loss_rate', np.nan):.4f}")


def get_best_metric(
    metrics: Dict[str, float],
    task_type: str = 'binary'
) -> float:
    """
    タスク別の総合スコアを計算（モデル比較用）

    Args:
        metrics (Dict[str, float]): 評価指標辞書
        task_type (str): 'binary' or 'regression'

    Returns:
        float: 0-1の総合スコア
    """

    score_components = []

    if task_type == 'binary':
        # F1スコアを主要指標とする（0.5の重み）
        f1 = metrics.get('f1', 0.0)
        score_components.append(f1 * 0.5)

        # Precision重視（精度の誤りを重視：0.3の重み）
        precision = metrics.get('precision', 0.0)
        score_components.append(precision * 0.3)

        # Recall（0.2の重み）
        recall = metrics.get('recall', 0.0)
        score_components.append(recall * 0.2)

    elif task_type == 'regression':
        # TOP3命中率を主要指標（0.5の重み）
        top3_hit = metrics.get('top3_hit_rate', 0.0)
        score_components.append(top3_hit * 0.5)

        # Spearman相関を正規化（-1-1を0-1に）
        spearman = metrics.get('spearman_corr', 0.0)
        spearman_normalized = (spearman + 1.0) / 2.0
        score_components.append(spearman_normalized * 0.3)

        # TOP3特化の相関（0.2の重み）
        spearman_top3 = metrics.get('spearman_top3', 0.0)
        spearman_top3_normalized = (spearman_top3 + 1.0) / 2.0
        score_components.append(spearman_top3_normalized * 0.2)

    # 総合スコア計算
    total_score = sum(score_components)
    return np.clip(total_score, 0.0, 1.0)
