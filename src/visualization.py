"""
可視化モジュール - 分析結果の視覚化機能

このモジュールは、モデル性能、利益分析、特徴量重要度などの
分析結果を視覚化するためのプロット関数を提供します。

主要機能:
    - モデル性能比較プロット
    - 累積利益分析プロット
    - 混同行列（Confusion Matrix）
    - イベント別比較ダッシュボード
    - 利益ヒートマップ
    - 特徴量重要度プロット
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from .logger import log_info, log_warning, log_debug

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_model_performance_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'f1',
    figsize: Tuple[int, int] = (14, 6),
    event_col: str = 'イベント',
    task_col: str = 'タスク'
) -> plt.Figure:
    """
    モデル性能を比較する棒グラフ

    Args:
        comparison_df (pd.DataFrame): 比較表データフレーム
        metric (str): 表示メトリクス ('f1', 'accuracy', 'mae' など)
        figsize (Tuple[int, int]): 図サイズ
        event_col (str): イベント列名
        task_col (str): タスク列名

    Returns:
        plt.Figure: 生成された図
    """

    if metric not in comparison_df.columns and metric != 'accuracy':
        log_warning(f"メトリクス '{metric}' が見つかりません")
        return None

    events = comparison_df[event_col].unique()
    fig, axes = plt.subplots(1, len(events), figsize=figsize)

    if len(events) == 1:
        axes = [axes]

    for idx, (event_name, ax) in enumerate(zip(events, axes)):
        event_data = comparison_df[comparison_df[event_col] == event_name]

        if not event_data.empty:
            # メトリクスがない場合は別のメトリクスを使用
            if metric not in event_data.columns:
                metric_col = 'accuracy' if 'accuracy' in event_data.columns else event_data.columns[-1]
            else:
                metric_col = metric

            event_data_sorted = event_data.sort_values(metric_col, ascending=False)

            colors = plt.cm.Set3(np.linspace(0, 1, len(event_data_sorted)))
            ax.bar(
                range(len(event_data_sorted)),
                event_data_sorted[metric_col],
                color=colors
            )
            ax.set_xlabel('Task')
            ax.set_ylabel(metric_col)
            ax.set_title(f'{event_name}')
            ax.set_xticks(range(len(event_data_sorted)))
            ax.set_xticklabels(event_data_sorted[task_col], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    log_debug(f"モデル性能比較プロット作成完了: {metric}")
    return fig


def plot_cumulative_profit(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    current_diff: Optional[np.ndarray] = None,
    event_name: str = 'Event',
    base_bet: float = 100.0,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    累積利益をプロット

    Args:
        y_test (np.ndarray): 真値
        y_pred (np.ndarray): 予測値
        current_diff (Optional[np.ndarray]): 差枚データ（利益計算用）
        event_name (str): イベント名
        base_bet (float): 基本賭け金
        figsize (Tuple[int, int]): 図サイズ

    Returns:
        plt.Figure: 生成された図
    """

    # 累積利益計算（簡易版）
    if current_diff is not None:
        # 予測が正解の場合、差枚を利益として計上
        correct_mask = (y_pred == y_test)
        profit = np.where(correct_mask, current_diff, -base_bet)
    else:
        # 差枚データがない場合、正解/不正解で固定利益
        correct_mask = (y_pred == y_test)
        profit = np.where(correct_mask, base_bet, -base_bet)

    cumulative = np.cumsum(profit)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(cumulative, linewidth=2, color='#2E86AB', label='Cumulative Profit')
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # 最大利益、最小損失を表示
    max_profit_idx = np.argmax(cumulative)
    min_loss_idx = np.argmin(cumulative)

    ax.scatter([max_profit_idx], [cumulative[max_profit_idx]], color='green', s=100, zorder=5)
    ax.scatter([min_loss_idx], [cumulative[min_loss_idx]], color='red', s=100, zorder=5)

    ax.set_xlabel('Prediction Count')
    ax.set_ylabel('Cumulative Profit (Yen)')
    ax.set_title(f'Cumulative Profit Analysis: {event_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    log_debug(f"累積利益プロット作成完了: {event_name}")
    return fig


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = 'Task',
    figsize: Tuple[int, int] = (6, 5)
) -> plt.Figure:
    """
    混同行列をプロット（二値分類用）

    Args:
        y_test (np.ndarray): 真値
        y_pred (np.ndarray): 予測値
        task_name (str): タスク名
        figsize (Tuple[int, int]): 図サイズ

    Returns:
        plt.Figure: 生成された図
    """

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix: {task_name}')

    plt.tight_layout()
    log_debug(f"混同行列プロット作成完了: {task_name}")
    return fig


def plot_event_comparison_dashboard(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    event_col: str = 'イベント',
    task_col: str = 'タスク'
) -> plt.Figure:
    """
    イベント別・タスク別の総合比較ダッシュボード

    Args:
        comparison_df (pd.DataFrame): 比較表データフレーム
        figsize (Tuple[int, int]): 図サイズ
        event_col (str): イベント列名
        task_col (str): タスク列名

    Returns:
        plt.Figure: 生成された図
    """

    events = comparison_df[event_col].unique()
    n_events = len(events)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_events, hspace=0.3, wspace=0.3)

    for idx, event_name in enumerate(events):
        event_data = comparison_df[comparison_df[event_col] == event_name]

        # Row 1: F1 Score
        ax1 = fig.add_subplot(gs[0, idx])
        if 'f1' in event_data.columns:
            ax1.bar(range(len(event_data)), event_data['f1'], color='#2E86AB')
            ax1.set_title(f'{event_name}: F1 Score')
            ax1.set_ylabel('Score')

        # Row 2: Accuracy/MAE
        ax2 = fig.add_subplot(gs[1, idx])
        metric_col = 'accuracy' if 'accuracy' in event_data.columns else 'mae'
        if metric_col in event_data.columns:
            ax2.bar(range(len(event_data)), event_data[metric_col], color='#A23B72')
            ax2.set_title(f'{event_name}: {metric_col}')
            ax2.set_ylabel(metric_col)

        # Row 3: Training Time
        ax3 = fig.add_subplot(gs[2, idx])
        if '訓練時間' in event_data.columns:
            ax3.bar(range(len(event_data)), event_data['訓練時間'], color='#F18F01')
            ax3.set_title(f'{event_name}: Training Time')
            ax3.set_ylabel('Time (sec)')

        # X軸ラベル設定
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks(range(len(event_data)))
            ax.set_xticklabels(event_data[task_col], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Event Comparison Dashboard', fontsize=16, y=0.995)
    log_debug("イベント比較ダッシュボード作成完了")
    return fig


def plot_profit_heatmap(
    profit_results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    利益結果をヒートマップで表示

    Args:
        profit_results (Dict[str, Dict[str, float]]): {イベント: {タスク: 利益}}
        figsize (Tuple[int, int]): 図サイズ

    Returns:
        plt.Figure: 生成された図
    """

    # DataFrameに変換
    df_pivot = pd.DataFrame([
        {'イベント': event, 'タスク': task, '利益': profit}
        for event, tasks in profit_results.items()
        for task, profit in tasks.items()
    ]).pivot(index='イベント', columns='タスク', values='利益')

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_pivot,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Profit (Yen)'}
    )

    ax.set_title('Profit Heatmap by Event and Task')

    plt.tight_layout()
    log_debug("利益ヒートマップ作成完了")
    return fig


def plot_feature_importance(
    feature_importances: Dict[str, float],
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    特徴量重要度をプロット

    Args:
        feature_importances (Dict[str, float]): {特徴量名: 重要度}
        top_n (int): 表示する上位特徴量数
        figsize (Tuple[int, int]): 図サイズ

    Returns:
        plt.Figure: 生成された図
    """

    # ソート
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.invert_yaxis()

    plt.tight_layout()
    log_debug(f"特徴量重要度プロット作成完了: TOP {top_n}")
    return fig


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = './output',
    dpi: int = 100
) -> None:
    """
    図をファイルに保存

    Args:
        fig (plt.Figure): 保存する図
        filename (str): ファイル名
        output_dir (str): 出力ディレクトリ
        dpi (int): 解像度
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    log_info(f"図を保存しました: {filepath}")


def create_comparison_table(
    results_dict: Dict[str, Dict[str, Any]],
    event_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    結果辞書から比較表を作成

    Args:
        results_dict (Dict[str, Dict[str, Any]]): {イベント: {タスク: 結果}}
        event_names (Optional[List[str]]): イベント名リスト（順序指定用）

    Returns:
        pd.DataFrame: 比較表
    """

    if event_names is None:
        event_names = list(results_dict.keys())

    rows = []
    for event in event_names:
        if event not in results_dict:
            continue

        for task, result in results_dict[event].items():
            row = {
                'イベント': event,
                'タスク': task
            }

            # メトリクスを追加
            if 'metrics' in result:
                row.update(result['metrics'])

            # その他の情報を追加
            if 'model_name' in result:
                row['モデル'] = result['model_name']
            if 'n_features' in result:
                row['特徴量数'] = result['n_features']
            if 'training_time' in result:
                row['訓練時間'] = result['training_time']

            rows.append(row)

    return pd.DataFrame(rows)
