"""
設定ファイル - 全ての定数・設定を一元管理

このモジュールは、パチスロ分析システムの全ての設定値、
イベント定義、ディレクトリパスなどを集約しています。
"""

# ============================================================
# 基本設定
# ============================================================

CONFIG = {
    # ===== データベース =====
    'DB_PATH': 'pachinko_analysis_マルハンメガシティ柏.db',

    # ===== データ分割 =====
    'N_TEST_DAYS': 3,                      # テストデータ日数
    'N_VALID_DAYS': 2,                     # 検証データ日数
    'MIN_EVENT_DAYS': 8,                   # 最低必要イベント日数
    'TRAIN_RATIO': 0.7,                    # 訓練データ比率
    'TEST_SIZE': 0.1,                      # テスト比率

    # ===== イベント設定 =====
    'TEST_EVENTS': ['1day', '4day', '0day', '40day'],

    # ===== モデル最適化 =====
    'N_TRIALS': 20,                        # Optuna試行回数
    'CV_FOLDS': 5,                         # Cross-validation分割数
    'RANDOM_STATE': 42,

    # ===== 特徴量選択 =====
    'MIN_FEATURES': 10,                    # 最小特徴量数
    'MAX_FEATURES': 150,                   # 最大特徴量数
    'LASSO_THRESHOLD': 0.0001,             # Lasso係数閾値
    'CORRELATION_THRESHOLD': 0.85,         # 相関除去閾値
    'F_TEST_PVALUE': 0.05,                 # F検定p値
    'MI_THRESHOLD': 0.01,                  # 相互情報量閾値

    # ===== モデル選択 =====
    'MODELS': ['LogisticRegression', 'RandomForest', 'Ridge', 'XGBoost', 'LightGBM'],
    'DEFAULT_MODEL': 'RandomForest',

    # ===== ランク学習特有 =====
    'TOP3_ENABLED': True,                  # TOP3特化モード有効
    'TOP3_WEIGHT': 3.0,                    # TOP3への重み
    'MIN_RANK': 1,
    'MAX_RANK': 11,

    # ===== 予測 =====
    'PREDICTION_CONFIDENCE_THRESHOLD': 0.6,
    'ENSEMBLE_METHOD': 'auto_best',         # 'auto_best', 'ensemble', 'manual'

    # ===== 出力 =====
    'SAVE_MODELS': True,
    'SAVE_RESULTS': True,
    'VERBOSE': True,
    'CONFIDENCE_HIGH': 0.7,                 # 高信頼度閾値
    'CONFIDENCE_MEDIUM': 0.5,               # 中信頼度閾値
}


# ============================================================
# イベント定義
# ============================================================

EVENT_DEFINITIONS = {
    'is_1day': '1day',
    'is_2day': '2day',
    'is_3day': '3day',
    'is_4day': '4day',
    'is_5day': '5day',
    'is_6day': '6day',
    'is_7day': '7day',
    'is_8day': '8day',
    'is_9day': '9day',
    'is_0day': '0day',
    'is_39day': '39day',
    'is_40day': '40day',
    'is_zorome': 'zorome',
    'is_saturday': 'saturday',
    'is_sunday': 'sunday',
}


# ============================================================
# 末尾桁の順序定義
# ============================================================

DIGIT_ORDER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ゾロ目']


# ============================================================
# メトリクス列定義
# ============================================================

# prev系特徴量で使用する基本メトリクス
METRIC_COLS = [
    'avg_diff_coins',
    'last_digit_rank_diff',
    'total_games',
    'big_count',
    'reg_count',
    'total_bonus',
]

# ラグ特徴量で使用するターゲット列
LAG_TARGET_COLS = [
    'avg_diff_coins',
    'total_games',
    'big_count',
    'reg_count',
    'total_bonus',
    'last_digit_rank_diff',
]

# ラグ日数の設定
LAG_DAYS = [1, 2, 3, 4, 7, 14, 21, 28]

# 移動平均のウィンドウサイズ
MOVING_AVG_WINDOWS = [3, 5, 7]


# ============================================================
# ライブラリ利用可能フラグ（実行時に更新）
# ============================================================

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False


def check_library_availability():
    """
    オプションライブラリの利用可能性をチェック

    Returns:
        dict: ライブラリの利用可能状況
    """
    global XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

    try:
        import xgboost
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False

    try:
        import lightgbm
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False

    return {
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE
    }


def get_config(key=None):
    """
    設定値を取得

    Args:
        key (str, optional): 取得したい設定のキー。Noneの場合は全設定を返す

    Returns:
        設定値（keyがNoneの場合は辞書全体）
    """
    if key is None:
        return CONFIG.copy()
    return CONFIG.get(key)


def update_config(key, value):
    """
    設定値を更新

    Args:
        key (str): 更新する設定のキー
        value: 新しい値
    """
    CONFIG[key] = value
