# ml_model.py - Model Training Module
# ============================================================
# VERSION : v1.0
# PURPOSE : Trains a GradientBoostingClassifier on engineered
#           features from the leads_features table.
#           Evaluates model quality using F1 score.
#           Saves trained model and scaler to disk as .pkl files.
#           Called by main.py via run_model(state).
#
# TRAINING STEPS (in order):
#   Step 1  → Load data         — from state or leads_features table
#   Step 2  → Extract X and y   — features and hot_lead label
#   Step 3  → Train/test split  — 80% train, 20% test
#   Step 4  → StandardScaler    — fit on train, transform both sets
#   Step 5  → Train model       — GradientBoostingClassifier
#   Step 6  → Cross-validation  — 5-fold CV on training data
#   Step 7  → Evaluate          — F1, precision, recall on test set
#   Step 8  → Feature importance — log which features matter most
#   Step 9  → Save or reject    — save .pkl files only if F1 >= threshold
#   Step 10 → Write to state    — f1_score written back for main.py
#
# STATE CONTRACT:
#   Reads  → state['featured_df']  (DataFrame from features.py)
#   Writes → state['f1_score']     (float — test set F1)
#   Errors → state['errors']       (appended on failure)
#
# OUTPUT FILES (saved to models/ folder):
#   hot_lead_model.pkl  — trained GradientBoostingClassifier
#   scaler.pkl          — fitted StandardScaler
#
# THESE FILES ARE USED BY:
#   predictor.py → loads both files to score new properties
# ============================================================

import os
import logging
import sqlite3
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.model_selection  import train_test_split, cross_val_score,StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import f1_score, precision_score, recall_score

import project_config as config

logger = logging.getLogger(__name__)


# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
# These control the behaviour of GradientBoostingClassifier.
# They are set here rather than in project_config because
# they are internal ML concerns, not pipeline configuration.
#
#   n_estimators  — how many trees to build sequentially
#                   More trees = more learning, but slower.
#                   200 is the industry standard starting point.
#
#   learning_rate — how much each new tree corrects the previous one
#                   Lower = more careful learning, better generalization.
#                   0.05 pairs well with 200 trees.
#
#   max_depth     — how deep each individual tree can grow
#                   Deeper = more complex patterns, but risks overfitting.
#                   4 is the standard for tabular real estate data.
#
#   random_state  — seed for reproducibility (from config)
#                   Same seed = same results every run on same data.

GBC_PARAMS = {
    "n_estimators"  : 200,
    "learning_rate" : 0.05,
    "max_depth"     : 4,
    "random_state"  : config.RANDOM_STATE,
}


# ============================================================
# INDIVIDUAL TRAINING STEPS
# ============================================================

def _step1_load_data(state: dict) -> pd.DataFrame:
    """
    Step 1 — Load featured data.

    Reads from state['featured_df'] if available.
    Falls back to loading from leads_features table directly.
    This handles the case where ml_model.py is run standalone
    after a previous pipeline run already populated the table.
    """
    df = state.get("featured_df")

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        logger.info("featured_df is empty. Loading from leads_features table.")
        conn = sqlite3.connect(state["db_path"])
        df   = pd.read_sql_query("SELECT * FROM leads_features", conn)
        conn.close()
        logger.info(f"Loaded {len(df):,} records from leads_features.")

    logger.info(f"Step 1  | Data loaded           | {len(df):,} records")
    return df


def _step2_extract_features(df: pd.DataFrame):
    """
    Step 2 — Extract X (features) and y (target label).

    X contains only the 11 columns in FEATURE_COLUMNS.
    y contains only the hot_lead column (0 or 1).

    Returns X and y as separate objects.
    All downstream steps work with X and y, not the full DataFrame.
    """
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()

    hot_count  = int(y.sum())
    cold_count = int((y == 0).sum())
    hot_pct    = y.mean() * 100

    logger.info(
        f"Step 2  | Features extracted    | "
        f"X={X.shape} | "
        f"hot={hot_count:,} ({hot_pct:.1f}%) | "
        f"cold={cold_count:,}"
    )
    return X, y


def _step3_train_test_split(X, y):
    """
    Step 3 — Split data into training and test sets.

    TEST_SIZE = 0.20 → 80% train, 20% test.
    stratify=y ensures the hot/cold ratio is the same in both sets.
    For example if 14% of all records are hot leads, the train set
    will also be ~14% hot leads and so will the test set.
    Without stratify, a random split could produce a lopsided test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config.TEST_SIZE,
        random_state = config.RANDOM_STATE,
        stratify     = y
    )

    logger.info(
        f"Step 3  | Train/test split      | "
        f"train={len(X_train):,} | "
        f"test={len(X_test):,} | "
        f"split={int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)}"
    )
    return X_train, X_test, y_train, y_test


def _step4_scale_features(X_train, X_test):
    """
    Step 4 — Fit StandardScaler on training data, transform both sets.

    IMPORTANT: fit_transform is called only on X_train.
    X_test is transformed using the same scaler — never re-fitted.

    Why: The scaler learns the mean and standard deviation of each
    feature from training data only. If we re-fit on test data,
    we introduce data leakage — the model indirectly learns from
    test data during training. This inflates performance metrics.

    Returns the fitted scaler so it can be saved to disk later.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    logger.info(
        f"Step 4  | StandardScaler        | "
        f"fitted on {len(X_train):,} training records"
    )
    return scaler, X_train, X_test


def _step5_train_model(X_train, y_train) -> GradientBoostingClassifier:
    """
    Step 5 — Train GradientBoostingClassifier.

    Builds 200 trees sequentially. Each tree focuses on correcting
    the mistakes of the previous tree. The result is a model that
    progressively improves its predictions with every tree added.

    This is the core of VERIDEX — the model that will assign
    every property a probability score between 0 and 1.
    """
    model = GradientBoostingClassifier(**GBC_PARAMS)
    model.fit(X_train, y_train)

    logger.info(
        f"Step 5  | Model trained         | "
        f"trees={model.n_estimators} | "
        f"depth={model.max_depth} | "
        f"lr={model.learning_rate}"
    )
    return model


def _step6_cross_validation(model_params: dict, X_all_scaled, y) -> float:
    """
    Step 6 — 5-fold cross-validation on the full scaled dataset.

    Trains a fresh model 5 times, each time with a different 20%
    held out as the test set. Logs all 5 F1 scores and their mean.

    The CV mean F1 is a more reliable indicator of real-world
    performance than a single train/test split because it tests
    the model across all parts of the data.

    Returns the mean CV F1 score.
    """
    cv_splitter = StratifiedKFold(
        n_splits     = config.CV_FOLDS,
        shuffle      = True,
        random_state = config.RANDOM_STATE
    )

    cv_model  = GradientBoostingClassifier(**model_params)
    cv_scores = cross_val_score(
        cv_model,
        X_all_scaled,
        y,
        cv      = cv_splitter,
        scoring = "f1"
    )

    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())

    logger.info(
        f"Step 6  | Cross-validation      | "
        f"folds={config.CV_FOLDS} | "
        f"scores={[round(s, 3) for s in cv_scores]} | "
        f"mean={cv_mean:.3f} | "
        f"std={cv_std:.3f}"
    )
    return cv_mean


def _step7_evaluate(model, X_test, y_test) -> tuple:
    """
    Step 7 — Evaluate model on the held-out test set.

    Uses the test set the model has never seen during training.
    Returns F1, precision, and recall.

    F1        — balance of precision and recall (main quality metric)
    Precision — of all properties predicted as hot leads, how many are?
    Recall    — of all real hot leads, how many did the model find?
    """
    y_pred    = model.predict(X_test)
    f1        = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall    = float(recall_score(y_test, y_pred))

    logger.info(
        f"Step 7  | Test set evaluation   | "
        f"F1={f1:.3f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f}"
    )
    return f1, precision, recall


def _step8_feature_importance(model: GradientBoostingClassifier):
    """
    Step 8 — Log feature importances ranked highest to lowest.

    Shows which of the 11 features contributed most to the model's
    decisions. All values sum to 1.0.

    This is the result you show a  investor to explain
    why the model flags certain properties as hot leads.
    """
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    logger.info("Step 8  | Feature importances   | ranked by contribution:")
    for rank, i in enumerate(sorted_idx, start=1):
        logger.info(
            f"        |  {rank:>2}. {config.FEATURE_COLUMNS[i]:<30} "
            f"{importances[i]:.4f} "
            f"({importances[i] * 100:.1f}%)"
        )


def _step9_save_model(model, scaler, f1: float) -> bool:
    """
    Step 9 — Save model and scaler to disk if F1 meets threshold.

    Saves to the models/ directory defined in project_config.
    Creates the directory if it does not exist.

    Both files must be saved together — the scaler must be the exact
    same one that was fitted during this training run. When predictor.py
    loads the model, it must scale new data identically to how the
    training data was scaled. A mismatched scaler produces garbage scores.

    Returns True if files were saved, False if F1 was below threshold.
    """
    if f1 < config.F1_THRESHOLD:
        logger.warning(
            f"Step 9  | Model REJECTED        | "
            f"F1={f1:.3f} < threshold {config.F1_THRESHOLD} | "
            f"Files NOT saved. Review features or training data."
        )
        return False

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    joblib.dump(model,  config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)

    model_size_kb  = os.path.getsize(config.MODEL_PATH)  / 1024
    scaler_size_kb = os.path.getsize(config.SCALER_PATH) / 1024

    logger.info(
        f"Step 9  | Model ACCEPTED        | "
        f"F1={f1:.3f} >= threshold {config.F1_THRESHOLD}"
    )
    logger.info(
        f"        | Saved: {config.MODEL_PATH} "
        f"({model_size_kb:.1f} KB)"
    )
    logger.info(
        f"        | Saved: {config.SCALER_PATH} "
        f"({scaler_size_kb:.1f} KB)"
    )
    return True


# ============================================================
# MAIN NODE FUNCTION — called by main.py
# ============================================================

def run_model(state: dict) -> dict:
    """
    Main model training node. Called by main.py orchestrator.

    Reads featured data from state['featured_df'].
    Runs all 10 training steps in sequence.
    Writes f1_score to state['f1_score'] for main.py to log.

    If F1 >= F1_THRESHOLD:
        → hot_lead_model.pkl and scaler.pkl are saved to models/
        → main.py logs acceptance and guides user to PREDICT mode

    If F1 < F1_THRESHOLD:
        → Files are NOT saved
        → main.py logs a warning

    Args:
        state : Shared pipeline State dict from build_state()

    Returns:
        Updated state with state['f1_score'] populated.
        On failure, appends to state['errors'] and returns
        state with f1_score = None.
    """
    logger.info("-" * 55)
    logger.info("NODE: run_model")
    logger.info("-" * 55)

    try:
        # ── Step 1: Load data ─────────────────────────────────
        df = _step1_load_data(state)

        if df is None or len(df) == 0:
            logger.error("No data for model training. leads_features is empty.")
            state["errors"].append({
                "stage" : "model",
                "error" : "No data available. leads_features table is empty."
            })
            state["f1_score"] = None
            return state

        # ── Step 2: Extract X and y ───────────────────────────
        X, y = _step2_extract_features(df)

        # ── Step 3: Train/test split ──────────────────────────
        X_train, X_test, y_train, y_test = _step3_train_test_split(X, y)

        # ── Step 4: Scale features ────────────────────────────
        scaler, X_train_scaled, X_test_scaled = _step4_scale_features(
            X_train, X_test
        )

        # ── Step 5: Train model ───────────────────────────────
        model = _step5_train_model(X_train_scaled, y_train)

        # ── Step 6: Cross-validation ──────────────────────────
        # Scale full dataset using the already-fitted scaler
        # for consistent CV evaluation
        X_all_scaled = scaler.transform(X)
        _step6_cross_validation(GBC_PARAMS, X_all_scaled, y)

        # ── Step 7: Evaluate on test set ──────────────────────
        f1, precision, recall = _step7_evaluate(
            model, X_test_scaled, y_test
        )

        # ── Step 8: Feature importances ───────────────────────
        _step8_feature_importance(model)

        # ── Step 9: Save or reject ────────────────────────────
        _step9_save_model(model, scaler, f1)

        # ── Step 10: Write to state ───────────────────────────
        state["f1_score"] = f1

        logger.info("-" * 55)
        logger.info(
            f"run_model complete | "
            f"F1={f1:.3f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f}"
        )
        logger.info("-" * 55)

    except Exception as e:
        logger.error(f"run_model failed: {e}")
        state["errors"].append({
            "stage" : "model",
            "error" : str(e)
        })
        state["f1_score"] = None

    return state
