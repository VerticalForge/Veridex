# predictor.py - Hot Lead Scoring Module
# ============================================================
# VERSION : v1.0
# PURPOSE : Loads the trained GradientBoostingClassifier and its
#           fitted StandardScaler. Scores every property in the
#           featured dataset with a hot_lead probability (0.0-1.0),
#           ranks all properties hottest-first, and writes the
#           result to the scored_leads table for investor export.
#           Called by main.py via run_predictor(state) in PREDICT mode.
#
# SCORING STEPS (in order):
#   Step 1 → Load model + scaler from .pkl files (joblib)
#   Step 2 → Load featured data (state['featured_df'] or leads_features)
#   Step 3 → Build feature matrix — select 9 FEATURE_COLUMNS in order,
#            null-fill, scale with the LOADED scaler (transform only)
#   Step 4 → predict_proba → hot_lead_score (probability of class 1)
#   Step 5 → Rank descending — rank 1 = hottest lead
#   Step 6 → Save to scored_leads table + write state['scored_df']
#
# CRITICAL CONTRACT:
#   · The scaler is LOADED, never re-fit. We call transform(),
#     NOT fit_transform(). Re-fitting would rescale this run's data
#     against itself and silently corrupt every score.
#   · FEATURE_COLUMNS order must match training order exactly.
#     project_config.FEATURE_COLUMNS is the single source of truth.
#   · NaN is already filled in features.py before leads_features is
#     written. The defensive fillna(0) here mirrors the exact
#     transformation the scaler was fitted on at training time.
#   · database.save_scored_leads() consumes [OBJECTID,
#     hot_lead_score, rank] and adds run_id + pipeline_mode itself.
#
# STATE CONTRACT:
#   Reads  → state['featured_df']  (DataFrame from features node)
#   Writes → state['scored_df']    (ranked, scored DataFrame)
#   Errors → state['errors']       (appended on failure)
# ============================================================

import os
import sqlite3
import logging
import joblib
import pandas as pd

import project_config as config
import database

logger = logging.getLogger(__name__)

# Display columns carried into state['scored_df'] for investor /
# Power BI use when present. These are NOT required by the DB write
# (save_scored_leads only uses OBJECTID, hot_lead_score, rank) — they
# are included only if they exist in the featured DataFrame.
DISPLAY_COLUMNS = [
    "OBJECTID", "OWN_NAME", "PHY_ADDR1", "PHY_CITY",
    "PHY_ZIPCD", "JV", "SALE_PRC1", "SALE_YR1",
]


# ============================================================
# INDIVIDUAL SCORING STEPS
# ============================================================

def _step1_load_artifacts():
    """
    Step 1 — Load the trained model and fitted scaler from disk.

    Both files were written by ml_model.py via joblib.dump after
    the model passed the F1 threshold. joblib.load restores them
    to the exact state they had when training finished — same
    learned weights, same learned scaler means/stds.

    Raises FileNotFoundError if either artifact is missing, so the
    pipeline fails loudly instead of scoring with a partial setup.
    """
    model_path  = config.MODEL_PATH
    scaler_path = config.SCALER_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Run PIPELINE_MODE='TRAIN' first to build it."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}. "
            f"Run PIPELINE_MODE='TRAIN' first to build it."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    logger.info(
        f"Step 1 | Load artifacts     | "
        f"model={os.path.basename(model_path)} | "
        f"scaler={os.path.basename(scaler_path)}"
    )
    return model, scaler


def _step2_load_featured(state: dict) -> pd.DataFrame:
    """
    Step 2 — Get the featured DataFrame to score.

    Prefers state['featured_df'] written by the features node in
    the same run. Falls back to reading leads_features directly
    from the database if state is empty (e.g. predictor run in
    isolation). Both sources already have FEATURE_COLUMNS null-filled
    by features.py, so no source produces NaN into the scaler.
    """
    df = state.get("featured_df")

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        logger.info(
            "featured_df is empty. Loading from leads_features table directly."
        )
        conn = sqlite3.connect(state["db_path"])
        df   = pd.read_sql_query("SELECT * FROM leads_features", conn)
        conn.close()
        logger.info(f"Loaded {len(df):,} records from leads_features.")

    if df is None or len(df) == 0:
        raise ValueError(
            "No data to score. featured_df and leads_features are both empty."
        )

    logger.info(f"Step 2 | Load featured data | {len(df):,} records to score")
    return df


def _step3_build_scaled_matrix(df: pd.DataFrame, scaler):
    """
    Step 3 — Build the scaled feature matrix for prediction.

    · Select exactly the FEATURE_COLUMNS, in config order. The
      column order MUST match what the scaler/model were trained on.
    · Defensive fillna(0): leads_features is already null-filled by
      features.py, but this guarantees the matrix the scaler sees is
      identical in shape to training, even if scored in isolation.
    · scaler.transform() — NOT fit_transform(). The scaler applies
      the means/stds it learned at training time. We never re-fit.

    Raises ValueError if any FEATURE_COLUMN is missing from the data.
    """
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing FEATURE_COLUMNS for scoring: {missing}. "
            f"Check that features.py produced all required columns."
        )

    X = df[config.FEATURE_COLUMNS].fillna(0)

    null_remaining = int(X.isnull().sum().sum())
    if null_remaining > 0:
        # Should never happen after fillna — guard against silent NaN.
        raise ValueError(
            f"{null_remaining} NaN values remain in feature matrix after fill."
        )

    X_scaled = scaler.transform(X.values)

    logger.info(
        f"Step 3 | Scale features     | "
        f"matrix={X_scaled.shape[0]:,} x {X_scaled.shape[1]} | "
        f"transform (scaler not re-fit)"
    )
    return X_scaled


def _step4_score(model, X_scaled, df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4 — Generate hot_lead_score for every property.

    predict_proba returns two columns: P(class 0) and P(class 1).
    We take column index 1 — the probability that the property is a
    hot lead. This continuous score (0.0-1.0) is what makes the
    output rankable, unlike the hard 0/1 label from predict().
    """
    proba = model.predict_proba(X_scaled)[:, 1]

    scored = df.copy()
    scored["hot_lead_score"] = proba

    logger.info(
        f"Step 4 | Score (predict_proba) | "
        f"mean={proba.mean():.3f} | "
        f">0.5={int((proba > 0.5).sum()):,} | "
        f">0.8={int((proba > 0.8).sum()):,}"
    )
    return scored


def _step5_rank(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5 — Rank all properties hottest-first.

    Sort by hot_lead_score descending, then assign sequential
    integer ranks starting at 1. Rank 1 is the single most likely
    motivated seller. Investors work the list from the top down.
    """
    scored = (
        scored
        .sort_values("hot_lead_score", ascending=False)
        .reset_index(drop=True)
    )
    scored["rank"] = range(1, len(scored) + 1)

    logger.info(
        f"Step 5 | Rank leads          | "
        f"top score={scored['hot_lead_score'].iloc[0]:.3f} | "
        f"ranked {len(scored):,} properties"
    )
    return scored


def _step6_save(scored: pd.DataFrame, run_id: str):
    """
    Step 6 — Persist the ranked scores.

    Builds the investor-facing scored_df from DISPLAY_COLUMNS that
    actually exist, plus hot_lead_score and rank. Casts OBJECTID to
    int for the scored_leads schema. database.save_scored_leads()
    selects only [OBJECTID, hot_lead_score, rank] and appends the
    run_id + pipeline_mode itself, so extra display columns are safe.
    """
    present_display = [c for c in DISPLAY_COLUMNS if c in scored.columns]
    out_cols        = present_display + ["hot_lead_score", "rank"]

    scored_df = scored[out_cols].copy()
    scored_df["OBJECTID"] = scored_df["OBJECTID"].astype(int)
    scored_df["rank"]     = scored_df["rank"].astype(int)

    database.save_scored_leads(scored_df, run_id)

    logger.info(
        f"Step 6 | Save scored leads   | "
        f"scored_leads ← {len(scored_df):,} rows | run_id={run_id}"
    )
    return scored_df


# ============================================================
# MAIN NODE FUNCTION — called by main.py
# ============================================================

def run_predictor(state: dict) -> dict:
    """
    Main scoring node. Called by main.py orchestrator in PREDICT mode.

    Loads the trained model + scaler, scores every featured property
    with a hot_lead probability, ranks them hottest-first, saves the
    ranked list to scored_leads, and writes the scored DataFrame to
    state['scored_df'].

    Args:
        state : Shared pipeline State dict from build_state()

    Returns:
        Updated state with state['scored_df'] populated.
        On failure, appends to state['errors'] and returns
        state with scored_df = None.
    """
    logger.info("-" * 55)
    logger.info("NODE: run_predictor")
    logger.info("-" * 55)

    try:
        # ── Load model + scaler ───────────────────────────────
        model, scaler = _step1_load_artifacts()

        # ── Load featured data ────────────────────────────────
        df = _step2_load_featured(state)

        # ── Scale, score, rank ────────────────────────────────
        X_scaled = _step3_build_scaled_matrix(df, scaler)
        scored   = _step4_score(model, X_scaled, df)
        scored   = _step5_rank(scored)

        # ── Persist ───────────────────────────────────────────
        run_id    = state.get("run_id")
        scored_df = _step6_save(scored, run_id)

        # ── Write to state ────────────────────────────────────
        state["scored_df"] = scored_df

        logger.info("-" * 55)
        logger.info(
            f"run_predictor complete | "
            f"{len(scored_df):,} leads ranked → state['scored_df']"
        )
        logger.info(
            f"Top lead: OBJECTID={int(scored_df['OBJECTID'].iloc[0])} | "
            f"score={scored_df['hot_lead_score'].iloc[0]:.3f}"
        )
        logger.info("-" * 55)

    except Exception as e:
        logger.error(f"run_predictor failed: {e}")
        state["errors"].append({
            "stage" : "predictor",
            "error" : str(e)
        })
        state["scored_df"] = None

    return state
