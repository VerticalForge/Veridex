# cleaner.py - Data Cleaning Module
# ============================================================
# VERSION : v1.1
# PURPOSE : Cleans raw leads from leads table.
#           Produces clean records saved to leads_clean table.
#           Called by main.py orchestrator via run_cleaning(state).
#
# CLEANING STEPS (in order):
#   Step 1 → Residential filter     — keep DOR_UC 001, 002, 007, 010
#   Step 2 → JV minimum filter      — drop JV below $1,000
#   Step 3 → JV outlier removal     — IQR upper bound only
#   Step 4 → PHY_CITY standardize   — strip, uppercase, fix spelling,
#                                      fill blanks from ZIP, fallback UNKNOWN
#   Step 5 → ACT_YR_BLT imputation  — fill zeros from DOR group median
#   Step 6 → EFF_YR_BLT imputation  — fill zeros from ACT_YR_BLT
#   Step 7 → TOT_LVG_AR zeros→NaN   — zero living area is not data,
#                                      feature engineering handles NaN
#   Step 8 → Save to leads_clean    — write clean DataFrame to database
#
# WHAT IS NOT IN THIS FILE:
#   · never_sold flag  → features.py (it is a feature, not a cleaning step)
#   · Validation guards → not included per project decision
#   · Lower IQR bound  → not included per project decision
#   · if_exists logic  → handled by PIPELINE_MODE in main.py
#
# STATE CONTRACT:
#   Reads  → state['raw_df']    (DataFrame from scraper, or loaded from DB)
#   Writes → state['clean_df']  (cleaned DataFrame)
#   Errors → state['errors']    (appended on failure)
# ============================================================

import sqlite3
import logging
import numpy as np
import pandas as pd
import project_config as config

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS — stay inside cleaner.py per project decision
# ============================================================

RESIDENTIAL_CODES = ['001', '002', '007', '010']

ZIP_TO_CITY = {
    32601: 'GAINESVILLE', 32602: 'GAINESVILLE',
    32603: 'GAINESVILLE', 32604: 'GAINESVILLE',
    32605: 'GAINESVILLE', 32606: 'GAINESVILLE',
    32607: 'GAINESVILLE', 32608: 'GAINESVILLE',
    32609: 'GAINESVILLE', 32610: 'GAINESVILLE',
    32611: 'GAINESVILLE', 32612: 'GAINESVILLE',
    32641: 'GAINESVILLE', 32653: 'GAINESVILLE',
    32615: 'ALACHUA',     32616: 'ALACHUA',
    32694: 'WALDO',       32643: 'HIGH SPRINGS',
    32640: 'HAWTHORNE',   32666: 'KEYSTONE HEIGHTS',
    32668: 'MICANOPY',
}

# NOTE: UNKNOWN city is acceptable up to ~5% of records.
# Any ZIP not in ZIP_TO_CITY maps to UNKNOWN.
# Commercial expansion will extend this mapping if needed.

CITY_CORRECTIONS = {
    'LACROSSE' : 'LA CROSSE',
    'LA CROSSE': 'LA CROSSE',
}


# ============================================================
# INDIVIDUAL CLEANING STEPS
# Each step takes a DataFrame, returns a cleaned DataFrame.
# No step touches the database. Only run_cleaning() does that.
# ============================================================

def _step1_residential_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Keep only residential properties.

    Normalizes DOR_UC first:
        · Convert to string
        · Strip whitespace
        · Zero-pad to 3 digits (turns '1' into '001')

    Then keeps only records matching RESIDENTIAL_CODES.
    All other property types (commercial, agricultural,
    government, vacant land) are removed.
    """
    before = len(df)

    df['DOR_UC'] = (
        df['DOR_UC']
        .astype(str)
        .str.strip()
        .str.zfill(3)
    )

    df = df[df['DOR_UC'].isin(RESIDENTIAL_CODES)].copy()

    after = len(df)
    logger.info(
        f"Step 1 | Residential filter | "
        f"{before:,} → {after:,} | "
        f"removed {before - after:,} non-residential"
    )
    return df


def _step2_jv_minimum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 — Remove records where JV (Just Value) is below $1,000.

    Properties with JV below $1,000 are data errors or
    parcels with no meaningful assessed value.
    They cannot produce reliable ML features.
    """
    before = len(df)

    df = df[df['JV'] >= 1000].copy()

    after = len(df)
    logger.info(
        f"Step 2 | JV minimum $1,000  | "
        f"{before:,} → {after:,} | "
        f"removed {before - after:,} low-value records"
    )
    return df


def _step3_jv_outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 — Remove extreme JV outliers using IQR method.

    Upper bound only: Q3 + (1.5 * IQR)
    Lower bound not applied — Step 2 already handles the floor.

    Properties above the upper bound are statistical extremes
    that would distort the ML model's learning.
    """
    before = len(df)

    Q1          = df['JV'].quantile(0.25)
    Q3          = df['JV'].quantile(0.75)
    IQR         = Q3 - Q1
    upper_bound = Q3 + (1.5 * IQR)

    df = df[df['JV'] <= upper_bound].copy()

    after = len(df)
    logger.info(
        f"Step 3 | JV outlier removal | "
        f"{before:,} → {after:,} | "
        f"removed {before - after:,} | "
        f"upper bound ${upper_bound:,.0f}"
    )
    return df


def _step4_city_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4 — Standardize PHY_CITY values.

    4a → Strip whitespace and uppercase everything
    4b → Replace 'NAN' string (appears after astype conversion)
    4c → Apply known spelling corrections (LACROSSE → LA CROSSE)
    4d → Convert PHY_ZIPCD to int for dictionary lookup
    4e → Fill empty cities from ZIP_TO_CITY mapping
    4f → Fallback to UNKNOWN for any remaining blanks

    UNKNOWN is acceptable up to ~5% of records.
    ZIP_TO_CITY covers all known Alachua County ZIPs.
    """
    # 4a — Strip and uppercase
    df['PHY_CITY'] = (
        df['PHY_CITY']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 4b — Replace NAN string
    df['PHY_CITY'] = df['PHY_CITY'].replace('NAN', '')

    # 4c — Apply spelling corrections
    df['PHY_CITY'] = df['PHY_CITY'].replace(CITY_CORRECTIONS)

    # 4d — Convert ZIP to int for dict lookup
    df['PHY_ZIPCD'] = df['PHY_ZIPCD'].fillna(0).astype(int)

    # 4e — Fill empty cities from ZIP lookup
    empty_mask = df['PHY_CITY'] == ''
    df.loc[empty_mask, 'PHY_CITY'] = (
        df.loc[empty_mask, 'PHY_ZIPCD']
        .map(ZIP_TO_CITY)
        .fillna('UNKNOWN')
    )

    unknown_count = (df['PHY_CITY'] == 'UNKNOWN').sum()
    unknown_pct   = (unknown_count / len(df)) * 100

    logger.info(
        f"Step 4 | City standardize   | "
        f"UNKNOWN={unknown_count:,} ({unknown_pct:.1f}%)"
    )

    if unknown_pct > 5.0:
        logger.warning(
            f"UNKNOWN city exceeds 5% ({unknown_pct:.1f}%). "
            f"Review ZIP_TO_CITY mapping in cleaner.py."
        )

    return df


def _step5_act_yr_blt_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5 — Impute zero ACT_YR_BLT (actual year built).

    Zero means year built was not recorded.
    Strategy:
        · Calculate median year by DOR_UC group from valid records
        · Fill zeros with their DOR group median
        · Any remaining zeros (rare edge case) → overall median
    """
    before_zeros = (df['ACT_YR_BLT'] == 0).sum()

    group_medians  = (
        df[df['ACT_YR_BLT'] > 0]
        .groupby('DOR_UC')['ACT_YR_BLT']
        .median()
    )
    overall_median = int(
        df[df['ACT_YR_BLT'] > 0]['ACT_YR_BLT'].median()
    )

    zero_mask = df['ACT_YR_BLT'] == 0
    df.loc[zero_mask, 'ACT_YR_BLT'] = (
        df.loc[zero_mask, 'DOR_UC']
        .map(group_medians)
        .fillna(overall_median)
    )
    df['ACT_YR_BLT'] = df['ACT_YR_BLT'].astype(int)

    after_zeros = (df['ACT_YR_BLT'] == 0).sum()
    logger.info(
        f"Step 5 | ACT_YR_BLT impute  | "
        f"filled {before_zeros:,} zeros | "
        f"remaining {after_zeros:,}"
    )
    return df


def _step6_eff_yr_blt_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6 — Impute zero EFF_YR_BLT (effective year built).

    Zero means effective year was not recorded.
    Strategy: set equal to ACT_YR_BLT where zero.
    ACT_YR_BLT is already clean from Step 5.
    """
    before_zeros = (df['EFF_YR_BLT'] == 0).sum()

    eff_zero_mask = df['EFF_YR_BLT'] == 0
    df.loc[eff_zero_mask, 'EFF_YR_BLT'] = (
        df.loc[eff_zero_mask, 'ACT_YR_BLT']
    )
    df['EFF_YR_BLT'] = df['EFF_YR_BLT'].astype(int)

    after_zeros = (df['EFF_YR_BLT'] == 0).sum()
    logger.info(
        f"Step 6 | EFF_YR_BLT impute  | "
        f"filled {before_zeros:,} zeros | "
        f"remaining {after_zeros:,}"
    )
    return df


def _step7_living_area_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7 — Convert TOT_LVG_AR zeros to NaN.

    Zero living area means vacant land or unrecorded data.
    It is not a valid measurement — it cannot be used
    to calculate value_per_sqft in feature engineering.
    Setting to NaN signals features.py to handle it correctly.
    """
    before_zeros = (df['TOT_LVG_AR'] == 0).sum()

    df.loc[df['TOT_LVG_AR'] == 0, 'TOT_LVG_AR'] = np.nan

    after_nan = df['TOT_LVG_AR'].isnull().sum()
    logger.info(
        f"Step 7 | Living area zeros  | "
        f"converted {before_zeros:,} zeros to NaN | "
        f"total NaN={after_nan:,}"
    )
    return df


def _step8_save_to_db(df: pd.DataFrame):
    """
    Step 8 — Save clean DataFrame to leads_clean table.

    Uses if_exists='replace' — pipeline mode logic
    is handled by main.py orchestrator, not here.

    Verifies save by reading count back from database.
    """
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df.to_sql('leads_clean', conn, if_exists='replace', index=False)
        saved_count = pd.read_sql_query(
            "SELECT COUNT(*) as total FROM leads_clean", conn
        ).iloc[0]['total']
        logger.info(
            f"Step 8 | Saved to DB       | "
            f"leads_clean table → {saved_count:,} records"
        )
    except Exception as e:
        logger.error(f"Step 8 | Save failed: {e}")
        raise
    finally:
        conn.close()


# ============================================================
# MAIN NODE FUNCTION — called by main.py
# ============================================================

def run_cleaning(state: dict) -> dict:
    """
    Main cleaning node. Called by main.py orchestrator.

    Reads raw data from state['raw_df'].
    If raw_df is None or empty, falls back to loading
    from leads table directly — handles DRY_RUN mode
    where scraper skips the API.

    Runs all 8 cleaning steps in sequence.
    Writes clean DataFrame to state['clean_df'].
    Saves clean data to leads_clean table in database.

    Args:
        state : Shared pipeline State dict from build_state()

    Returns:
        Updated state with state['clean_df'] populated.
        On failure, appends to state['errors'] and returns
        state with clean_df = None.
    """
    logger.info("-" * 55)
    logger.info("NODE: run_cleaning")
    logger.info("-" * 55)

    try:
        # ── Load data ────────────────────────────────────────
        # raw_df comes from scraper in FULL_REFRESH / INCREMENTAL.
        # In DRY_RUN, scraper loads from DB into raw_df already.
        # If raw_df is still None for any reason, load from DB.

        df = state.get("raw_df")

        if df is None or (hasattr(df, '__len__') and len(df) == 0):
            logger.info(
                "raw_df is empty. Loading from leads table directly."
            )
            conn = sqlite3.connect(state["db_path"])
            df   = pd.read_sql_query("SELECT * FROM leads", conn)
            conn.close()
            logger.info(f"Loaded {len(df):,} records from leads table.")

        if df is None or len(df) == 0:
            logger.error("No data to clean. leads table is empty.")
            state["errors"].append({
                "stage" : "cleaner",
                "error" : "No data available. leads table is empty."
            })
            state["clean_df"] = None
            return state

        logger.info(f"Starting records : {len(df):,}")

        # ── Run cleaning steps in order ───────────────────────
        df = _step1_residential_filter(df)
        df = _step2_jv_minimum(df)
        df = _step3_jv_outlier_removal(df)
        df = _step4_city_standardize(df)
        df = _step5_act_yr_blt_impute(df)
        df = _step6_eff_yr_blt_impute(df)
        df = _step7_living_area_zeros(df)

        # ── Save to database ──────────────────────────────────
        _step8_save_to_db(df)

        # ── Write to state ────────────────────────────────────
        state["clean_df"] = df

        logger.info("-" * 55)
        logger.info(
            f"run_cleaning complete | "
            f"{len(df):,} clean records → state['clean_df']"
        )
        logger.info("-" * 55)

    except Exception as e:
        logger.error(f"run_cleaning failed: {e}")
        state["errors"].append({
            "stage" : "cleaner",
            "error" : str(e)
        })
        state["clean_df"] = None

    return state
