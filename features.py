456# features.py - Feature Engineering Module
# ============================================================
# VERSION : v1.1
# PURPOSE : Engineers all ML features from clean leads data.
#           Generates hot_lead labels via weak supervision.
#           Saves output to leads_features table.
#           Called by main.py via run_features(state).
#
# FEATURE ENGINEERING STEPS (in order):
#   Step 1  → property_age           — building age in years
#   Step 2  → is_corporate_owner     — LLC/TRUST/CORP in owner name
#   Step 3  → is_absentee_owner      — no homestead exemption
#   Step 4  → is_out_of_state_owner  — owner state != FL
#   Step 5  → price_per_sqft         — JV / living area
#   Step 6  → lot_coverage_ratio     — living area / land sqft
#   Step 7  → renovation_gap         — effective year - actual year
#   Step 8  → value_tier             — JV bracket label
#   Step 9  → neighborhood_value_ratio — property JV vs neighbourhood mean
#   Step 10 → distressed_sale_history — QUAL_CD1 == U
#   Step 11 → never_sold             — SALE_PRC1 == 0
#   Step 12 → value_tier_encoded     — value_tier converted to integer
#   Step 13 → hot_lead               — weak supervision label (target column)
#   Step 14 → Save to leads_features table
#
# STATE CONTRACT:
#   Reads  → state['clean_df']     (DataFrame from cleaner)
#   Writes → state['featured_df']  (DataFrame with all features)
#   Errors → state['errors']       (appended on failure)
#
# NOTE ON FEATURE_COLUMNS:
#   The model trains only on columns listed in
#   project_config.FEATURE_COLUMNS.
#   Extra columns (value_tier, is_corporate_owner raw etc.)
#   are kept in the table for reference and Power BI use.
# ============================================================

import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import project_config as config

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

CORPORATE_KEYWORDS = [
    'LLC', 'INC', 'CORP', 'TRUST', 'ESTATE',
    'PROPERTIES', 'GROUP', 'HOLDINGS', 'PARTNERS',
    'INVESTMENTS', 'REALTY', 'ENTERPRISES', 'LTD'
]

# Value tier bins and labels — matches notebook exactly
TIER_BINS   = [0, 50000, 150000, 300000, 500000, float('inf')]
TIER_LABELS = ['distressed', 'entry', 'mid', 'upper', 'premium']

# Encoding map for value_tier → integer for ML model
TIER_ENCODING = {
    'distressed': 0,
    'entry'     : 1,
    'mid'       : 2,
    'upper'     : 3,
    'premium'   : 4,
}


# ============================================================
# INDIVIDUAL FEATURE STEPS
# Each step takes a DataFrame, returns it with new column added.
# ============================================================

def _step1_property_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Property age in years.
    Uses current year dynamically so it stays accurate in production.
    Negative values indicate data errors in ACT_YR_BLT — clamped to 0.
    """
    current_year     = datetime.now().year
    df['property_age'] = current_year - df['ACT_YR_BLT']
    df['property_age'] = df['property_age'].clip(lower=0)

    logger.info(
        f"Step 1  | property_age          | "
        f"mean={df['property_age'].mean():.1f}yrs | "
        f"negative clamped={( (current_year - df['ACT_YR_BLT']) < 0 ).sum()}"
    )
    return df


def _step2_corporate_owner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2 — Corporate owner flag.
    1 if OWN_NAME contains any corporate keyword, 0 otherwise.
    Corporate owners are more likely to sell for financial reasons.
    """
    df['is_corporate_owner'] = (
        df['OWN_NAME']
        .astype(str)
        .str.upper()
        .str.contains('|'.join(CORPORATE_KEYWORDS), na=False)
        .astype(int)
    )

    logger.info(
        f"Step 2  | is_corporate_owner    | "
        f"corporate={df['is_corporate_owner'].sum():,} | "
        f"individual={( df['is_corporate_owner'] == 0 ).sum():,}"
    )
    return df


def _step3_absentee_owner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3 — Absentee owner flag.
    1 if JV_HMSTD == 0 (no homestead exemption filed).
    Homestead exemption is only granted to owner-occupants in Florida.
    Zero homestead = owner does not live at the property.
    """
    df['is_absentee_owner'] = (df['JV_HMSTD'] == 0).astype(int)

    logger.info(
        f"Step 3  | is_absentee_owner     | "
        f"absentee={df['is_absentee_owner'].sum():,} | "
        f"owner-occupied={( df['is_absentee_owner'] == 0 ).sum():,}"
    )
    return df


def _step4_out_of_state_owner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4 — Out of state owner flag.
    1 if OWN_STATE_ is not FL and not empty.
    Out of state owners are less attached to the property
    and more likely to sell at a discount.
    """
    owner_state = df['OWN_STATE_'].astype(str).str.strip()

    df['is_out_of_state_owner'] = (
        (owner_state != 'FL') &
        (owner_state != '')
    ).astype(int)

    logger.info(
        f"Step 4  | is_out_of_state_owner | "
        f"out-of-state={df['is_out_of_state_owner'].sum():,} | "
        f"Florida={( df['is_out_of_state_owner'] == 0 ).sum():,}"
    )
    return df


def _step5_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5 — Price per square foot.
    JV / TOT_LVG_AR.
    TOT_LVG_AR NaN values (vacant land) produce NaN here.
    NaN filled with 0 — confirmed per project decision.
    """
    df['price_per_sqft'] = (
        df['JV'] / df['TOT_LVG_AR']
    ).fillna(0)

    logger.info(
        f"Step 5  | price_per_sqft        | "
        f"valid={df['price_per_sqft'].gt(0).sum():,} | "
        f"zero(vacant)={df['price_per_sqft'].eq(0).sum():,}"
    )
    return df


def _step6_lot_coverage_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6 — Lot coverage ratio.
    TOT_LVG_AR / LND_SQFOOT.
    LND_SQFOOT zeros replaced with NaN before dividing
    to avoid division by zero.
    NaN result filled with 0.
    """
    df['lot_coverage_ratio'] = (
        df['TOT_LVG_AR'] / df['LND_SQFOOT'].replace(0, np.nan)
    ).fillna(0)

    logger.info(
        f"Step 6  | lot_coverage_ratio    | "
        f"valid={df['lot_coverage_ratio'].gt(0).sum():,} | "
        f"zero={df['lot_coverage_ratio'].eq(0).sum():,}"
    )
    return df


def _step7_renovation_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 7 — Renovation gap.
    EFF_YR_BLT - ACT_YR_BLT.
    Zero means never renovated since construction.
    Positive means property was updated after being built.
    """
    df['renovation_gap'] = df['EFF_YR_BLT'] - df['ACT_YR_BLT']

    logger.info(
        f"Step 7  | renovation_gap        | "
        f"never renovated={df['renovation_gap'].eq(0).sum():,} | "
        f"mean gap={df['renovation_gap'].mean():.1f}yrs"
    )
    return df


def _step8_value_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 8 — Value tier bracket.
    Categorizes JV into 5 price bands:
        distressed : $0      - $50,000
        entry      : $50,001 - $150,000
        mid        : $150,001- $300,000
        upper      : $300,001- $500,000
        premium    : $500,001+

    Kept as string category in table for Power BI filtering.
    Encoded to integer in Step 12 for ML model.
    """
    df['value_tier'] = pd.cut(
        df['JV'],
        bins   = TIER_BINS,
        labels = TIER_LABELS,
        right  = True
    )

    logger.info(
        f"Step 8  | value_tier            | "
        f"distressed+entry={df['value_tier'].isin(['distressed','entry']).sum():,}"
    )
    return df


def _step9_neighborhood_value_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 9 — Neighbourhood value ratio.
    Property JV / mean JV of all properties in same NBRHD_CD.
    Ratio < 0.8 means undervalued vs neighbourhood — motivated seller signal.
    Ratio > 1.2 means overvalued vs neighbourhood.
    NaN filled with 1.0 (neutral) for properties with no neighbourhood group.
    """
    nbrhd_mean_jv = df.groupby('NBRHD_CD')['JV'].transform('mean')

    df['neighborhood_value_ratio'] = (
        df['JV'] / nbrhd_mean_jv
    ).fillna(1.0)

    logger.info(
        f"Step 9  | neighborhood_value_ratio | "
        f"undervalued(<0.8)={df['neighborhood_value_ratio'].lt(0.8).sum():,} | "
        f"overvalued(>1.2)={df['neighborhood_value_ratio'].gt(1.2).sum():,}"
    )
    return df


def _step10_distressed_sale_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 10 — Distressed sale history flag.
    1 if QUAL_CD1 == 'U' (unqualified sale).
    Unqualified sales indicate distress, foreclosure, or
    non-arm's-length transactions — strong motivated seller signal.
    """
    df['distressed_sale_history'] = (
        df['QUAL_CD1']
        .astype(str)
        .str.strip()
        .str.upper()
        .str.startswith('U')
    ).astype(int)

    logger.info(
        f"Step 10 | distressed_sale_history | "
        f"distressed={df['distressed_sale_history'].sum():,} | "
        f"clean={df['distressed_sale_history'].eq(0).sum():,}"
    )
    return df


def _step11_never_sold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 11 — Never sold flag.
    1 if SALE_PRC1 == 0 (no recorded sale price).
    Zero sale price means property has never been sold
    or was transferred for $0 (gift, inheritance, etc.).
    This is information, not missing data — kept as a feature.
    Moved here from cleaner.py — it is a feature, not a cleaning step.
    """
    df['never_sold'] = (df['SALE_PRC1'] == 0).astype(int)

    logger.info(
        f"Step 11 | never_sold            | "
        f"never sold={df['never_sold'].sum():,} | "
        f"has sale={df['never_sold'].eq(0).sum():,} | "
        f"{df['never_sold'].mean()*100:.1f}%"
    )
    return df


def _step12_value_tier_encoded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 12 — Encode value_tier to integer for ML model.
    GradientBoostingClassifier requires numeric input.
    value_tier string labels are mapped to integers 0-4.
        distressed → 0
        entry      → 1
        mid        → 2
        upper      → 3
        premium    → 4
    Original value_tier column kept for Power BI display.
    """
    df['value_tier_encoded'] = (
        df['value_tier']
        .map(TIER_ENCODING)
        .fillna(0)
        .astype(int)
    )

    logger.info(
        f"Step 12 | value_tier_encoded    | "
        f"distribution: {df['value_tier_encoded'].value_counts().to_dict()}"
    )
    return df


def _step13_hot_lead_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 13 — Hot lead label via weak supervision.
    Each property scores 1 point per signal triggered.
    If total score >= HOT_LEAD_THRESHOLD (3) → hot_lead = 1.

    7 signals:
        1. never_sold              == 1
        2. is_absentee_owner       == 1
        3. is_out_of_state_owner   == 1
        4. property_age            > 30 years
        5. value_tier in distressed or entry bracket
        6. neighborhood_value_ratio < 0.8 (undervalued)
        7. distressed_sale_history == 1

    This label is the training target for GradientBoostingClassifier.
    Rules give 0 or 1. The model learns the probability weights.
    """
    signal_score = (
        df['never_sold'] +
        df['is_absentee_owner'] +
        df['is_out_of_state_owner'] +
        (df['property_age'] > 30).astype(int) +
        df['value_tier'].isin(['distressed', 'entry']).astype(int) +
        (df['neighborhood_value_ratio'] < 0.8).astype(int) +
        df['distressed_sale_history']
    )

    df['hot_lead'] = (
        signal_score >= config.HOT_LEAD_THRESHOLD
    ).astype(int)

    hot_count  = df['hot_lead'].sum()
    cold_count = df['hot_lead'].eq(0).sum()
    hot_pct    = df['hot_lead'].mean() * 100

    logger.info(f"Step 13 | hot_lead label        | "
                f"hot={hot_count:,} | cold={cold_count:,} | {hot_pct:.1f}%")
    logger.info(f"        | Signal score distribution:")

    for score, count in signal_score.value_counts().sort_index().items():
        logger.info(f"        |   score {score} → {count:,} properties")

    return df


def _step14_save_to_db(df: pd.DataFrame):
    """
    Step 14 — Save featured DataFrame to leads_features table.
    Uses if_exists='replace' — same pattern as cleaner.py.
    Verifies save by reading count back from database.
    """
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df.to_sql('leads_features', conn, if_exists='replace', index=False)
        saved_count = pd.read_sql_query(
            "SELECT COUNT(*) as total FROM leads_features", conn
        ).iloc[0]['total']
        logger.info(
            f"Step 14 | Saved to DB           | "
            f"leads_features → {saved_count:,} records"
        )
    except Exception as e:
        logger.error(f"Step 14 | Save failed: {e}")
        raise
    finally:
        conn.close()


# ============================================================
# MAIN NODE FUNCTION — called by main.py
# ============================================================

def run_features(state: dict) -> dict:
    """
    Main feature engineering node. Called by main.py orchestrator.

    Reads clean data from state['clean_df'].
    If clean_df is None or empty, falls back to loading
    from leads_clean table directly.

    Runs all 14 steps in sequence.
    Writes featured DataFrame to state['featured_df'].
    Saves to leads_features table in database.

    Args:
        state : Shared pipeline State dict from build_state()

    Returns:
        Updated state with state['featured_df'] populated.
        On failure, appends to state['errors'] and returns
        state with featured_df = None.
    """
    logger.info("-" * 55)
    logger.info("NODE: run_features")
    logger.info("-" * 55)

    try:
        # ── Load data ─────────────────────────────────────────
        df = state.get("clean_df")

        if df is None or (hasattr(df, '__len__') and len(df) == 0):
            logger.info(
                "clean_df is empty. Loading from leads_clean table directly."
            )
            conn = sqlite3.connect(state["db_path"])
            df   = pd.read_sql_query("SELECT * FROM leads_clean", conn)
            conn.close()
            logger.info(f"Loaded {len(df):,} records from leads_clean.")

        if df is None or len(df) == 0:
            logger.error("No data for feature engineering. leads_clean is empty.")
            state["errors"].append({
                "stage" : "features",
                "error" : "No data available. leads_clean table is empty."
            })
            state["featured_df"] = None
            return state

        logger.info(f"Starting records : {len(df):,}")

        # ── Run feature steps in order ────────────────────────
        df = _step1_property_age(df)
        df = _step2_corporate_owner(df)
        df = _step3_absentee_owner(df)
        df = _step4_out_of_state_owner(df)
        df = _step5_price_per_sqft(df)
        df = _step6_lot_coverage_ratio(df)
        df = _step7_renovation_gap(df)
        df = _step8_value_tier(df)
        df = _step9_neighborhood_value_ratio(df)
        df = _step10_distressed_sale_history(df)
        df = _step11_never_sold(df)
        df = _step12_value_tier_encoded(df)
        df = _step13_hot_lead_label(df)

        # ── Null-fill raw FEATURE_COLUMNS ─────────────────────
        # Raw columns (JV, JV_HMSTD, TOT_LVG_AR etc.) may contain
        # NaN values — for example TOT_LVG_AR was set to NaN by
        # cleaner.py Step 7 for vacant properties.
        # GradientBoostingClassifier cannot receive NaN.
        # Fill with 0 so the model treats missing values as
        # "no recorded data" — a valid signal in itself.
        for col in config.FEATURE_COLUMNS:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col] = df[col].fillna(0)
                logger.info(
                    f"        | Null-fill: {col:<15} "
                    f"filled {null_count:,} NaN → 0"
                )

        # ── Verify all FEATURE_COLUMNS exist ─────────────────
        missing = [
            col for col in config.FEATURE_COLUMNS
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Missing FEATURE_COLUMNS after engineering: {missing}. "
                f"Check features.py steps match project_config.FEATURE_COLUMNS."
            )

        # ── Verify TARGET_COLUMN exists ───────────────────────
        if config.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{config.TARGET_COLUMN}' not found. "
                f"Check _step13_hot_lead_label()."
            )

        # ── Save to database ──────────────────────────────────
        _step14_save_to_db(df)

        # ── Write to state ────────────────────────────────────
        state["featured_df"] = df

        logger.info("-" * 55)
        logger.info(
            f"run_features complete | "
            f"{len(df):,} records → state['featured_df']"
        )
        logger.info(
            f"FEATURE_COLUMNS verified : {len(config.FEATURE_COLUMNS)} columns"
        )
        logger.info(
            f"TARGET_COLUMN verified   : {config.TARGET_COLUMN}"
        )
        logger.info("-" * 55)

    except Exception as e:
        logger.error(f"run_features failed: {e}")
        state["errors"].append({
            "stage" : "features",
            "error" : str(e)
        })
        state["featured_df"] = None

    return state
