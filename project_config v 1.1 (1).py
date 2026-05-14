# project_config.py - Central Configuration File
# ============================================================
# VERSION HISTORY
#   v1.0 — Original: API, scraper, fields, DB, logging config
#   v1.1 — Added:
#           · PIPELINE_MODE  — TRAIN or PREDICT
#           · RUN_MODE       — FULL_REFRESH / INCREMENTAL / DRY_RUN
#           · DEMO_MODE      — True = wipe and start fresh
#           · build_state()  — creates shared State dict for pipeline
#
# WHAT WAS NOT TOUCHED
#   · BASE_URL        — identical
#   · TOTAL_GOAL      — identical (still fully editable)
#   · MAX_BATCH_SIZE  — identical
#   · REQUEST_TIMEOUT — identical
#   · MAX_RETRIES     — identical
#   · RETRY_DELAY     — identical
#   · OUT_FIELDS      — identical
#   · QUERY_PARAMS    — identical
#   · DB_NAME         — identical
#   · LOG_FILE        — identical
#   · LOG_LEVEL       — identical
# ============================================================

import logging
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. API ENDPOINT (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
BASE_URL = (
    "https://services9.arcgis.com/Gh9awoU677aKree0/arcgis/rest/services"
    "/Florida_Statewide_Cadastral/FeatureServer/0/query"
)

# ─────────────────────────────────────────────────────────────
# 2. SCRAPER SETTINGS (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
TOTAL_GOAL      = 8000   # How many records to fetch. Edit freely.
MAX_BATCH_SIZE  = 2000   # Records per API request. Keep at 2000.
REQUEST_TIMEOUT = 30     # Seconds before request times out.
MAX_RETRIES     = 3      # How many times to retry a failed request.
RETRY_DELAY     = 5      # Seconds between retries (multiplied by attempt).

# ─────────────────────────────────────────────────────────────
# 3. DATA FIELDS (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
OUT_FIELDS = [
    "OBJECTID",
    "OWN_NAME",
    "OWN_ADDR1",
    "OWN_STATE_",
    "PHY_ADDR1",
    "PHY_CITY",
    "PHY_ZIPCD",
    "JV",
    "JV_HMSTD",
    "JV_CHNG",
    "LND_VAL",
    "LND_SQFOOT",
    "DOR_UC",
    "ACT_YR_BLT",
    "EFF_YR_BLT",
    "TOT_LVG_AR",
    "NO_BULDNG",
    "NO_RES_UNT",
    "SALE_PRC1",
    "SALE_PRC2",
    "SALE_YR1",
    "SALE_YR2",
    "QUAL_CD1",
    "NBRHD_CD"
]

# ─────────────────────────────────────────────────────────────
# 4. REQUEST PARAMETERS (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
QUERY_PARAMS = {
    "where"         : "1=1",
    "outFields"     : ",".join(OUT_FIELDS),
    "returnGeometry": "false",
    "f"             : "pjson",
    "orderByFields" : "OBJECTID ASC"
}

# ─────────────────────────────────────────────────────────────
# 5. DATABASE SETTINGS (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
DB_NAME = "florida_leads.db"

# ─────────────────────────────────────────────────────────────
# 6. LOGGING SETTINGS (v1.0 — unchanged)
# ─────────────────────────────────────────────────────────────
LOG_FILE  = "pipeline.log"
LOG_LEVEL = logging.INFO

# ─────────────────────────────────────────────────────────────
# 7. MODEL SETTINGS (v1.1 — NEW)
# ─────────────────────────────────────────────────────────────
MODEL_DIR      = "models"                          # Folder where model is saved
MODEL_PATH     = "models/hot_lead_model.pkl"       # Trained model file
SCALER_PATH    = "models/scaler.pkl"               # StandardScaler file
F1_THRESHOLD   = 0.65                             # Minimum F1 to accept model
TEST_SIZE      = 0.20                             # 80/20 train/test split
CV_FOLDS       = 5                                # Cross-validation folds
RANDOM_STATE   = 42                               # Reproducibility seed

# ─────────────────────────────────────────────────────────────
# 8. MODE CONTROLLERS (v1.1 — NEW)
# ─────────────────────────────────────────────────────────────
#
# These three settings control the entire system behaviour.
# Change them here. Every module reads from this file.
# Nothing else needs to be touched between modes.
#
# ── PIPELINE_MODE ────────────────────────────────────────────
# Controls WHAT the pipeline does this run.
#
#   "TRAIN"   → scrape → clean → features → train model → save model.pkl
#               Use when: building the system, first run, retraining.
#
#   "PREDICT" → scrape new → clean → features → load model → score → export
#               Use when: model is trained and system is in production.
#               Switch to this after model.pkl is saved and F1 > 0.65.
#
PIPELINE_MODE = "TRAIN"

# ── RUN_MODE ─────────────────────────────────────────────────
# Controls HOW the scraper fetches data this run.
#
#   "DRY_RUN"      → Skip API entirely. Load existing DB data.
#                    Use when: testing cleaner, features, ML logic.
#                    No API calls. Instant. Safe for repeated testing.
#                    USE THIS during all development work.
#
#   "FULL_REFRESH" → Hit API. Fetch all records up to TOTAL_GOAL.
#                    Use when: first real run, supervisor demo,
#                    annual data refresh.
#
#   "INCREMENTAL"  → Hit API. Fetch only records beyond current offset.
#                    Use when: expanding TOTAL_GOAL after first load,
#                    or weekly PREDICT mode runs.
#
RUN_MODE = "DRY_RUN"

# ── DEMO_MODE ─────────────────────────────────────────────────
# Controls WHERE the pipeline starts from.
#
#   False → Normal run. Read pipeline_log. Resume from last stop.
#           Skips stages already completed. Continues where it stopped.
#           Use for: all normal development and production runs.
#
#   True  → Wipe everything. Start from record 1.
#           Clears: leads, leads_clean, leads_featured,
#                   scored_leads, pipeline_log.
#           Use for: supervisor demos, full system reset, fresh testing.
#           After demo: set back to False immediately.
#
#   ⚠ WARNING: DEMO_MODE = True deletes all data. No undo.
#
DEMO_MODE = False

# ─────────────────────────────────────────────────────────────
# QUICK REFERENCE — Common Configurations
# ─────────────────────────────────────────────────────────────
#
#   SITUATION                          SETTINGS
#   ─────────────────────────────────────────────────────────
#   Development / testing code       → RUN_MODE = "DRY_RUN"
#                                      DEMO_MODE = False
#
#   Supervisor demo (full pipeline)  → RUN_MODE = "FULL_REFRESH"
#                                      DEMO_MODE = True
#                                      PIPELINE_MODE = "TRAIN"
#
#   Something broke, start clean     → DEMO_MODE = True
#
#   Model trained, going production  → PIPELINE_MODE = "PREDICT"
#                                      RUN_MODE = "INCREMENTAL"
#                                      DEMO_MODE = False
#
#   Expand training data             → TOTAL_GOAL = 15000
#                                      RUN_MODE = "INCREMENTAL"
#                                      DEMO_MODE = False
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 9. ML FEATURES (v1.1 — NEW)
# ─────────────────────────────────────────────────────────────
# The 13 features used for model training and prediction.
# This list is frozen after model training.
# NEVER change the order — it must match what the model was trained on.
# features.py writes these columns. ml_model.py and predict.py read them.
#
FEATURE_COLUMNS = [
    "value_per_sqft",       # JV / LND_SQFOOT
    "homestead_ratio",      # JV_HMSTD / JV
    "land_to_value_ratio",  # LND_VAL / JV
    "value_change_pct",     # JV_CHNG / JV
    "building_age",         # current_year - ACT_YR_BLT
    "effective_age_gap",    # ACT_YR_BLT - EFF_YR_BLT
    "price_drop_ratio",     # (SALE_PRC1 - SALE_PRC2) / SALE_PRC1
    "years_since_sale",     # current_year - SALE_YR1
    "multi_unit_flag",      # NO_RES_UNT > 1
    "out_of_state_flag",    # OWN_STATE_ != 'FL'
    "absentee_flag",        # OWN_ADDR1 != PHY_ADDR1
    "distressed_uc_flag",   # DOR_UC in distressed codes
    "no_living_area_flag",  # TOT_LVG_AR == 0
]

# Target column — the label features.py generates via weak supervision
TARGET_COLUMN = "hot_lead"

# DOR_UC codes that indicate distressed / non-standard property use
# Used to generate distressed_uc_flag feature in features.py
DISTRESSED_DOR_CODES = [
    "99",   # Acreage — not agriculture / not classified
    "98",   # Centrally assessed
    "97",   # Undefined
    "00",   # Vacant residential
]

# ─────────────────────────────────────────────────────────────
# 10. WEAK SUPERVISION SETTINGS (v1.1 — NEW)
# ─────────────────────────────────────────────────────────────
# Rules used to generate hot_lead label in features.py.
# A property is labelled hot_lead = 1 if it triggers
# HOT_LEAD_THRESHOLD or more of the 7 signal rules.
# This label is then used to train the GBC model.
#
HOT_LEAD_THRESHOLD   = 3    # Minimum signals to be labelled hot_lead = 1
YEARS_SINCE_SALE_MIN = 10   # Property unsold for this many years = 1 signal
VALUE_DROP_PCT_MIN   = 0.10 # 10% value drop = 1 signal

# ─────────────────────────────────────────────────────────────
# 11. STATE BUILDER (v1.1 — NEW)
# ─────────────────────────────────────────────────────────────

def build_state() -> dict:
    """
    Creates and returns the shared pipeline State dictionary.

    Called once at the start of main.py.
    Every node function (scraper, cleaner, features, model, predictor)
    receives this dict, reads what it needs, writes its output back,
    and returns the updated dict to main.py.

    This is the single source of truth shared across all pipeline stages.
    No module needs to import another module's output directly —
    everything flows through State.

    Structure:
        run_id        → unique ID for this execution (set by main.py)
        pipeline_mode → "TRAIN" or "PREDICT" (from config)
        run_mode      → "DRY_RUN" / "FULL_REFRESH" / "INCREMENTAL"
        demo_mode     → True or False (from config)
        db_path       → path to SQLite database
        raw_df        → scraper output  (DataFrame or None)
        clean_df      → cleaner output (DataFrame or None)
        featured_df   → features output (DataFrame or None)
        scored_df     → predictor output (DataFrame or None)
        model_path    → where model.pkl is saved/loaded
        errors        → list of error dicts from any stage
        stage_times   → dict of stage → duration in seconds
        run_timestamp → ISO timestamp of when this run started

    Returns:
        dict: Fully initialised State ready for main.py to pass
              into the first pipeline node.
    """
    return {
        # ── Identity ─────────────────────────────────────────
        "run_id"        : None,           # Set by main.py after generate_run_id()
        "run_timestamp" : datetime.now().isoformat(),

        # ── Mode flags (read from config above) ──────────────
        "pipeline_mode" : PIPELINE_MODE,
        "run_mode"      : RUN_MODE,
        "demo_mode"     : DEMO_MODE,

        # ── Database ─────────────────────────────────────────
        "db_path"       : DB_NAME,

        # ── Data payloads (filled by each node) ──────────────
        "raw_df"        : None,           # scraper writes this
        "clean_df"      : None,           # cleaner writes this
        "featured_df"   : None,           # features writes this
        "scored_df"     : None,           # predictor writes this

        # ── Model ─────────────────────────────────────────────
        "model_path"    : MODEL_PATH,
        "scaler_path"   : SCALER_PATH,
        "f1_score"      : None,           # ml_model writes this after evaluation

        # ── Tracking ──────────────────────────────────────────
        "errors"        : [],             # each node appends errors here
        "stage_times"   : {},             # main.py writes {stage: secs} here
    }
