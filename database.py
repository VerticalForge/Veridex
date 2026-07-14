# database.py - SQLite Database Handler
# ============================================================
# VERSION HISTORY
#   v1.0 — Original: leads, leads_clean, leads_featured tables
#           + 7 core functions
#   v1.1 — Added:
#           · pipeline_log table  — full audit trail per stage
#           · scored_leads table  — predict.py output
#           · save_pipeline_log() — write one row per stage
#           · get_stage_status()  — did this stage complete this run?
#           · get_last_run_id()   — resume: what was the last run?
#           · log_stage_error()   — separate error entry per stage
#           · reset_all_tables()  — DEMO_MODE full wipe
#           · load_featured_leads() — ml_model.py reads from here
#           · save_scored_leads()   — predict.py writes here
#           · load_scored_leads()   — export/Power BI reads from here
#
# WHAT WAS NOT TOUCHED
#   · TEXT_FIELDS set          — identical to v1.0
#   · initialize_db()          — extended only (new tables added inside)
#   · save_batch()             — identical to v1.0
#   · get_record_count()       — identical to v1.0
#   · load_all_leads()         — identical to v1.0
#   · load_leads_by_city()     — identical to v1.0
#   · load_leads_by_dor()      — identical to v1.0
#   · get_summary_stats()      — identical to v1.0
# ============================================================

import sqlite3
import pandas as pd
import logging
from datetime import datetime
import project_config as config

logger = logging.getLogger(__name__)

# Fields that should be stored as TEXT
# Add any new text/code fields here if you expand later
TEXT_FIELDS = {
    "Parcel", "OWN_NAME", "OWN_ADDR1", "OWN_ADDR2", "OWN_CITY",
    "OWN_STATE", "OWN_STATE_",
    "PHY_ADDR1", "PHY_ADDR2", "PHY_CITY",
    "DOR_UC", "QUAL_CD1", "QUAL_CD2",
    "VI_CD1", "VI_CD2",
    "NBRHD_CD", "MKT_AR",
    "FILE_T", "APP_STAT", "S_LEGAL",
    "PARCEL_ID", "ALT_KEY",
}


# ============================================================
# EXISTING CODE — v1.0 — initialize_db() EXTENDED (not rewritten)
# ============================================================

def initialize_db():
    """
    Creates all tables and indexes if they don't exist.

    v1.0: Created leads table + 5 indexes.
    v1.1: Added pipeline_log and scored_leads tables.
          leads, leads_clean, leads_featured are untouched —
          they already exist in your database correctly.
    """
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()

    # ── leads table (v1.0 — unchanged) ───────────────────────
    column_defs = []
    for field in config.OUT_FIELDS:
        if field == "OBJECTID":
            column_defs.append(f"{field} INTEGER UNIQUE")
        elif field in TEXT_FIELDS:
            column_defs.append(f"{field} TEXT")
        else:
            column_defs.append(f"{field} REAL")

    columns_sql = ", ".join(column_defs)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS leads (
            {columns_sql},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    indexes = {
        "idx_city"   : "PHY_CITY",
        "idx_zip"    : "PHY_ZIPCD",
        "idx_dor_uc" : "DOR_UC",
        "idx_sale_yr": "SALE_YR1",
        "idx_jv"     : "JV",
    }
    for index_name, column in indexes.items():
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON leads ({column})"
        )

    # ── pipeline_log table (v1.1 — NEW) ──────────────────────
    # One row per stage per run.
    # run_id groups all stages of a single pipeline execution.
    # This is the audit trail your supervisor will see.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id         TEXT    NOT NULL,
            stage          TEXT    NOT NULL,
            status         TEXT    NOT NULL,
            records_in     INTEGER DEFAULT 0,
            records_out    INTEGER DEFAULT 0,
            duration_secs  REAL    DEFAULT 0.0,
            pipeline_mode  TEXT,
            run_mode       TEXT,
            error_message  TEXT,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pipeline_run_id
        ON pipeline_log (run_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pipeline_stage
        ON pipeline_log (stage, status)
    """)

    # ── scored_leads table (v1.1 — NEW) ──────────────────────
    # Output of predict.py. Contains OBJECTID + hot_lead_score
    # + rank + the run_id that produced this score.
    # Power BI / CRM reads from this table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scored_leads (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           TEXT    NOT NULL,
            OBJECTID         INTEGER NOT NULL,
            hot_lead_score   REAL    NOT NULL,
            rank             INTEGER NOT NULL,
            pipeline_mode    TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, OBJECTID)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scored_run_id
        ON scored_leads (run_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scored_rank
        ON scored_leads (rank)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully. All tables ready.")


# ============================================================
# EXISTING FUNCTIONS — v1.0 — IDENTICAL, NOT TOUCHED
# ============================================================

def save_batch(batch: list):
    """Saves a list of records. Skips duplicates automatically."""
    if not batch:
        return

    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()

    keys         = config.OUT_FIELDS
    placeholders = ", ".join([f":{k}" for k in keys])
    sql          = f"""
        INSERT OR IGNORE INTO leads ({', '.join(keys)})
        VALUES ({placeholders})
    """

    try:
        cursor.executemany(sql, batch)
        conn.commit()
        logger.info(f"Batch saved: {cursor.rowcount} new records inserted.")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        conn.close()


def get_record_count() -> int:
    """Returns total number of records in the leads table."""
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads")
    count  = cursor.fetchone()[0]
    conn.close()
    return count


def load_all_leads() -> pd.DataFrame:
    """Loads all raw records into a Pandas DataFrame."""
    conn = sqlite3.connect(config.DB_NAME)
    df   = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()
    logger.info(f"Loaded {len(df):,} raw records from leads table.")
    return df


def load_leads_by_city(city: str) -> pd.DataFrame:
    """Loads records filtered by city name."""
    conn = sqlite3.connect(config.DB_NAME)
    df   = pd.read_sql_query(
        "SELECT * FROM leads WHERE PHY_CITY = ?", conn, params=(city.upper(),)
    )
    conn.close()
    return df


def load_leads_by_dor(dor_codes: list) -> pd.DataFrame:
    """Loads records filtered by DOR_UC codes. Pass a list e.g. ['001','002']"""
    placeholders = ",".join(["?" for _ in dor_codes])
    conn = sqlite3.connect(config.DB_NAME)
    df   = pd.read_sql_query(
        f"SELECT * FROM leads WHERE DOR_UC IN ({placeholders})",
        conn, params=dor_codes
    )
    conn.close()
    return df


def get_summary_stats() -> dict:
    """Returns a quick summary of database contents."""
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    stats  = {}

    cursor.execute("SELECT COUNT(*) FROM leads")
    stats["total_records"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT PHY_CITY) FROM leads")
    stats["unique_cities"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT PHY_ZIPCD) FROM leads")
    stats["unique_zips"] = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(JV), MAX(JV), AVG(JV) FROM leads WHERE JV > 0")
    row = cursor.fetchone()
    stats["min_value"] = round(row[0] or 0, 2)
    stats["max_value"] = round(row[1] or 0, 2)
    stats["avg_value"] = round(row[2] or 0, 2)

    cursor.execute("SELECT COUNT(*) FROM leads WHERE TOT_LVG_AR > 0")
    stats["has_living_area"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM leads WHERE SALE_PRC2 > 0")
    stats["has_second_sale"] = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM leads "
        "WHERE OWN_STATE_ != 'FL' AND OWN_STATE_ IS NOT NULL AND OWN_STATE_ != ''"
    )
    stats["out_of_state_owners"] = cursor.fetchone()[0]

    conn.close()
    return stats


# ============================================================
# v1.1 ADDITIONS — NEW FUNCTIONS BELOW
# All functions below are NEW. Nothing above was modified.
# ============================================================

# ── Run ID Generator ─────────────────────────────────────────

def generate_run_id() -> str:
    """
    Generates a unique run ID for one full pipeline execution.
    Format: RUN_YYYYMMDD_HHMMSS
    Example: RUN_20250510_060000

    Called once at the start of main.py and passed into state.
    Every pipeline_log row for that execution shares this run_id.
    This is how you group all stages of one run together.
    """
    return datetime.now().strftime("RUN_%Y%m%d_%H%M%S")


# ── pipeline_log Functions ────────────────────────────────────

def save_pipeline_log(
    run_id        : str,
    stage         : str,
    status        : str,
    records_in    : int  = 0,
    records_out   : int  = 0,
    duration_secs : float = 0.0,
    error_message : str  = None
):
    """
    Writes one row to pipeline_log for a completed/failed stage.

    Called by main.py after each node function returns.

    Args:
        run_id        : Unique ID for this pipeline run (from generate_run_id)
        stage         : Stage name — "scraper" / "cleaner" / "features"
                        / "model" / "predictor"
        status        : "completed" / "failed" / "skipped"
        records_in    : Records that entered this stage
        records_out   : Records that came out of this stage
        duration_secs : How long this stage took in seconds
        error_message : None if success, error string if failed

    Example log row your supervisor will see:
        run_id        = "RUN_20250510_060000"
        stage         = "cleaner"
        status        = "completed"
        records_in    = 8000
        records_out   = 4837
        duration_secs = 3.2
        pipeline_mode = "TRAIN"
        run_mode      = "FULL_REFRESH"
        error_message = None
    """
    pipeline_mode = getattr(config, "PIPELINE_MODE", "TRAIN")
    run_mode      = getattr(config, "RUN_MODE",      "INCREMENTAL")

    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO pipeline_log
                (run_id, stage, status, records_in, records_out,
                 duration_secs, pipeline_mode, run_mode, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, stage, status, records_in, records_out,
            round(duration_secs, 2), pipeline_mode, run_mode, error_message
        ))
        conn.commit()
        logger.info(
            f"pipeline_log | {stage} | {status} | "
            f"in={records_in:,} out={records_out:,} | "
            f"{duration_secs:.1f}s"
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to write pipeline_log: {e}")
    finally:
        conn.close()


def log_stage_error(run_id: str, stage: str, error: str):
    """
    Shortcut to log a failed stage with full error message.
    Called by main.py exception handlers.

    This writes a 'failed' status row so the audit trail
    always shows what went wrong and at which stage.
    """
    save_pipeline_log(
        run_id        = run_id,
        stage         = stage,
        status        = "failed",
        error_message = str(error)
    )
    logger.error(f"Stage '{stage}' failed: {error}")


def get_stage_status(run_id: str, stage: str) -> str | None:
    """
    Returns the status of a specific stage in a specific run.

    Returns:
        "completed" — stage finished successfully
        "failed"    — stage failed
        "skipped"   — stage was skipped
        None        — stage has not run yet in this run_id

    Used by main.py to decide whether to skip a stage
    that already completed (DEMO_MODE=False, resuming a run).

    Example usage in main.py:
        if get_stage_status(run_id, "scraper") == "completed":
            logger.info("Scraper already done. Skipping.")
        else:
            state = scrape_records(state)
    """
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT status FROM pipeline_log
        WHERE run_id = ? AND stage = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (run_id, stage))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_last_run_id() -> str | None:
    """
    Returns the run_id of the most recent pipeline execution.

    Used by main.py on startup when DEMO_MODE=False:
        → "What was the last run? Did all stages complete?"
        → If not complete, resume using that run_id.
        → If complete, generate a new run_id.

    Returns None if no runs exist yet (first ever execution).
    """
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id FROM pipeline_log
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_pipeline_run_summary(run_id: str) -> list[dict]:
    """
    Returns a full summary of all stages for a given run_id.
    Used for logging the complete run summary at end of main.py.

    Your supervisor will see this printed at the end of every run:

        RUN_20250510_060000 SUMMARY
        ─────────────────────────────────────────────────────
        scraper    | completed | in=0      | out=8000  | 14.2s
        cleaner    | completed | in=8000   | out=4837  | 3.1s
        features   | completed | in=4837   | out=4837  | 1.8s
        model      | completed | in=4837   | out=4837  | 22.4s
        ─────────────────────────────────────────────────────
    """
    conn = sqlite3.connect(config.DB_NAME)
    df   = pd.read_sql_query("""
        SELECT stage, status, records_in, records_out,
               duration_secs, error_message, created_at
        FROM pipeline_log
        WHERE run_id = ?
        ORDER BY created_at ASC
    """, conn, params=(run_id,))
    conn.close()
    return df.to_dict(orient="records")


# ── Data Loading Functions (for pipeline nodes) ───────────────

def load_clean_leads() -> pd.DataFrame:
    """
    Loads all records from leads_clean table.
    Called by features.py (feature engineering node reads clean data).

    Returns empty DataFrame if table is empty or doesn't exist.
    """
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM leads_clean", conn)
        logger.info(f"Loaded {len(df):,} records from leads_clean.")
        return df
    except Exception as e:
        logger.error(f"Could not load leads_clean: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def load_featured_leads() -> pd.DataFrame:
    """
    Loads all records from leads_features table.
    Called by ml_model.py (model training reads engineered features).

    Returns empty DataFrame if table is empty or doesn't exist.
    """
    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM leads_features", conn)
        logger.info(f"Loaded {len(df):,} records from leads_features.")
        return df
    except Exception as e:
        logger.error(f"Could not load leads_features: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# ── scored_leads Functions ────────────────────────────────────

def save_scored_leads(df: pd.DataFrame, run_id: str):
    """
    Saves prediction results to scored_leads table.
    Called by predict.py after scoring.

    Args:
        df     : DataFrame with columns [OBJECTID, hot_lead_score, rank]
        run_id : Current run ID from state

    The UNIQUE(run_id, OBJECTID) constraint prevents duplicate
    scores for the same property in the same run.
    """
    if df is None or df.empty:
        logger.warning("save_scored_leads: Empty DataFrame. Nothing saved.")
        return

    pipeline_mode = getattr(config, "PIPELINE_MODE", "TRAIN")
    df            = df.copy()
    df["run_id"]        = run_id
    df["pipeline_mode"] = pipeline_mode

    conn = sqlite3.connect(config.DB_NAME)
    try:
        df[["run_id", "OBJECTID", "hot_lead_score", "rank", "pipeline_mode"]]\
            .to_sql(
                "scored_leads",
                conn,
                if_exists = "append",
                index     = False
            )
        logger.info(f"Saved {len(df):,} scored leads for run {run_id}.")
    except Exception as e:
        logger.error(f"Failed to save scored leads: {e}")
    finally:
        conn.close()


def load_scored_leads(run_id: str = None) -> pd.DataFrame:
    """
    Loads scored leads from scored_leads table.

    Args:
        run_id : If provided, loads only scores from that run.
                 If None, loads the most recent run's scores.

    Used by:
        · Power BI export
        · CRM export
        · Dashboard
        · Manual inspection after predict.py runs
    """
    conn = sqlite3.connect(config.DB_NAME)
    try:
        if run_id:
            df = pd.read_sql_query(
                "SELECT * FROM scored_leads WHERE run_id = ? ORDER BY rank ASC",
                conn, params=(run_id,)
            )
        else:
            # Load most recent run automatically
            latest = pd.read_sql_query(
                "SELECT run_id FROM scored_leads ORDER BY created_at DESC LIMIT 1",
                conn
            )
            if latest.empty:
                logger.warning("No scored leads found in database.")
                return pd.DataFrame()
            latest_run_id = latest.iloc[0]["run_id"]
            df = pd.read_sql_query(
                "SELECT * FROM scored_leads WHERE run_id = ? ORDER BY rank ASC",
                conn, params=(latest_run_id,)
            )
            logger.info(f"Loaded scores from most recent run: {latest_run_id}")

        logger.info(f"Loaded {len(df):,} scored leads.")
        return df
    except Exception as e:
        logger.error(f"Failed to load scored leads: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# ── DEMO_MODE Reset Function ──────────────────────────────────

def reset_all_tables():
    """
    Wipes ALL data from all tables. Keeps table structure intact.
    Called by main.py when DEMO_MODE = True in project_config.py.

    What gets wiped:
        · leads            → raw fetched records
        · leads_clean      → cleaned records
        · leads_featured   → engineered features
        · scored_leads     → prediction results
        · pipeline_log     → full audit trail

    What is NOT touched:
        · Table structure (columns, indexes) — preserved
        · project_config.py settings — preserved
        · model.pkl — NOT wiped (you may want to keep trained model)

    After this runs, the next pipeline execution starts from
    record 1 as if the system was just deployed for the first time.

    DEMO WORKFLOW:
        1. Set DEMO_MODE = True in project_config.py
        2. Set RUN_MODE  = "FULL_REFRESH"
        3. python main.py
        → reset_all_tables() called automatically
        → Full pipeline runs from zero
        → Supervisor sees complete execution

    SAFETY: This function logs a warning before wiping.
            There is no undo. Use only for demos/testing.
    """
    logger.warning("=" * 55)
    logger.warning("DEMO_MODE = True — WIPING ALL TABLE DATA")
    logger.warning("This cannot be undone. Starting fresh.")
    logger.warning("=" * 55)

    tables = ["leads", "leads_clean", "leads_features",
              "scored_leads", "pipeline_log"]

    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()

    try:
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"  Cleared: {table}")
            except sqlite3.OperationalError:
                # Table doesn't exist yet — safe to ignore
                logger.info(f"  Skipped (not found): {table}")

        conn.commit()
        logger.info("All tables cleared. System ready for fresh run.")

    except sqlite3.Error as e:
        logger.error(f"reset_all_tables failed: {e}")
        conn.rollback()
    finally:
        conn.close()
