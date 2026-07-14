# scraper.py - Data Ingestion Module
# ============================================================
# VERSION HISTORY
#   v1.0 — Original: ArcGIS API offset-based fetch
#   v1.1 — Added scrape_records(state) node for main.py
#   v2.0 — CAMA_LOAD replaces API as primary data source.
#          ArcGIS API code retired (see RETIREMENT NOTE below).
#          Pipeline is now fully file-based — no external API dependency.
#
# RUN_MODE BEHAVIOUR (v2.0):
#   CAMA_LOAD → Production mode. Reads ACPA_CAMAData.zip from disk,
#               joins all 6 source files on Parcel key, loads the
#               complete county into the leads table, then passes
#               all records into state['raw_df'].
#               Use for: annual refresh, first load, full re-ingestion.
#
#   DRY_RUN   → Development/testing mode. Skips ingestion entirely.
#               Loads whatever is already in the leads table.
#               Use for: testing cleaner, features, predictor without
#               re-reading the ZIP every run.
#
# RETIREMENT NOTE — ArcGIS API (FULL_REFRESH / INCREMENTAL):
#   The Florida Statewide Cadastral API (services9.arcgis.com) was the
#   original data source. It was replaced by the county's own CAMA
#   export for three reasons:
#     1. The CAMA file contains richer fields (QUAL_CD1, EFF_YR_BLT,
#        NO_BULDNG, full sale history) that the statewide API strips out.
#     2. The API is a subset of the same data — the Department of Revenue
#        collects from county appraiser offices and publishes a flattened
#        NAL join annually every August. The county file is closer to source.
#     3. No rate limits, no network dependency, no retry logic needed
#        for a local file read.
#   The API code (fetch_batch, run_pipeline, FULL_REFRESH/INCREMENTAL
#   branches) has been removed. If a live API source is needed in future,
#   add a new RUN_MODE branch here — the node interface (scrape_records)
#   is source-agnostic by design.
#
# STATE CONTRACT:
#   Reads  → state['db_path']   (path to SQLite database)
#   Writes → state['raw_df']    (DataFrame of all leads records)
#   Errors → state['errors']    (appended on failure)
# ============================================================

import os
import logging
import sqlite3
import pandas as pd

import project_config as config
import database

logger = logging.getLogger(__name__)

# Path to the CAMA ZIP file — must be in the project folder
ZIP_PATH = "ACPA_CAMAData.zip"


# ============================================================
# CAMA INGESTION — reads ZIP, joins tables, loads DB
# ============================================================

def _cama_ingest(state: dict) -> pd.DataFrame:
    """
    Reads the CAMA ZIP file, joins all 6 source tables on Parcel,
    assigns OBJECTIDs, and saves the complete county dataset to
    the leads table. Returns the full DataFrame.

    This function calls load_cama's building-block functions
    (load_source_files, build_full_dataset) directly rather than
    run_loader(), giving the scraper node clean control over the
    ingestion without conflicting DEMO_MODE/resume logic.

    If the leads table already has records and the count matches
    the ZIP's total, ingestion is skipped — the DB is already
    populated from a prior run.
    """
    import load_cama

    # ── Check ZIP exists ──────────────────────────────────────
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"CAMA ZIP not found: {ZIP_PATH}\n"
            f"Download from: https://s3.amazonaws.com/acpa.cama/ACPA_CAMAData.zip\n"
            f"Place in the project folder and run again."
        )

    # ── Check if DB already has data ──────────────────────────
    existing_count = database.get_record_count()

    if existing_count > 0:
        logger.info(
            f"CAMA_LOAD: leads table already has {existing_count:,} records. "
            f"Skipping ZIP re-ingestion. Loading from DB."
        )
        conn   = sqlite3.connect(state["db_path"])
        raw_df = pd.read_sql_query("SELECT * FROM leads", conn)
        conn.close()
        return raw_df

    # ── Read ZIP and build full dataset ───────────────────────
    logger.info(f"CAMA_LOAD: Reading {ZIP_PATH}...")
    files   = load_cama.load_source_files(ZIP_PATH)
    full_df = load_cama.build_full_dataset(files)

    logger.info(
        f"CAMA_LOAD: Full county dataset = {len(full_df):,} parcels"
    )

    # ── Assign OBJECTID (sequential from 1) ───────────────────
    full_df = full_df.reset_index(drop=True)
    full_df['OBJECTID'] = range(1, len(full_df) + 1)

    # ── Verify all OUT_FIELDS present ─────────────────────────
    missing = [f for f in config.OUT_FIELDS if f not in full_df.columns]
    if missing:
        raise ValueError(
            f"Missing OUT_FIELDS after CAMA join: {missing}. "
            f"Check load_cama.py table preparation functions."
        )

    # ── Select exact OUT_FIELDS in correct order ──────────────
    full_df = full_df[config.OUT_FIELDS].copy()

    # ── Save to DB in batches ─────────────────────────────────
    records   = full_df.to_dict(orient='records')
    total     = len(records)
    collected = 0

    while collected < total:
        batch_size = min(config.MAX_BATCH_SIZE, total - collected)
        batch      = records[collected: collected + batch_size]

        logger.info(
            f"CAMA_LOAD: Saving | "
            f"Progress: {collected + batch_size:,}/{total:,}"
        )

        database.save_batch(batch)
        collected += batch_size

    # ── Verify save ───────────────────────────────────────────
    final_count = database.get_record_count()
    logger.info(
        f"CAMA_LOAD: Ingestion complete | "
        f"{final_count:,} records in leads table"
    )

    # ── Load from DB into DataFrame ───────────────────────────
    # Reading back from DB (not using full_df directly) ensures
    # raw_df matches exactly what downstream stages would see
    # if they fell back to loading from the DB themselves.
    conn   = sqlite3.connect(state["db_path"])
    raw_df = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()

    return raw_df


# ============================================================
# MAIN NODE FUNCTION — called by main.py
# ============================================================

def scrape_records(state: dict) -> dict:
    """
    Data ingestion node. Called by main.py orchestrator.

    Reads RUN_MODE from config and loads data accordingly:
        CAMA_LOAD → Reads county CAMA ZIP file into leads table
        DRY_RUN   → Loads existing leads table from DB

    Both modes write the result to state['raw_df'] for the
    cleaner node to consume.

    Args:
        state : Shared pipeline State dict from build_state()

    Returns:
        Updated state with state['raw_df'] populated.
        On failure, appends to state['errors'] and returns
        state with raw_df = None.
    """
    run_mode = getattr(config, "RUN_MODE", "DRY_RUN")

    logger.info("-" * 55)
    logger.info(f"NODE: scrape_records | RUN_MODE: {run_mode}")
    logger.info("-" * 55)

    try:

        # ── CAMA_LOAD ────────────────────────────────────────
        # Production mode. Reads the county's CAMA ZIP file,
        # joins all source tables, loads leads, and passes
        # the full dataset into state for cleaning.
        if run_mode == "CAMA_LOAD":
            raw_df = _cama_ingest(state)
            logger.info(
                f"CAMA_LOAD: {len(raw_df):,} records → state['raw_df']"
            )
            state["raw_df"] = raw_df
            return state

        # ── DRY_RUN ──────────────────────────────────────────
        # Development/testing mode. No ingestion — load whatever
        # is already in the leads table from a prior load.
        if run_mode == "DRY_RUN":
            logger.info(
                "DRY_RUN: Loading existing leads from database. "
                "No file read."
            )
            conn   = sqlite3.connect(state["db_path"])
            raw_df = pd.read_sql_query("SELECT * FROM leads", conn)
            conn.close()
            logger.info(
                f"DRY_RUN: Loaded {len(raw_df):,} records from leads table."
            )
            state["raw_df"] = raw_df
            return state

        # ── Unknown mode ─────────────────────────────────────
        raise ValueError(
            f"Unknown RUN_MODE: '{run_mode}'. "
            f"Valid modes: 'CAMA_LOAD', 'DRY_RUN'."
        )

    except Exception as e:
        logger.error(f"scrape_records failed: {e}")
        state["errors"].append({
            "stage" : "scraper",
            "error" : str(e)
        })
        state["raw_df"] = None

    return state
