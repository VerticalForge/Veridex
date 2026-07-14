# main.py - Veridex Pipeline Orchestrator
# ============================================================
# VERSION : v1.1
# PURPOSE : Single entry point for the entire pipeline.
#           One command runs everything end to end.
#
# USAGE:
#   python main.py
#
# BEHAVIOUR IS CONTROLLED ENTIRELY BY project_config.py:
#   PIPELINE_MODE = "TRAIN"         → scrape→clean→features→train→save
#   PIPELINE_MODE = "PREDICT"       → scrape→clean→features→score→export
#  #   RUN_MODE      = "CAMA_LOAD"     → read county CAMA ZIP, load leads table
#   RUN_MODE      = "DRY_RUN"       → use existing DB data, skip ingestion
#   DEMO_MODE     = True            → wipe all data, start from zero
#   DEMO_MODE     = False           → resume from last completed stage
#
# PIPELINE FLOW:
#
#   TRAIN mode:
#   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
#   │ scraper  │ → │ cleaner  │ → │ features │ → │ ml_model │
#   └──────────┘   └──────────┘   └──────────┘   └──────────┘
#
#   PREDICT mode:
#   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
#   │ scraper  │ → │ cleaner  │ → │ features │ → │ predictor│
#   └──────────┘   └──────────┘   └──────────┘   └──────────┘
#
# RESUME LOGIC (DEMO_MODE = False):
#   Reads pipeline_log to find last completed stage.
#   Skips already-completed stages automatically.
#   Resumes from the first incomplete stage.
#   If all stages completed → prints summary, exits cleanly.
#
# ============================================================

import logging
import os
import time
from datetime import datetime

import project_config as config
import database

# ── Node imports (placeholder until each module is built) ────
# These imports use a safe pattern:
# If the module exists → import it normally.
# If not yet built    → a stub function is used instead.
# This allows main.py to run and test partial pipelines
# before all modules are complete.

def _import_node(module_name: str, func_name: str):
    """
    Safely imports a node function from a module.
    Returns a stub function if the module is not yet built.
    The stub logs a clear message and returns state unchanged.
    """
    try:
        module = __import__(module_name)
        return getattr(module, func_name)
    except ImportError:
        def stub(state: dict) -> dict:
            logger.warning(
                f"MODULE NOT BUILT YET: {module_name}.{func_name}() "
                f"— returning state unchanged. Build {module_name}.py next."
            )
            return state
        return stub
    except AttributeError:
        def stub(state: dict) -> dict:
            logger.warning(
                f"FUNCTION NOT FOUND: {func_name} in {module_name}.py "
                f"— returning state unchanged."
            )
            return state
        return stub


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """
    Configures logging to both terminal and pipeline.log file.
    Uses settings from project_config.py.
    Called once at the very start of main.py before anything else.
    """
    logging.basicConfig(
        level   = config.LOG_LEVEL,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers= [
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        ]
    )


logger = logging.getLogger(__name__)


# ============================================================
# STAGE RUNNER — core execution wrapper for every node
# ============================================================

def run_stage(
    stage_name : str,
    node_func,
    state      : dict,
    run_id     : str
) -> dict:
    """
    Wraps every pipeline node with:
        · Resume check    — skip if already completed this run
        · Timing          — records duration in state['stage_times']
        · pipeline_log    — writes completion or failure row
        · Error capture   — appends to state['errors'] on failure

    Args:
        stage_name : "scraper" / "cleaner" / "features" / "model" / "predictor"
        node_func  : The actual function to call e.g. scrape_records
        state      : Current pipeline State dict
        run_id     : Current run ID

    Returns:
        Updated state dict after node execution.

    This function is the only place that calls pipeline_log.
    No node function writes to pipeline_log directly.
    Clean separation of concerns.
    """

    # ── Resume check ─────────────────────────────────────────
    # If DEMO_MODE is False and this stage already completed
    # in this run_id → skip it entirely.
    if not state.get("demo_mode"):
        existing_status = database.get_stage_status(run_id, stage_name)
        if existing_status == "completed":
            logger.info(
                f"SKIP  | {stage_name:12s} | Already completed in run {run_id}"
            )
            return state

    # ── Count records entering this stage ────────────────────
    records_in = _count_records(state, stage_name, position="in")

    # ── Execute node ─────────────────────────────────────────
    logger.info(f"START | {stage_name:12s} | records_in={records_in:,}")
    stage_start = time.time()

    try:
        state = node_func(state)
        duration = time.time() - stage_start

        # ── Count records produced by this stage ─────────────
        records_out = _count_records(state, stage_name, position="out")

        # ── Write success to pipeline_log ────────────────────
        database.save_pipeline_log(
            run_id        = run_id,
            stage         = stage_name,
            status        = "completed",
            records_in    = records_in,
            records_out   = records_out,
            duration_secs = duration
        )

        # ── Record timing in state ────────────────────────────
        state["stage_times"][stage_name] = round(duration, 2)

        logger.info(
            f"DONE  | {stage_name:12s} | "
            f"records_out={records_out:,} | "
            f"{duration:.1f}s"
        )

    except Exception as e:
        duration = time.time() - stage_start

        # ── Write failure to pipeline_log ─────────────────────
        database.log_stage_error(
            run_id = run_id,
            stage  = stage_name,
            error  = str(e)
        )

        state["errors"].append({
            "stage"    : stage_name,
            "error"    : str(e),
            "duration" : round(duration, 2)
        })

        logger.error(
            f"FAIL  | {stage_name:12s} | {e} | {duration:.1f}s"
        )

    return state


def _count_records(state: dict, stage_name: str, position: str) -> int:
    """
    Returns record count from the relevant DataFrame in State.

    position="in"  → count what this stage receives as input
    position="out" → count what this stage produced as output

    Stage input/output mapping:
        scraper   in  → leads table count   out → raw_df
        cleaner   in  → raw_df              out → clean_df
        features  in  → clean_df            out → featured_df
        model     in  → featured_df         out → featured_df (same)
        predictor in  → featured_df         out → scored_df
    """
    import sqlite3

    mapping = {
        # stage      : (input_key,      output_key)
        "scraper"    : ("db_leads",     "raw_df"),
        "cleaner"    : ("raw_df",       "clean_df"),
        "features"   : ("clean_df",     "featured_df"),
        "model"      : ("featured_df",  "featured_df"),
        "predictor"  : ("featured_df",  "scored_df"),
         "export"     : ("scored_df",    "export_df"),
    }

    key = mapping.get(stage_name, (None, None))[0 if position == "in" else 1]

    if key == "db_leads":
        # Special case: scraper input is the current DB count
        try:
            conn   = sqlite3.connect(state["db_path"])
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM leads")
            count  = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    df = state.get(key)
    if df is None:
        return 0
    try:
        return len(df)
    except Exception:
        return 0


# ============================================================
# PIPELINE ENTRY LOGIC
# ============================================================

def resolve_run_id(demo_mode: bool) -> str:
    """
    Determines the run_id for this execution.

    DEMO_MODE = True  → Always generate a new run_id (fresh start)
    DEMO_MODE = False → Check if last run was incomplete → resume it
                        If last run was complete → generate new run_id

    Returns:
        run_id string e.g. "RUN_20250510_060000"
    """
    if demo_mode:
        run_id = database.generate_run_id()
        logger.info(f"DEMO_MODE: New run started → {run_id}")
        return run_id
        
    # PREDICT always runs as a fresh scoring pass. Stages are NOT
    # resumed across modes: a completed TRAIN run shares stage names
    # (scraper/cleaner/features) with PREDICT, so resuming it would
    # SKIP those stages — leaving raw_df/clean_df/featured_df unset in
    # memory and aborting before the predictor ever runs. In DRY_RUN
    # the full chain reloads from the leads table in ~2s, so a fresh
    # run is both correct and cheap.
    if config.PIPELINE_MODE == "PREDICT":
        run_id = database.generate_run_id()
        logger.info(f"PREDICT mode: fresh scoring run → {run_id}")
        return run_id

    last_run_id = database.get_last_run_id()

    if last_run_id is None:
        # First ever run
        run_id = database.generate_run_id()
        logger.info(f"First run ever → {run_id}")
        return run_id

    # Check if last run completed all required stages
    pipeline_mode  = config.PIPELINE_MODE
    required_stages = _get_required_stages(pipeline_mode)
    all_done        = all(
        database.get_stage_status(last_run_id, s) == "completed"
        for s in required_stages
    )

    if all_done:
        run_id = database.generate_run_id()
        logger.info(f"Last run complete. New run → {run_id}")
    else:
        run_id = last_run_id
        logger.info(f"Resuming incomplete run → {run_id}")

    return run_id


def _get_required_stages(pipeline_mode: str) -> list:
    """
    Returns the list of stages required for a given PIPELINE_MODE.
    Used by resolve_run_id to check if last run was complete.
    """
    if pipeline_mode == "TRAIN":
        return ["scraper", "cleaner", "features", "predictor", "export"]
    elif pipeline_mode == "PREDICT":
        return ["scraper", "cleaner", "features", "predictor"]
    return []


def _check_model_exists() -> bool:
    """
    Checks if trained model file exists before running PREDICT mode.
    Returns True if model.pkl is found, False otherwise.
    """
    return os.path.exists(config.MODEL_PATH)


# ============================================================
# RUN SUMMARY PRINTER
# ============================================================

def print_run_summary(run_id: str, state: dict):
    """
    Prints a formatted summary table of all stages at end of run.
    Reads from pipeline_log via database.get_pipeline_run_summary().

    Example output your supervisor will see:

    ═══════════════════════════════════════════════════════
    VERIDEX PIPELINE — RUN SUMMARY
    Run ID  : RUN_20250510_060000
    Mode    : TRAIN | FULL_REFRESH
    ═══════════════════════════════════════════════════════
    Stage        Status      In        Out       Time
    ───────────────────────────────────────────────────────
    scraper      completed   0         8,000     14.2s
    cleaner      completed   8,000     4,837     3.1s
    features     completed   4,837     4,837     1.8s
    model        completed   4,837     4,837     22.4s
    ───────────────────────────────────────────────────────
    Total errors : 0
    F1 Score     : 0.71  ✓ (threshold: 0.65)
    ═══════════════════════════════════════════════════════
    """
    rows = database.get_pipeline_run_summary(run_id)

    logger.info("=" * 55)
    logger.info("VERIDEX PIPELINE — RUN SUMMARY")
    logger.info(f"Run ID  : {run_id}")
    logger.info(
        f"Mode    : {state['pipeline_mode']} | {state['run_mode']}"
    )
    logger.info("=" * 55)
    logger.info(
        f"{'Stage':<14} {'Status':<12} {'In':>8} {'Out':>8} {'Time':>8}"
    )
    logger.info("-" * 55)

    for row in rows:
        logger.info(
            f"{row['stage']:<14} "
            f"{row['status']:<12} "
            f"{row['records_in']:>8,} "
            f"{row['records_out']:>8,} "
            f"{row['duration_secs']:>7.1f}s"
        )

    logger.info("-" * 55)

    error_count = len(state.get("errors", []))
    logger.info(f"Total errors : {error_count}")

    # F1 score (TRAIN mode only)
    f1 = state.get("f1_score")
    if f1 is not None:
        threshold = config.F1_THRESHOLD
        status    = "✓" if f1 >= threshold else "✗ BELOW THRESHOLD"
        logger.info(
            f"F1 Score     : {f1:.2f}  {status} (threshold: {threshold})"
        )

    logger.info("=" * 55)

    # Print any errors that occurred
    if error_count > 0:
        logger.error("ERRORS ENCOUNTERED:")
        for err in state["errors"]:
            logger.error(
                f"  Stage: {err['stage']} | {err['error']}"
            )


# ============================================================
# MAIN PIPELINE RUNNER
# ============================================================

def main():
    """
    Main pipeline entry point.
    Reads config, builds state, runs all stages in order,
    writes pipeline_log after each stage, prints summary.
    """

    # ── Step 1: Logging ──────────────────────────────────────
    setup_logging()

    logger.info("=" * 55)
    logger.info("VERIDEX — FLORIDA REAL ESTATE INTELLIGENCE SYSTEM")
    logger.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode    : PIPELINE={config.PIPELINE_MODE} | "
                f"RUN={config.RUN_MODE} | "
                f"DEMO={config.DEMO_MODE}")
    logger.info("=" * 55)

    # ── Step 2: Database initialisation ──────────────────────
    database.initialize_db()

    # ── Step 3: DEMO_MODE — wipe all data ────────────────────
    if config.DEMO_MODE:
        database.reset_all_tables()

    # ── Step 4: PREDICT mode guard ───────────────────────────
    # Cannot predict without a trained model.
    if config.PIPELINE_MODE == "PREDICT" and not _check_model_exists():
        logger.error(
            f"PREDICT mode requires a trained model at: {config.MODEL_PATH}\n"
            f"Run with PIPELINE_MODE='TRAIN' first to build the model."
        )
        return

    # ── Step 5: Build State ───────────────────────────────────
    state         = config.build_state()
    run_id        = resolve_run_id(config.DEMO_MODE)
    state["run_id"] = run_id

    logger.info(f"Run ID  : {run_id}")

    # ── Step 6: Load node functions ───────────────────────────
    # Safe import — stubs used for modules not yet built.
    scrape_records  = _import_node("scraper",   "scrape_records")
    run_cleaning    = _import_node("cleaner",   "run_cleaning")
    run_features    = _import_node("features",  "run_features")
    run_model       = _import_node("ml_model",  "run_model")
    run_predictor   = _import_node("predictor", "run_predictor")
    run_export      = _import_node("export",    "run_export")

    # ── Step 7: TRAIN pipeline ────────────────────────────────
    if config.PIPELINE_MODE == "TRAIN":

        logger.info("PIPELINE MODE: TRAIN")
        logger.info("Flow: scraper → cleaner → features → model")
        logger.info("-" * 55)

        # Stage 1 — Scraper
        state = run_stage("scraper",  scrape_records, state, run_id)

        # Guard: stop if scraper produced nothing
        if _stage_failed(state, "scraper"):
            logger.error("Pipeline stopped: scraper stage failed.")
            print_run_summary(run_id, state)
            return

        # Guard: stop if no records in DB after scraping
        if _no_data(state, "raw_df"):
            logger.warning(
                "Scraper returned 0 records. "
                "Check RUN_MODE and TOTAL_GOAL in project_config.py."
            )
            print_run_summary(run_id, state)
            return

        # Stage 2 — Cleaner
        state = run_stage("cleaner",  run_cleaning,   state, run_id)

        if _stage_failed(state, "cleaner"):
            logger.error("Pipeline stopped: cleaner stage failed.")
            print_run_summary(run_id, state)
            return

        if _no_data(state, "clean_df"):
            logger.error("Cleaner produced 0 records. Check cleaning logic.")
            print_run_summary(run_id, state)
            return

        # Stage 3 — Feature Engineering
        state = run_stage("features", run_features,   state, run_id)

        if _stage_failed(state, "features"):
            logger.error("Pipeline stopped: features stage failed.")
            print_run_summary(run_id, state)
            return

        if _no_data(state, "featured_df"):
            logger.error("Feature engineering produced 0 records.")
            print_run_summary(run_id, state)
            return

        # Stage 4 — Model Training
        state = run_stage("model",    run_model,      state, run_id)

        if _stage_failed(state, "model"):
            logger.error("Pipeline stopped: model stage failed.")
            print_run_summary(run_id, state)
            return

        # F1 threshold check
        f1 = state.get("f1_score")
        if f1 is not None and f1 < config.F1_THRESHOLD:
            logger.warning(
                f"Model F1={f1:.2f} is below threshold {config.F1_THRESHOLD}. "
                f"Review features or training data before switching to PREDICT mode."
            )
        elif f1 is not None:
            logger.info(
                f"Model accepted. F1={f1:.2f} ≥ {config.F1_THRESHOLD}. "
                f"Set PIPELINE_MODE='PREDICT' in config to start scoring leads."
            )

    # ── Step 8: PREDICT pipeline ──────────────────────────────
    elif config.PIPELINE_MODE == "PREDICT":

        logger.info("PIPELINE MODE: PREDICT")
        logger.info("Flow: scraper → cleaner → features → predictor")
        logger.info("-" * 55)

        # Stage 1 — Scraper
        state = run_stage("scraper",   scrape_records, state, run_id)

        if _stage_failed(state, "scraper"):
            logger.error("Pipeline stopped: scraper stage failed.")
            print_run_summary(run_id, state)
            return

        if _no_data(state, "raw_df"):
            logger.info(
                "Scraper returned 0 new records. "
                "No new leads to score this run. Pipeline complete."
            )
            print_run_summary(run_id, state)
            return

        # Stage 2 — Cleaner
        state = run_stage("cleaner",   run_cleaning,   state, run_id)

        if _stage_failed(state, "cleaner") or _no_data(state, "clean_df"):
            logger.error("Pipeline stopped: cleaner stage failed or empty.")
            print_run_summary(run_id, state)
            return

        # Stage 3 — Feature Engineering
        state = run_stage("features",  run_features,   state, run_id)

        if _stage_failed(state, "features") or _no_data(state, "featured_df"):
            logger.error("Pipeline stopped: features stage failed or empty.")
            print_run_summary(run_id, state)
            return

        # Stage 4 — Predictor
        state = run_stage("predictor", run_predictor,  state, run_id)

        if _stage_failed(state, "predictor"):
            logger.error("Pipeline stopped: predictor stage failed.")
            print_run_summary(run_id, state)
            return

        # Stage 5 — Export
        state = run_stage("export", run_export, state, run_id)

        if _stage_failed(state, "export"):
            logger.error("Pipeline stopped: export stage failed.")
            print_run_summary(run_id, state)
            return

        logger.info(
            "Pipeline complete. Scored leads exported to Excel + database."
        )

    else:
        logger.error(
            f"Unknown PIPELINE_MODE: '{config.PIPELINE_MODE}'. "
            f"Must be 'TRAIN' or 'PREDICT'."
        )
        return

    # ── Step 9: Run summary ───────────────────────────────────
    print_run_summary(run_id, state)

    logger.info(
        f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ============================================================
# GUARD HELPERS — used inside main() for clean flow control
# ============================================================

def _stage_failed(state: dict, stage_name: str) -> bool:
    """
    Returns True if the given stage appended an error to state['errors'].
    Used to stop the pipeline cleanly after a failed stage.
    """
    return any(
        err.get("stage") == stage_name
        for err in state.get("errors", [])
    )


def _no_data(state: dict, df_key: str) -> bool:
    """
    Returns True if the DataFrame at state[df_key] is None or empty.
    Used to stop the pipeline if a stage produced no output records.
    """
    df = state.get(df_key)
    if df is None:
        return True
    try:
        return len(df) == 0
    except Exception:
        return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
