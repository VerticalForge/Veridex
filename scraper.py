# scraper.py - Main Data Pipeline
import requests
import time
import logging
import project_config as config
import database


def setup_logging():
    """Sets up logging to both terminal and log file."""
    logging.basicConfig(
        level   = config.LOG_LEVEL,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        handlers= [
            logging.StreamHandler(),
            logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        ]
    )


logger = logging.getLogger(__name__)


def fetch_batch(offset: int, limit: int):
    """
    Fetches one batch of records from the Florida API.
    Retries up to MAX_RETRIES times if the request fails.
    Returns a list of records or None if all retries failed.
    """
    params = config.QUERY_PARAMS.copy()
    params["resultOffset"]      = offset
    params["resultRecordCount"] = limit

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(
                config.BASE_URL,
                params  = params,
                timeout = config.REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data     = response.json()
            features = data.get("features", [])
            return [f["attributes"] for f in features]

        except requests.exceptions.Timeout:
            logger.warning(f"Attempt {attempt}/{config.MAX_RETRIES}: Request timed out.")

        except requests.exceptions.HTTPError as e:
            logger.warning(f"Attempt {attempt}/{config.MAX_RETRIES}: HTTP error: {e}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt}/{config.MAX_RETRIES}: Connection error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

        if attempt < config.MAX_RETRIES:
            wait = config.RETRY_DELAY * attempt
            logger.info(f"Waiting {wait}s before retry...")
            time.sleep(wait)

    logger.error(f"All {config.MAX_RETRIES} attempts failed at offset {offset}.")
    return None


def run_pipeline():
    """
    Main pipeline. Runs the full scraping process:
    1. Initialize database
    2. Check existing records (resume from where it stopped)
    3. Fetch remaining records in batches
    4. Save each batch to database
    5. Print final summary
    """
    setup_logging()

    logger.info("=" * 55)
    logger.info("PROPULSE AI — FLORIDA PIPELINE STARTING")
    logger.info("=" * 55)

    # Step 1: Prepare database
    database.initialize_db()

    # Step 2: Check how many records already exist
    existing_count = database.get_record_count()
    logger.info(f"Existing records in database: {existing_count:,}")

    # Step 3: Calculate remaining records needed
    remaining = config.TOTAL_GOAL - existing_count

    if remaining <= 0:
        logger.info(f"Goal of {config.TOTAL_GOAL:,} already reached. Nothing to fetch.")
        return

    logger.info(f"Need {remaining:,} more records to reach goal of {config.TOTAL_GOAL:,}.")

    # Step 4: Fetch in batches
    collected_this_run = 0
    offset             = existing_count  # Resume from where we stopped

    while collected_this_run < remaining:

        batch_size = min(config.MAX_BATCH_SIZE, remaining - collected_this_run)

        logger.info(
            f"Fetching | Offset: {offset:,} | "
            f"Batch size: {batch_size} | "
            f"Total so far: {existing_count + collected_this_run:,}/{config.TOTAL_GOAL:,}"
        )

        batch = fetch_batch(offset, batch_size)

        # None = all retries failed
        if batch is None:
            logger.error("Pipeline stopped due to repeated failures. Re-run to resume.")
            break

        # Empty = no more records on server
        if not batch:
            logger.info("No more records available on server.")
            break

        database.save_batch(batch)
        collected_this_run += len(batch)
        offset             += len(batch)

        time.sleep(0.5)  # Small pause — respectful to the server

    # Step 5: Final summary
    final_count = database.get_record_count()

    logger.info("=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Collected this run : {collected_this_run:,}")
    logger.info(f"Total in database  : {final_count:,}")
    logger.info(f"Goal               : {config.TOTAL_GOAL:,}")

    if final_count >= config.TOTAL_GOAL:
        logger.info("Goal reached. Ready for data cleaning phase.")
    else:
        logger.info(f"Still need {config.TOTAL_GOAL - final_count:,} more. Re-run to continue.")

    logger.info("=" * 55)

    stats = database.get_summary_stats()
    logger.info("DATABASE SUMMARY:")
    for key, value in stats.items():
        logger.info(f"  {key:20s}: {value}")


if __name__ == "__main__":
    run_pipeline()