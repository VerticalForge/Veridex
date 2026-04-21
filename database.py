# database.py - SQLite Database Handler
import sqlite3
import pandas as pd
import logging
import project_config as config

logger = logging.getLogger(__name__)

# Fields that should be stored as TEXT
# Add any new text/code fields here if you expand later
TEXT_FIELDS = {
    "OWN_NAME", "OWN_ADDR1", "OWN_ADDR2", "OWN_CITY",
    "OWN_STATE", "OWN_STATE_",
    "PHY_ADDR1", "PHY_ADDR2", "PHY_CITY",
    "DOR_UC", "QUAL_CD1", "QUAL_CD2",
    "VI_CD1", "VI_CD2",
    "NBRHD_CD", "MKT_AR",
    "FILE_T", "APP_STAT", "S_LEGAL",
    "PARCEL_ID", "ALT_KEY",
}


def initialize_db():
    """Creates the leads table and indexes if they don't exist."""
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()

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

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


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
    """Returns total number of records in the database."""
    conn   = sqlite3.connect(config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM leads")
    count  = cursor.fetchone()[0]
    conn.close()
    return count


def load_all_leads() -> pd.DataFrame:
    """Loads all records into a Pandas DataFrame for analysis and ML."""
    conn = sqlite3.connect(config.DB_NAME)
    df   = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()
    logger.info(f"Loaded {len(df):,} records from database.")
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

    # New field coverage checks
    cursor.execute("SELECT COUNT(*) FROM leads WHERE TOT_LVG_AR > 0")
    stats["has_living_area"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM leads WHERE SALE_PRC2 > 0")
    stats["has_second_sale"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM leads WHERE OWN_STATE_ != 'FL' AND OWN_STATE_ IS NOT NULL AND OWN_STATE_ != ''")
    stats["out_of_state_owners"] = cursor.fetchone()[0]

    conn.close()
    return stats