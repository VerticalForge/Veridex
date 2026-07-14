# load_cama.py - Alachua County CAMA Data Loader
# ============================================================
# VERSION : v1.0
#
# PURPOSE:
#   Replaces the ArcGIS API scraper for initial data loading.
#   Reads ACPA_CAMAData.zip from Alachua County Property Appraiser.
#   Joins 5 source files into a single leads table record format.
#   Behaves identically to the API scraper — same batch size,
#   same resume logic, same TOTAL_GOAL control, same DB output.
#
# SOURCE:
#   https://s3.amazonaws.com/acpa.cama/ACPA_CAMAData.zip
#   Updated annually every March by Alachua County.
#
# USAGE:
#   Place ACPA_CAMAData.zip in your project folder.
#   Set TOTAL_GOAL in project_config.py (e.g. 8000).
#   Run: python load_cama.py
#
# HOW IT MIRRORS THE API SCRAPER:
#   API scraper  → fetches TOTAL_GOAL records in MAX_BATCH_SIZE batches
#   CAMA loader  → reads  TOTAL_GOAL records in MAX_BATCH_SIZE batches
#   Both         → save to leads table via database.save_batch()
#   Both         → resume from existing DB count if interrupted
#   Both         → DEMO_MODE wipes first then loads fresh
#
# INCREMENTAL EXPANSION:
#   First run   → TOTAL_GOAL = 8000  → loads records 0-7999
#   Second run  → TOTAL_GOAL = 16000 → loads records 8000-15999
#   Raise TOTAL_GOAL in config, run again — same as API behaviour.
#
# AFTER RUNNING:
#   Set RUN_MODE = "DRY_RUN" in project_config.py
#   Run: python main.py
#
# FILES INSIDE ZIP (joined on Parcel key):
#   Property.txt     → DOR_UC, PHY_CITY, NBRHD_CD
#   Owners.txt       → OWN_NAME, OWN_ADDR1, OWN_STATE_, PHY_ZIPCD
#   HistoryRE.txt    → JV, LND_VAL, JV_HMSTD, JV_CHNG
#   Land.txt         → LND_SQFOOT
#   Improvements.txt → ACT_YR_BLT, EFF_YR_BLT, TOT_LVG_AR, NO_BULDNG
#   Sales.txt        → SALE_PRC1, SALE_YR1, QUAL_CD1, SALE_PRC2, SALE_YR2
# ============================================================

import zipfile
import logging
import os
import pandas as pd
import numpy as np
import project_config as config
import database

logging.basicConfig(
    level   = config.LOG_LEVEL,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

ZIP_PATH = "ACPA_CAMAData.zip"


# ============================================================
# STEP 1 — LOAD ALL SOURCE FILES FROM ZIP
# ============================================================

# FIXED:
def _read_file(z: zipfile.ZipFile, filename: str, usecols: list) -> pd.DataFrame:
    logger.info(f"  Reading {filename}...")
    with z.open(filename) as f:
        df = pd.read_csv(
            f,
            sep        = '\t',
            usecols    = usecols,
            dtype      = str,
            low_memory = False
        )
    logger.info(f"  {filename}: {len(df):,} rows")
    return df


def load_source_files(zip_path: str) -> dict:
    """Opens ZIP and reads all required source files."""
    logger.info(f"Opening {zip_path}...")
    z = zipfile.ZipFile(zip_path)

    files = {}

    files['property'] = _read_file(z, 'Property.txt', [
        'Parcel', 'City_Desc', 'NBHD_Code', 'Prop_Use_Code'
    ])
    files['owners'] = _read_file(z, 'Owners.txt', [
        'Parcel', 'Owner_Mail_Name', 'Owner_Mail_Addr1',
        'Owner_Mail_State', 'Owner_Mail_Zip'
    ])
    files['history'] = _read_file(z, 'HistoryRE.txt', [
        'Parcel', 'Hist_Tax_Year', 'Just_Value',
        'Land_Value', 'County_Exempt_Amount'
    ])
    files['land'] = _read_file(z, 'Land.txt', [
        'Parcel', 'Land_SqFt'
    ])
    files['improvements'] = _read_file(z, 'Improvements.txt', [
        'Parcel', 'Imprv_Type', 'Actual_YrBlt',
        'Effective_YrBlt', 'Heated_SquareFeet', 'Bldg_Num'
    ])
    files['sales'] = _read_file(z, 'Sales.txt', [
        'Parcel', 'Sale_Line_Num', 'Sale_Date',
        'Sale_Price', 'DOR_Qual_Code'
    ])

    z.close()
    return files


# ============================================================
# STEP 2 — PREPARE EACH TABLE
# ============================================================

def _prepare_property(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per parcel.
    DOR_UC normalization: CAMA uses 5-digit codes (e.g. '00100').
    Pipeline expects 3-digit (e.g. '001').
    Method: take first 3 characters — '00100'[:3] = '001'.
    This matches cleaner.py RESIDENTIAL_CODES exactly.
    """
    df = df.drop_duplicates(subset='Parcel', keep='first').copy()

    df['DOR_UC']   = df['Prop_Use_Code'].astype(str).str.strip().str[:3]
    df['PHY_CITY'] = df['City_Desc'].astype(str).str.strip().str.upper()
    df['NBRHD_CD'] = df['NBHD_Code'].astype(str).str.strip()

    return df[['Parcel', 'DOR_UC', 'PHY_CITY', 'NBRHD_CD']]


def _prepare_owners(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per parcel. First owner record.
    ZIP: keep 5-digit only, strip ZIP+4 extension.
    """
    df = df.drop_duplicates(subset='Parcel', keep='first').copy()

    df['OWN_NAME']   = df['Owner_Mail_Name'].astype(str).str.strip()
    df['OWN_ADDR1']  = df['Owner_Mail_Addr1'].astype(str).str.strip()
    df['OWN_STATE_'] = df['Owner_Mail_State'].astype(str).str.strip()
    df['PHY_ZIPCD']  = pd.to_numeric(
        df['Owner_Mail_Zip'].astype(str).str[:5],
        errors='coerce'
    ).fillna(0)

    return df[['Parcel', 'OWN_NAME', 'OWN_ADDR1', 'OWN_STATE_', 'PHY_ZIPCD']]


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Most recent tax year per parcel.
    JV_CHNG not in CAMA — set to 0.
    """
    df['Hist_Tax_Year'] = pd.to_numeric(df['Hist_Tax_Year'], errors='coerce')
    df = df.sort_values('Hist_Tax_Year', ascending=False)
    df = df.drop_duplicates(subset='Parcel', keep='first').copy()

    df['JV']       = pd.to_numeric(df['Just_Value'],           errors='coerce').fillna(0)
    df['LND_VAL']  = pd.to_numeric(df['Land_Value'],           errors='coerce').fillna(0)
    df['JV_HMSTD'] = pd.to_numeric(df['County_Exempt_Amount'], errors='coerce').fillna(0)
    df['JV_CHNG']  = 0

    return df[['Parcel', 'JV', 'LND_VAL', 'JV_HMSTD', 'JV_CHNG']]


def _prepare_land(df: pd.DataFrame) -> pd.DataFrame:
    """Sum land sqft across all land lines per parcel."""
    df['Land_SqFt'] = pd.to_numeric(df['Land_SqFt'], errors='coerce').fillna(0)
    agg = df.groupby('Parcel')['Land_SqFt'].sum().reset_index()
    agg = agg.rename(columns={'Land_SqFt': 'LND_SQFOOT'})
    return agg


def _prepare_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Building data per parcel.
    Excludes SOHM/NSOHM rows (Special Owner History — not real buildings).
    Primary building = row with max Heated_SquareFeet.
    NO_BULDNG = count of distinct buildings.
    NO_RES_UNT = same as NO_BULDNG for residential.
    """
    df = df[~df['Imprv_Type'].astype(str).str.strip().isin(
        ['SOHM', 'NSOHM']
    )].copy()

    df['Heated_SquareFeet'] = pd.to_numeric(df['Heated_SquareFeet'], errors='coerce').fillna(0)
    df['Actual_YrBlt']      = pd.to_numeric(df['Actual_YrBlt'],      errors='coerce').fillna(0)
    df['Effective_YrBlt']   = pd.to_numeric(df['Effective_YrBlt'],   errors='coerce').fillna(0)

    bldg_count = (
        df.groupby('Parcel')['Bldg_Num']
        .nunique()
        .reset_index()
        .rename(columns={'Bldg_Num': 'NO_BULDNG'})
    )

    idx     = df.groupby('Parcel')['Heated_SquareFeet'].idxmax()
    primary = df.loc[idx][
        ['Parcel', 'Actual_YrBlt', 'Effective_YrBlt', 'Heated_SquareFeet']
    ].copy()
    primary = primary.rename(columns={
        'Actual_YrBlt'      : 'ACT_YR_BLT',
        'Effective_YrBlt'   : 'EFF_YR_BLT',
        'Heated_SquareFeet' : 'TOT_LVG_AR',
    })

    result = primary.merge(bldg_count, on='Parcel', how='left')
    result['NO_BULDNG']  = result['NO_BULDNG'].fillna(1).astype(int)
    result['NO_RES_UNT'] = result['NO_BULDNG']

    return result[['Parcel', 'ACT_YR_BLT', 'EFF_YR_BLT',
                   'TOT_LVG_AR', 'NO_BULDNG', 'NO_RES_UNT']]


def _prepare_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Most recent sale (line 0) → SALE_PRC1, SALE_YR1, QUAL_CD1
    Previous sale (line 1)    → SALE_PRC2, SALE_YR2

    CAMA line numbering: 0 = most recent, 1 = second, 2+ = older.
    QUAL_CD1 stores full CAMA description e.g. 'U-UNQUALIFIED'.
    features.py Step 10 uses startswith('U') — matches correctly.
    Parcels with no sale get zeros.
    """
    df['Sale_Line_Num'] = pd.to_numeric(df['Sale_Line_Num'], errors='coerce')
    df['Sale_Price']    = pd.to_numeric(df['Sale_Price'],    errors='coerce').fillna(0)
    df['Sale_Year']     = pd.to_numeric(
        df['Sale_Date'].astype(str).str[:4], errors='coerce'
    ).fillna(0)

    sale1 = df[df['Sale_Line_Num'] == 0][[
        'Parcel', 'Sale_Price', 'Sale_Year', 'DOR_Qual_Code'
    ]].drop_duplicates('Parcel', keep='first').copy()
    sale1.columns = ['Parcel', 'SALE_PRC1', 'SALE_YR1', 'QUAL_CD1']

    sale2 = df[df['Sale_Line_Num'] == 1][[
        'Parcel', 'Sale_Price', 'Sale_Year'
    ]].drop_duplicates('Parcel', keep='first').copy()
    sale2.columns = ['Parcel', 'SALE_PRC2', 'SALE_YR2']

    result = sale1.merge(sale2, on='Parcel', how='left')
    result['SALE_PRC2'] = pd.to_numeric(result['SALE_PRC2'], errors='coerce').fillna(0)
    result['SALE_YR2']  = pd.to_numeric(result['SALE_YR2'],  errors='coerce').fillna(0)
    result['QUAL_CD1']  = result['QUAL_CD1'].fillna('')

    return result


# ============================================================
# STEP 3 — JOIN ALL TABLES
# ============================================================

def build_full_dataset(files: dict) -> pd.DataFrame:
    """
    Joins all prepared tables on Parcel.
    Returns full DataFrame with all OUT_FIELDS columns.
    OBJECTID is assigned later based on DB resume offset.
    """
    logger.info("Preparing individual tables...")
    prop  = _prepare_property(files['property'])
    own   = _prepare_owners(files['owners'])
    hist  = _prepare_history(files['history'])
    land  = _prepare_land(files['land'])
    imprv = _prepare_improvements(files['improvements'])
    sales = _prepare_sales(files['sales'])

    logger.info("Joining on Parcel...")
    df = prop.merge(own,   on='Parcel', how='left')
    df = df.merge(hist,    on='Parcel', how='left')
    df = df.merge(land,    on='Parcel', how='left')
    df = df.merge(imprv,   on='Parcel', how='left')
    df = df.merge(sales,   on='Parcel', how='left')

    # PHY_ADDR1 — no physical address in CAMA
    # OWN_ADDR1 used as proxy
    # PHY_ADDR1 is not used in any ML feature (features.py confirmed)
    df['PHY_ADDR1'] = df['OWN_ADDR1']

    # Fill numeric nulls
    numeric_cols = [
        'JV', 'JV_HMSTD', 'JV_CHNG', 'LND_VAL', 'LND_SQFOOT',
        'ACT_YR_BLT', 'EFF_YR_BLT', 'TOT_LVG_AR',
        'NO_BULDNG', 'NO_RES_UNT',
        'SALE_PRC1', 'SALE_PRC2', 'SALE_YR1', 'SALE_YR2', 'PHY_ZIPCD'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill text nulls
    text_cols = [
        'OWN_NAME', 'OWN_ADDR1', 'OWN_STATE_',
        'PHY_ADDR1', 'PHY_CITY', 'DOR_UC', 'QUAL_CD1', 'NBRHD_CD'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    df = df.reset_index(drop=True)
    logger.info(f"Full dataset: {len(df):,} records | {len(df.columns)} columns")
    return df


# ============================================================
# STEP 4 — BATCH SAVE MIRRORING API SCRAPER BEHAVIOUR
# ============================================================

def run_loader():
    """
    Main loader. Mirrors API scraper behaviour exactly.
    Uses TOTAL_GOAL and MAX_BATCH_SIZE from project_config.
    Resumes from existing DB count if interrupted.
    DEMO_MODE = True wipes all tables first.
    """
    logger.info("=" * 55)
    logger.info("VERIDEX — CAMA DATA LOADER")
    logger.info("Alachua County Property Appraiser")
    logger.info("=" * 55)
    logger.info(
        f"TOTAL_GOAL : {config.TOTAL_GOAL:,} | "
        f"BATCH_SIZE : {config.MAX_BATCH_SIZE:,} | "
        f"DEMO_MODE  : {config.DEMO_MODE}"
    )
    logger.info("=" * 55)

    # ── Check ZIP exists ──────────────────────────────────────
    if not os.path.exists(ZIP_PATH):
        logger.error(
            f"ZIP not found: {ZIP_PATH}\n"
            f"Download: https://s3.amazonaws.com/acpa.cama/ACPA_CAMAData.zip\n"
            f"Place in project folder and run again."
        )
        return

    # ── Initialize database ───────────────────────────────────
    database.initialize_db()

    # ── DEMO_MODE wipe ────────────────────────────────────────
    if config.DEMO_MODE:
        database.reset_all_tables()
        logger.info("DEMO_MODE: All tables cleared. Loading fresh.")

    # ── Resume logic — identical to API scraper ───────────────
    existing_count = database.get_record_count()
    logger.info(f"Existing records in DB : {existing_count:,}")

    remaining = config.TOTAL_GOAL - existing_count

    if remaining <= 0:
        logger.info(
            f"TOTAL_GOAL of {config.TOTAL_GOAL:,} already reached.\n"
            f"To load more: raise TOTAL_GOAL in project_config.py and run again."
        )
        return

    logger.info(f"Records to load : {remaining:,}")

    # ── Load source files from ZIP ────────────────────────────
    files = load_source_files(ZIP_PATH)

    # ── Build full joined dataset ─────────────────────────────
    full_df = build_full_dataset(files)

    # ── Verify dataset is large enough ───────────────────────
    available = len(full_df) - existing_count
    if available <= 0:
        logger.error(
            f"No more records available in ZIP after offset {existing_count:,}.\n"
            f"ZIP has {len(full_df):,} total records. Already loaded all of them."
        )
        return

    if available < remaining:
        logger.warning(
            f"ZIP only has {available:,} more records available. "
            f"Will load {available:,} instead of {remaining:,}."
        )
        remaining = available

    # ── Slice from resume offset ──────────────────────────────
    slice_df = full_df.iloc[existing_count: existing_count + remaining].copy()
    slice_df = slice_df.reset_index(drop=True)

    # Assign OBJECTID continuing from existing count
    slice_df['OBJECTID'] = range(
        existing_count + 1,
        existing_count + len(slice_df) + 1
    )

    # ── Select exact OUT_FIELDS in correct order ──────────────
    missing = [f for f in config.OUT_FIELDS if f not in slice_df.columns]
    if missing:
        logger.error(f"Missing OUT_FIELDS: {missing}")
        return

    slice_df = slice_df[config.OUT_FIELDS].copy()

    # ── Save in batches ───────────────────────────────────────
    collected = 0
    total     = len(slice_df)
    records   = slice_df.to_dict(orient='records')

    while collected < total:
        batch_size = min(config.MAX_BATCH_SIZE, total - collected)
        batch      = records[collected: collected + batch_size]

        logger.info(
            f"Saving | Offset: {existing_count + collected:,} | "
            f"Batch: {batch_size} | "
            f"Progress: {existing_count + collected:,}/{config.TOTAL_GOAL:,}"
        )

        database.save_batch(batch)
        collected += batch_size

    # ── Final summary ─────────────────────────────────────────
    final_count = database.get_record_count()

    logger.info("=" * 55)
    logger.info("LOAD COMPLETE")
    logger.info(f"Loaded this run : {collected:,}")
    logger.info(f"Total in DB     : {final_count:,}")
    logger.info(f"TOTAL_GOAL      : {config.TOTAL_GOAL:,}")

    if final_count >= config.TOTAL_GOAL:
        logger.info("Goal reached. Ready for pipeline.")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("  Set in project_config.py:")
        logger.info("    RUN_MODE  = 'DRY_RUN'")
        logger.info("    DEMO_MODE = False")
        logger.info("  Then run: python main.py")
    else:
        logger.info(
            f"Still need {config.TOTAL_GOAL - final_count:,} more. "
            f"Re-run to continue."
        )
    logger.info("=" * 55)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_loader()
