# project_config.py - Central Configuration File
import logging

# ─────────────────────────────────────────────
# 1. API ENDPOINT
# ─────────────────────────────────────────────
BASE_URL = (
    "https://services9.arcgis.com/Gh9awoU677aKree0/arcgis/rest/services"
    "/Florida_Statewide_Cadastral/FeatureServer/0/query"
)

# ─────────────────────────────────────────────
# 2. SCRAPER SETTINGS
# ─────────────────────────────────────────────
TOTAL_GOAL      = 8000
MAX_BATCH_SIZE  = 2000
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3
RETRY_DELAY     = 5

# ─────────────────────────────────────────────
# 3. DATA FIELDS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 4. REQUEST PARAMETERS
# ─────────────────────────────────────────────
QUERY_PARAMS = {
    "where"         : "1=1",
    "outFields"     : ",".join(OUT_FIELDS),
    "returnGeometry": "false",
    "f"             : "pjson",
    "orderByFields" : "OBJECTID ASC"
}

# ─────────────────────────────────────────────
# 5. DATABASE SETTINGS
# ─────────────────────────────────────────────
DB_NAME = "florida_leads.db"

# ─────────────────────────────────────────────
# 6. LOGGING SETTINGS
# ─────────────────────────────────────────────
LOG_FILE  = "pipeline.log"
LOG_LEVEL = logging.INFO