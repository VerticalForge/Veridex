# Veridex

**Real estate lead scoring pipeline that processes 109,000+ property records and surfaces high-value investment leads — fully automated, end-to-end.**

---

## The Problem

Real estate investors waste hours manually sifting through county property records trying to find undervalued or high-potential properties. The data is messy, scattered across multiple relational files, and impossible to act on without serious cleanup and analysis. Most investors either pay for expensive lead services or rely on gut instinct.

## What Veridex Does

Veridex ingests raw county property data, cleans and engineers features across 109,361 records, scores every property using machine learning, and outputs a ranked list of hot leads — ready for investor action. No manual filtering, no guesswork.

## How It Works

```
Raw CAMA Data (6 relational files, 109K+ records)
        │
        ▼
   Data Ingestion (load_cama.py)
        │
        ▼
   8-Step Cleaning Pipeline (cleaner.py)
        │
        ▼
   14-Step Feature Engineering (features.py)
   → 11 engineered features output
        │
        ▼
   ML Scoring Model (ml_model.py)
   → XGBoost / GradientBoostingClassifier
        │
        ▼
   Scored & Ranked Leads (database.py)
   → SQLite with audit trail
        │
        ▼
   Veridex Dashboard (Power BI)
   → KPI cards, score distribution, ranked leads table
```

### Pipeline Modules

| Module | What It Does |
|---|---|
| `project_config.py` | Three mode controllers — PIPELINE_MODE, RUN_MODE, DEMO_MODE |
| `database.py` | Manages pipeline_log and scored_leads tables, audit trail, reset utilities |
| `load_cama.py` | Ingests Alachua County CAMA data across 6 relational files |
| `cleaner.py` | 8-step cleaning pipeline with residential code filtering and ZIP-to-city mapping |
| `features.py` | 14-step feature engineering producing 11 ML-ready features with weak supervision labels |
| `ml_model.py` | XGBoost and GradientBoostingClassifier scoring model |
| `main.py` | Full orchestrator with resume logic and run summary |

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SQLite · Power BI

## Results

| Metric | Value |
|---|---|
| Total records ingested | 109,361 |
| Clean records after pipeline | 4,837 |
| Engineered features | 11 |
| Hot leads identified (weak supervision) | 2,024 |

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/VerticalForge/Veridex.git
cd Veridex

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Add your credentials to .env

# Run the pipeline
python main.py
```

## Data Source

Alachua County, Florida — CAMA (Computer Assisted Mass Appraisal) public data download. 6 relational property files covering residential, commercial, and land records.

---

**Built by [Vertical Forge](https://github.com/VerticalForge)** — AI systems agency specializing in ML pipelines, AI agents, and agentic automation.

📧 a.ahmad67937@gmail.com
