# ============================================================
# FMP FULL DATASET BUILDER
# ============================================================
#
# This script:
# 1. Pulls ALL requested endpoints
# 2. Pulls data for ALL requested companies
# 3. Saves separate CSVs per endpoint
# 4. Saves ONE combined parquet dataset
# 5. Creates clean folders automatically
# 6. Handles API failures safely
#
# ============================================================

import os
import time
import requests
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

API_KEY = "gBazFCpPCqU8BOOiyGgMnZ0KTNhvRDnm"

BASE_URL = "https://financialmodelingprep.com/stable"

COMPANIES = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "NFLX",
    "AMD",
    "INTC",
    "JPM",
    "V",
    "MA",
    "COST",
    "WMT"
]

ENDPOINTS = [
    "income-statement",
    "balance-sheet-statement",
    "cash-flow-statement",
    "ratios",
    "ratios-ttm",
    "key-metrics",
    "key-metrics-ttm",
    "enterprise-values",
    "quote",
    "discounted-cash-flow"
]

# ============================================================
# OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR = "fmp_dataset"

CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

PARQUET_DIR = os.path.join(OUTPUT_DIR, "parquet")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

# ============================================================
# REQUEST SETTINGS
# ============================================================

REQUEST_DELAY = 1  # seconds between requests

TIMEOUT = 30

# ============================================================
# FETCH FUNCTION
# ============================================================

def fetch_endpoint(symbol, endpoint):

    url = (
        f"{BASE_URL}/{endpoint}"
        f"?symbol={symbol}&apikey={API_KEY}"
    )

    print(f"\nFetching {endpoint} for {symbol}")

    try:

        response = requests.get(url, timeout=TIMEOUT)

        if response.status_code != 200:

            print(f"FAILED [{response.status_code}]")
            return None

        data = response.json()

        if not data:

            print("No data returned")
            return None

        # Convert to dataframe
        df = pd.DataFrame(data)

        # Add metadata
        df["symbol"] = symbol
        df["endpoint"] = endpoint

        print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        return df

    except Exception as e:

        print(f"ERROR: {e}")

        return None

# ============================================================
# MAIN STORAGE
# ============================================================

all_dataframes = {}

# ============================================================
# DOWNLOAD ALL DATA
# ============================================================

for endpoint in ENDPOINTS:

    print("\n")
    print("=" * 80)
    print(f"PROCESSING ENDPOINT: {endpoint}")
    print("=" * 80)

    endpoint_frames = []

    for symbol in COMPANIES:

        df = fetch_endpoint(symbol, endpoint)

        if df is not None:

            endpoint_frames.append(df)

        # Avoid rate limits
        time.sleep(REQUEST_DELAY)

    # ========================================================
    # COMBINE ALL COMPANIES FOR THIS ENDPOINT
    # ========================================================

    if endpoint_frames:

        combined_df = pd.concat(endpoint_frames, ignore_index=True)

        all_dataframes[endpoint] = combined_df

        # ====================================================
        # SAVE CSV
        # ====================================================

        csv_path = os.path.join(
            CSV_DIR,
            f"{endpoint}.csv"
        )

        combined_df.to_csv(csv_path, index=False)

        print(f"\nSaved CSV: {csv_path}")

        # ====================================================
        # SAVE PARQUET
        # ====================================================

        parquet_path = os.path.join(
            PARQUET_DIR,
            f"{endpoint}.parquet"
        )

        combined_df.to_parquet(
            parquet_path,
            index=False
        )

        print(f"Saved Parquet: {parquet_path}")

# ============================================================
# CREATE MASTER DATASET
# ============================================================

print("\n")
print("=" * 80)
print("CREATING MASTER DATASET")
print("=" * 80)

master_frames = []

for endpoint, df in all_dataframes.items():

    temp_df = df.copy()

    temp_df["source_endpoint"] = endpoint

    master_frames.append(temp_df)

if master_frames:

    master_df = pd.concat(
        master_frames,
        ignore_index=True,
        sort=False
    )

    # ========================================================
    # SAVE MASTER CSV
    # ========================================================

    master_csv = os.path.join(
        OUTPUT_DIR,
        "master_dataset.csv"
    )

    master_df.to_csv(master_csv, index=False)

    print(f"Saved master CSV: {master_csv}")

    # ========================================================
    # SAVE MASTER PARQUET
    # ========================================================

    master_parquet = os.path.join(
        OUTPUT_DIR,
        "master_dataset.parquet"
    )

    master_df.to_parquet(
        master_parquet,
        index=False
    )

    print(f"Saved master Parquet: {master_parquet}")

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

print(f"Companies: {len(COMPANIES)}")
print(f"Endpoints: {len(ENDPOINTS)}")

print("\nFiles saved to:")

print(f"CSV Folder: {CSV_DIR}")
print(f"Parquet Folder: {PARQUET_DIR}")

print("\nMaster dataset files:")

print("master_dataset.csv")
print("master_dataset.parquet")

# ============================================================
# OPTIONAL: LOAD DATA LATER
# ============================================================

"""
Example usage later in another project:

import pandas as pd

df = pd.read_parquet("fmp_dataset/master_dataset.parquet")

print(df.head())

"""

# ============================================================
# OPTIONAL: CREATE SQLITE DATABASE
# ============================================================

"""
import sqlite3

conn = sqlite3.connect("fmp_dataset/fmp_data.db")

for endpoint, df in all_dataframes.items():

    table_name = endpoint.replace("-", "_")

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

conn.close()

print("SQLite database created!")

"""