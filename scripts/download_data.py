"""
Download AIMS Logger Data + CRW Satellite SST
==============================================
Automates data acquisition for the PINN reef thermal reconstruction.

Downloads:
  1. AIMS temperature logger CSVs from the AIMS Data Platform API
  2. CRW SST daily grids from NOAA CoastWatch ERDDAP

Creates the data/ALZ/ directory structure expected by src/data_loader_v2.py.

Usage:
    python scripts/download_data.py              # download all reefs
    python scripts/download_data.py --reef davies_reef   # single reef
    python scripts/download_data.py --sst-only   # satellite SST only
"""

import os
import sys
import argparse
import time

import requests
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ALZ_DIR = os.path.join(BASE_DIR, "data", "ALZ")
SST_DIR = os.path.join(BASE_DIR, "data", "sst")

# ═══════════════════════════════════════════════════════════════════════
# Reef configurations
# ═══════════════════════════════════════════════════════════════════════
REEF_CONFIGS = {
    "davies_reef": {
        "site_name": "Davies Reef",
        "lat": -18.8316,
        "lon": 147.6345,
        "date_from": "2011-01-01",
        "date_to": "2015-01-01",
    },
    "myrmidon_reef": {
        "site_name": "Myrmidon Reef",
        "lat": -18.2660,
        "lon": 147.3830,
        "date_from": "2020-01-01",
        "date_to": "2025-01-01",
    },
    "rib_reef": {
        "site_name": "Rib Reef",
        "lat": -18.473,
        "lon": 146.8774,
        "date_from": "2018-01-01",
        "date_to": "2024-01-01",
    },
    "kelso_reef": {
        "site_name": "Kelso Reef",
        "lat": -18.4317,
        "lon": 146.9883,
        "date_from": "1998-01-01",
        "date_to": "2002-01-01",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# AIMS Data Download
# ═══════════════════════════════════════════════════════════════════════
AIMS_API_BASE = "https://data.aims.gov.au/api/v1/data"

def download_aims_reef(reef_key, config, skip_existing=True):
    """Download AIMS temperature logger data for a single reef.

    Uses the AIMS Time Series Explorer API to fetch all available
    temperature deployments for the reef site.

    Note: The AIMS API may require manual download for some datasets.
    If the automated download fails, visit:
        https://apps.aims.gov.au/ts-explorer/
    Search for the reef name and download CSV files manually into data/ALZ/.
    """
    os.makedirs(ALZ_DIR, exist_ok=True)
    site_name = config["site_name"]
    print(f"\n{'='*60}")
    print(f"Downloading AIMS data: {site_name}")
    print(f"{'='*60}")

    # Check for existing files
    existing = [f for f in os.listdir(ALZ_DIR)
                if site_name.split()[0] in f and f.endswith(".csv")]
    if existing and skip_existing:
        print(f"  Found {len(existing)} existing CSV files, skipping.")
        print(f"  Use --force to re-download.")
        return len(existing)

    # Try the AIMS API
    # The AIMS data platform provides a REST API for accessing monitoring data
    try:
        # Search for temperature monitoring data at the reef
        search_url = f"{AIMS_API_BASE}/search"
        params = {
            "site": site_name,
            "parameter": "Water Temperature",
            "from": config["date_from"],
            "to": config["date_to"],
            "format": "csv",
        }
        print(f"  Querying AIMS API for {site_name} temperature data...")
        r = requests.get(search_url, params=params, timeout=60)

        if r.status_code == 200:
            # Parse and save the data
            out_path = os.path.join(ALZ_DIR, f"{site_name.replace(' ', '_')}_temperature.csv")
            with open(out_path, "w") as f:
                f.write(r.text)
            print(f"  Saved: {out_path}")
            return 1
        else:
            print(f"  AIMS API returned status {r.status_code}")
            raise requests.exceptions.HTTPError(f"Status {r.status_code}")

    except (requests.exceptions.RequestException, Exception) as e:
        print(f"  Automated download failed: {e}")
        print(f"\n  *** MANUAL DOWNLOAD REQUIRED ***")
        print(f"  The AIMS Time Series Explorer requires interactive access.")
        print(f"  Please download data manually:")
        print(f"    1. Visit: https://apps.aims.gov.au/ts-explorer/")
        print(f"    2. Search for: {site_name}")
        print(f"    3. Select 'Water Temperature' deployments")
        print(f"    4. Date range: {config['date_from']} to {config['date_to']}")
        print(f"    5. Download all CSV files to: {ALZ_DIR}/")
        print()
        return 0


# ═══════════════════════════════════════════════════════════════════════
# CRW SST Download
# ═══════════════════════════════════════════════════════════════════════
def download_crw_sst(reef_key, config, skip_existing=True):
    """Download CRW SST from NOAA ERDDAP for a single reef."""
    os.makedirs(SST_DIR, exist_ok=True)
    out_path = os.path.join(SST_DIR, f"{reef_key}_sst.csv")

    if os.path.exists(out_path) and skip_existing:
        print(f"  SST already exists: {out_path}, skipping.")
        return True

    lat, lon = config["lat"], config["lon"]
    date_from, date_to = config["date_from"], config["date_to"]
    bbox = 0.025

    base = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
    query = (
        f"?CRW_SST[({date_from}T12:00:00Z):1:({date_to}T12:00:00Z)]"
        f"[({lat - bbox}):1:({lat + bbox})]"
        f"[({lon - bbox}):1:({lon + bbox})]"
    )
    url = base + query

    print(f"  Downloading CRW SST for {reef_key}...")
    print(f"  URL: {base}?CRW_SST[...]")

    try:
        r = requests.get(url, timeout=300, allow_redirects=True)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        header = lines[0]
        data_lines = [header] + lines[2:]  # skip units row

        from io import StringIO
        sst = pd.read_csv(StringIO("\n".join(data_lines)))
        sst["CRW_SST"] = pd.to_numeric(sst["CRW_SST"], errors="coerce")
        sst = sst.dropna(subset=["CRW_SST"])

        # Average across grid cells per day
        sst_daily = sst.groupby("time").agg(CRW_SST=("CRW_SST", "mean")).reset_index()
        sst_daily.to_csv(out_path, index=False)

        print(f"  Saved: {out_path} ({len(sst_daily)} days)")
        return True

    except Exception as e:
        print(f"  SST download failed: {e}")
        print(f"  You can retry later or download manually from ERDDAP.")
        return False


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Download AIMS logger + CRW SST data for PINN experiments"
    )
    parser.add_argument("--reef", type=str, default=None,
                        choices=list(REEF_CONFIGS.keys()),
                        help="Download data for a single reef (default: all)")
    parser.add_argument("--sst-only", action="store_true",
                        help="Download satellite SST only (skip AIMS loggers)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args()

    skip = not args.force
    reefs = [args.reef] if args.reef else list(REEF_CONFIGS.keys())

    print("PINN Reef Thermal — Data Download")
    print("=" * 60)
    print(f"Reefs: {', '.join(reefs)}")
    print(f"AIMS logger dir: {ALZ_DIR}")
    print(f"SST dir: {SST_DIR}")
    print()

    # Download AIMS logger data
    if not args.sst_only:
        print("\n--- AIMS Temperature Logger Data ---")
        for reef_key in reefs:
            download_aims_reef(reef_key, REEF_CONFIGS[reef_key], skip_existing=skip)
            time.sleep(1)  # be polite to the API

    # Download CRW SST
    print("\n--- CRW Satellite SST (NOAA ERDDAP) ---")
    for reef_key in reefs:
        download_crw_sst(reef_key, REEF_CONFIGS[reef_key], skip_existing=skip)
        time.sleep(1)

    print("\n" + "=" * 60)
    print("Download complete!")
    print()
    print("If AIMS logger download required manual steps, place CSV files in:")
    print(f"  {ALZ_DIR}/")
    print()
    print("Expected file naming: '<Reef Name> ... .csv'")
    print("  e.g., 'Davies Reef Weather Station Temperature@5m.csv'")
    print()
    print("Then run experiments with:")
    print("  python scripts/run_experiments.py")


if __name__ == "__main__":
    main()
