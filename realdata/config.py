"""
Configuration and constants for real-data pipelines and trace ingestion.
"""

from pathlib import Path

# Directory paths
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PACKAGE_ROOT / "reports"
CALIBRATION_DIR = PACKAGE_ROOT / "calibration"
RESULTS_DIR = PACKAGE_ROOT / "results"

# Ensure essential directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs
UCI_VIDEO_PAGE_URL = (
    "https://archive.ics.uci.edu/dataset/335/online+video+characteristics+and+transcoding+time+dataset"
)
UCI_VIDEO_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/335/online+video+characteristics+and+transcoding+time+dataset.zip"
)
BURSTGPT_CSV_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
)

# Raw filenames
UCI_VIDEO_FILENAME = "transcoding_mesurment.tsv"
BURSTGPT_FILENAME = "BurstGPT_1.csv"
DATA_PROFILE_REPORT_FILENAME = "data_profile.md"

# File paths
UCI_VIDEO_RAW_PATH = RAW_DATA_DIR / UCI_VIDEO_FILENAME
BURSTGPT_RAW_PATH = RAW_DATA_DIR / BURSTGPT_FILENAME
DATA_PROFILE_REPORT_PATH = REPORTS_DIR / DATA_PROFILE_REPORT_FILENAME
TRACE_SUMMARY_REPORT_PATH = REPORTS_DIR / "trace_summary.md"
VIDEO_CALIBRATION_PATH = CALIBRATION_DIR / "video_calibration.json"
LLM_CALIBRATION_PATH = CALIBRATION_DIR / "llm_calibration.json"
BURSTGPT_WINDOWS_PATH = CALIBRATION_DIR / "burstgpt_windows.json"
HAZARD_CURVES_PATH = CALIBRATION_DIR / "hazard_curves.json"

# Expected schemas for validation
UCI_VIDEO_REQUIRED_COLUMNS = [
    "id",
    "duration",
    "codec",
    "width",
    "height",
    "bitrate",
    "framerate",
    "umem",
    "utime",
]

BURSTGPT_REQUIRED_COLUMNS = [
    "Timestamp",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
]

# Standard percentile cuts for profiling
PERCENTILES = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
PERCENTILE_LABELS = ["p5", "p25", "p50", "p75", "p95", "p99"]
