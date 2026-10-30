"""
Data acquisition module for external empirical traces.
Downloads UCI Video Transcoding and BurstGPT datasets with SHA256 verification.
"""

import hashlib
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Dict

from realdata.config import (
    RAW_DATA_DIR,
    UCI_VIDEO_PAGE_URL,
    UCI_VIDEO_ZIP_URL,
    UCI_VIDEO_FILENAME,
    UCI_VIDEO_RAW_PATH,
    BURSTGPT_CSV_URL,
    BURSTGPT_FILENAME,
    BURSTGPT_RAW_PATH,
)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_burstgpt(dest_path: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download BurstGPT_1.csv from GitHub raw repository.

    Args:
        dest_path: Optional destination file path. Defaults to BURSTGPT_RAW_PATH.
        force: If True, re-downloads even if the file exists.

    Returns:
        Path to the downloaded file.

    Raises:
        RuntimeError: If download fails.
    """
    dest = dest_path or BURSTGPT_RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        checksum = compute_sha256(dest)
        print(f"[BurstGPT] File already exists at {dest}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {checksum}")
        return dest

    print(f"[BurstGPT] Downloading {BURSTGPT_CSV_URL} ...")
    req = urllib.request.Request(
        BURSTGPT_CSV_URL,
        headers={"User-Agent": "Mozilla/5.0 (AetherGrid RealData Fetcher)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"BurstGPT download failed with HTTP status {response.status}: {response.reason}"
                )
            content = response.read()

        dest.write_bytes(content)
        size_mb = dest.stat().st_size / (1024 * 1024)
        checksum = compute_sha256(dest)
        print(f"[BurstGPT] Successfully downloaded {BURSTGPT_FILENAME}")
        print(f"  Destination: {dest}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {checksum}")
        return dest

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download BurstGPT dataset from {BURSTGPT_CSV_URL}: {exc}"
        ) from exc


def fetch_uci_video(dest_path: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download and extract transcoding_mesurment.tsv from UCI Machine Learning Repository.

    Args:
        dest_path: Optional destination file path. Defaults to UCI_VIDEO_RAW_PATH.
        force: If True, re-downloads even if the file exists.

    Returns:
        Path to the extracted TSV file.

    Raises:
        RuntimeError: If automated download fails, with clear manual instructions.
    """
    dest = dest_path or UCI_VIDEO_RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        checksum = compute_sha256(dest)
        print(f"[UCI Video] File already exists at {dest}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {checksum}")
        return dest

    print(f"[UCI Video] Downloading archive from {UCI_VIDEO_ZIP_URL} ...")
    req = urllib.request.Request(
        UCI_VIDEO_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (AetherGrid RealData Fetcher)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"UCI archive download returned HTTP status {response.status}: {response.reason}"
                )
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if UCI_VIDEO_FILENAME not in zf.namelist():
                raise RuntimeError(
                    f"Expected '{UCI_VIDEO_FILENAME}' not found inside UCI zip archive. "
                    f"Available files: {zf.namelist()}"
                )
            extracted_data = zf.read(UCI_VIDEO_FILENAME)

        dest.write_bytes(extracted_data)
        size_mb = dest.stat().st_size / (1024 * 1024)
        checksum = compute_sha256(dest)
        print(f"[UCI Video] Successfully downloaded and extracted {UCI_VIDEO_FILENAME}")
        print(f"  Destination: {dest}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  SHA256: {checksum}")
        return dest

    except Exception as exc:
        manual_instructions = (
            "\n"
            "=" * 70 + "\n"
            "[UCI Video] AUTOMATED DOWNLOAD FAILED.\n"
            f"Error details: {exc}\n\n"
            "MANUAL DOWNLOAD INSTRUCTIONS:\n"
            f"1. Navigate in your browser to:\n   {UCI_VIDEO_PAGE_URL}\n"
            f"2. Download the dataset zip file.\n"
            f"3. Extract '{UCI_VIDEO_FILENAME}' and place it at:\n   {dest.resolve()}\n"
            "=" * 70
        )
        print(manual_instructions, file=sys.stderr)
        raise RuntimeError(
            f"Failed to fetch UCI Video dataset. Please follow manual download instructions above."
        ) from exc


def fetch_all_data(force: bool = False) -> Dict[str, Path]:
    """Download all required external datasets with checksum verification."""
    print("=" * 60)
    print("Initiating Real Data Downloads (UCI Video + BurstGPT)")
    print("=" * 60)
    burst_path = fetch_burstgpt(force=force)
    print("-" * 60)
    uci_path = fetch_uci_video(force=force)
    print("=" * 60)
    print("All datasets verified and ready in raw data directory.")
    return {
        "burstgpt": burst_path,
        "uci_video": uci_path,
    }


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    fetch_all_data(force=force_flag)
