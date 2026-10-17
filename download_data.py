"""
download_data.py
----------------
Downloads the CTU-UHB CTG database as a single ZIP from PhysioNet
using curl (supports resume + automatic retries), then extracts it.

Dataset: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
License: Open Data Commons Attribution License v1.0
Citation: Václav Chudáček et al., BMC Pregnancy and Childbirth 2014 14:16

Run: python download_data.py
"""

import os
import subprocess
import zipfile
import shutil
from pathlib import Path

ZIP_URL    = "https://physionet.org/content/ctu-uhb-ctgdb/get-zip/1.0.0/"
OUTPUT_DIR = Path("data") / "raw"
ZIP_PATH   = Path("data") / "ctu-uhb-ctgdb.zip"


def download_with_curl(url: str, dest: Path) -> None:
    """
    Download via curl with:
      -L  : follow redirects
      -C - : resume partial download automatically
      --retry 10           : retry up to 10 times on transient errors
      --retry-delay 5      : wait 5 s between retries
      --retry-connrefused  : also retry on connection refused
      --connect-timeout 30 : fail fast on bad connections
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe", "-L", "-C", "-",
        "--retry", "10",
        "--retry-delay", "5",
        "--retry-connrefused",
        "--connect-timeout", "30",
        "--progress-bar",
        "-o", str(dest),
        url,
    ]
    print(f"Downloading via curl:\n  {url}\n  -> {dest}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed with exit code {result.returncode}")
    print(f"\nDownload complete → {dest}")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract only .dat and .hea files, flattening any subdirectories."""
    print(f"\nExtracting {zip_path} …")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist()
                   if m.filename.endswith((".dat", ".hea"))]
        print(f"  Found {len(members)} signal files in ZIP")
        for i, member in enumerate(members, 1):
            fname = Path(member.filename).name
            dest  = out_dir / fname
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if i % 200 == 0:
                print(f"  Extracted {i}/{len(members)} …")
    print(f"Extraction complete → {out_dir}/")


def verify(out_dir: Path) -> None:
    dat_files = sorted(out_dir.glob("*.dat"))
    hea_files = sorted(out_dir.glob("*.hea"))
    print(f"\n=== Verification ===")
    print(f"  .dat files : {len(dat_files)}")
    print(f"  .hea files : {len(hea_files)}")
    if len(dat_files) == 552 and len(hea_files) == 552:
        print("  ✓ All 552 recordings present.")
    else:
        print(f"  ⚠ Expected 552 of each. Re-run if counts differ.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / 1e6
        print(f"Partial/complete ZIP found ({size_mb:.1f} MB) — resuming with curl.")
    
    download_with_curl(ZIP_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, OUTPUT_DIR)
    verify(OUTPUT_DIR)

    ZIP_PATH.unlink(missing_ok=True)
    print("ZIP removed.")


if __name__ == "__main__":
    main()

