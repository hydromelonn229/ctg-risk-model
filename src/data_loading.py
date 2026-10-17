"""
src/data_loading.py
-------------------
Load CTU-UHB CTG recordings and parse clinical metadata from .hea headers.

Dataset: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/
Signals: FHR (channel 0) + UC (channel 1), sampled at 4 Hz
"""

import re
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data") / "raw"
FS = 4.0  # Hz – confirmed from PhysioNet dataset page


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

# Clinical fields buried in .hea comment lines
# Public metadata names and their header labels. The downloaded PhysioNet
# headers use labels such as "Age" and "Weight(g)", not the names used by the
# project table.
HEADER_FIELDS = {
    "pH":        float,
    "pCO2":      float,
    "pO2":       float,
    "BE":        float,       # base excess
    "BDecf":     float,       # base deficit in extracellular fluid
    "Apgar1":    float,
    "Apgar5":    float,
    "age":       float,       # maternal age
    "parity":    float,
    "gravidity": float,
    "sex":       str,         # fetal sex
    "weight":    float,       # birth weight (g)
    "Meconium":  str,
    "Delivery":  str,         # vaginal / operative vaginal / CS
    "Presentation": str,
    "dbID":      str,         # record identifier
}

HEADER_LABELS = {
    "pH": "pH",
    "pCO2": "pCO2",
    "pO2": "pO2",
    "BE": "BE",
    "BDecf": "BDecf",
    "Apgar1": "Apgar1",
    "Apgar5": "Apgar5",
    "age": "Age",
    "parity": "Parity",
    "gravidity": "Gravidity",
    "sex": "Sex",
    "weight": "Weight(g)",
    "Meconium": "Meconium",
    "Delivery": "Deliv. type",
    "Presentation": "Presentation",
    "dbID": "dbID",
}


def resolve_data_dir(raw_dir: Path = RAW_DIR) -> Path:
    """Return the directory containing the dataset's WFDB files.

    Supports both a flattened ``data/raw`` download and the directory layout
    produced by PhysioNet's ZIP archive.
    """
    raw_dir = Path(raw_dir)
    if any(raw_dir.glob("*.hea")):
        return raw_dir

    candidates = sorted(raw_dir.rglob("*.hea"))
    if candidates:
        return candidates[0].parent

    return raw_dir


def parse_header_comments(record_name: str, raw_dir: Path = RAW_DIR) -> dict:
    """
    Read the .hea file for a record and extract clinical metadata from
    comment lines (lines starting with '#').

    Returns a dict with keys matching HEADER_FIELDS where present.
    """
    raw_dir = resolve_data_dir(raw_dir)
    hea_path = raw_dir / f"{record_name}.hea"
    metadata = {"record": record_name}

    with hea_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            # Strip leading '#' and optional whitespace
            content = line.lstrip("#").strip()
            for field, dtype in HEADER_FIELDS.items():
                label = HEADER_LABELS[field]
                # Header labels are case-sensitive, but allow flexible spacing
                # around labels such as "Weight(g)".
                pattern = rf"^{re.escape(label)}\s+(.+)$"
                m = re.match(pattern, content)
                if m:
                    raw_val = m.group(1).strip()
                    try:
                        metadata[field] = dtype(raw_val)
                    except (ValueError, TypeError):
                        metadata[field] = None
                    break

    return metadata


def load_record(record_name: str, raw_dir: Path = RAW_DIR):
    """
    Load a single WFDB record. Returns (record, fhr, uc) where:
      - record : wfdb.Record object
      - fhr    : np.ndarray, fetal heart rate signal (bpm)
      - uc     : np.ndarray, uterine contraction signal
    """
    raw_dir = resolve_data_dir(raw_dir)
    path = str(raw_dir / record_name)
    record = wfdb.rdrecord(path)
    fhr = record.p_signal[:, 0].astype(float)
    uc  = record.p_signal[:, 1].astype(float)
    return record, fhr, uc


def get_record_names(raw_dir: Path = RAW_DIR) -> list[str]:
    """Return sorted list of all record names (without extension)."""
    raw_dir = resolve_data_dir(raw_dir)
    dat_files = sorted(raw_dir.glob("*.dat"))
    return [f.stem for f in dat_files]


# ---------------------------------------------------------------------------
# Build the summary table (one row per recording)
# ---------------------------------------------------------------------------

def build_metadata_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Parse all .hea files and return a DataFrame with one row per recording.
    Columns: record, pH, pCO2, pO2, BE, BDecf, Apgar1, Apgar5, age,
             parity, gravidity, sex, weight, Meconium, Delivery,
             Presentation, n_samples, duration_min
    """
    records = get_record_names(raw_dir)
    rows = []
    for rec in records:
        meta = parse_header_comments(rec, raw_dir)
        # Add signal-level info
        try:
            wf_rec, fhr, _ = load_record(rec, raw_dir)
            meta["n_samples"]    = len(fhr)
            meta["duration_min"] = len(fhr) / FS / 60.0
            meta["fs"]           = wf_rec.fs
        except Exception as e:
            meta["n_samples"]    = None
            meta["duration_min"] = None
            meta["fs"]           = None
        rows.append(meta)

    df = pd.DataFrame(rows)
    # Coerce numeric columns
    numeric_cols = ["pH", "pCO2", "pO2", "BE", "BDecf",
                    "Apgar1", "Apgar5", "age", "parity",
                    "gravidity", "weight", "n_samples", "duration_min", "fs"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
