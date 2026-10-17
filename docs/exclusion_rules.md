# Exclusion Rules

**Status:** FROZEN for the exploratory analysis — rules were recorded before modelling.

## Principle

Exclusion rules are written here before looking at model performance. Any recording that meets any rule below is dropped from all analyses (training and evaluation).

---

## Rule 1 — Missing pH

**Rule:** Drop any recording where `pH` is missing (NaN).
**Rationale:** No label → can't be used for supervised learning.
**Expected impact:** 0 recordings in Phase 1; all 552 have a pH value.

---

## Rule 2 — Excessive signal loss

**Rule:** Drop recordings where more than **50%** of the FHR signal in the analysis window is missing (0, dropout, or physiologically implausible value).
**Threshold:** 50% missing FHR. Phase 1 measured this across the full recording; Phase 2 must recompute it for the frozen decision window.
**Rationale:** A recording with majority missing FHR provides too little information for feature extraction to be meaningful.
**Expected impact:** Phase 2 recomputes this on the final 30-minute window. The final count is recorded in `metadata_preprocessed.csv`.

---

## Rule 3 — Recording too short

**Rule:** Drop recordings shorter than **10 minutes** of recording duration.
**Threshold:** 10 minutes.
**Rationale:** Feature extraction (variability, deceleration counts) is unreliable on very short windows.
**Expected impact:** 0 recordings; the shortest recording is 60 minutes.

---

## Exclusion counts (fill in after EDA)

| Rule | Recordings excluded | Cumulative remaining |
|---|---|---|
| Start | — | 552 |
| Rule 1: missing pH | 0 | 552 |
| Rule 2: excessive signal loss (>50%) | 71 | 481 |
| Rule 3: too short (<10 min) | 0 | 547 |
| **Final analysis set** | — | **481** |

---

## Notes

- Exclusions are applied in the order listed above.
- Any additional rules discovered during EDA must be added here with a rationale **before** looking at model results.
- The frozen analysis window is the final 30 minutes before delivery. Phase 2 recomputes signal loss within this window; use `data/processed/metadata_preprocessed.csv` for the final counts.
