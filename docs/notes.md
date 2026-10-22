# Project Notes

Notes captured during Phase 0 reading and Phase 1 EDA.

---

## Dataset facts (from PhysioNet page)

- **Source:** Czech Technical University (CTU) + University Hospital Brno (UHB)
- **N recordings:** 552 (selected from 9,164 recorded between 2010–2012)
- **Signals:** FHR (fetal heart rate) + UC (uterine contractions)
- **Sampling rate:** 4 Hz (both channels)
- **Max duration:** 90 minutes (starting ≤ 90 min before delivery)
- **File format:** WFDB `.dat` (binary signal) + `.hea` (header with metadata)
- **Total size:** ~38 MB
- **License:** Open Data Commons Attribution License v1.0 — attribution required, no login
- **Citation required:** Chudáček et al., BMC Pregnancy and Childbirth 2014 14:16

### Inclusion criteria used by dataset creators

- Singleton pregnancy, gestational age > 36 weeks
- No known developmental defects
- Stage 2 of labour ≤ 30 minutes
- FHR quality > 50% in each 30-min window
- Available umbilical artery pH
- Mostly vaginal deliveries (46 CS included)

### Available outcome columns (in .hea comments)

- `pH`, `pCO2`, `pO2`, `BE` (base excess), `BDecf`
- `Apgar1`, `Apgar5`
- Neonatology: O2 need, seizures, NICU admission
- Maternal: `age`, `parity`, `gravidity`
- Delivery: type, duration, meconium staining
- Fetal: `sex`, `weight`

---

## EDA findings (Phase 1, 2026-10-17)

- Signal length: 60.0–90.08 minutes; median 71.71 minutes; all recordings sampled at 4 Hz.
- Data quality: 552 metadata rows, 0 duplicate record IDs, 0 duplicate metadata rows, and complete `.dat`/`.hea` pairing.
- Data quality: 0 missing pH values, 0 implausible pH values outside 6.5–8.0, and all recordings sampled at 4 Hz.
- Other metadata missingness: BDecf/pCO2/BE each missing in 11 records, gravidity in 4, and weight in 1. These fields are not used as CTG predictors because they are outcome or clinical metadata.
- FHR missing: median 18.36%; mean 18.83%; maximum 53.50%, using values below 50 bpm or above 220 bpm as missing.
- Longest consecutive FHR gap: median 271.1 seconds; maximum 1,764.5 seconds. This confirms that missingness is not limited to short interpolation gaps.
- pH < 7.15 cases: 105/552 (19.02%).
- pH missing: 0 recordings.
- pH range: 6.85–7.47; no values outside the predeclared plausibility range of 6.5–8.0.
- Sampling rate: all 552 recordings report 4 Hz.
- Short recordings: 0 recordings under 10 minutes.
- Proposed signal-loss exclusion: 5 recordings exceed 50% missing FHR over the full recording; this count must be recomputed after Phase 2 selects the decision window.
- Surprising finding: the dataset has long missing runs despite having a high-quality inclusion criterion, so Phase 2 should avoid blanket interpolation and retain signal-loss features.

## Phase 2 preprocessing decisions (2026-10-17)

- Analysis window: the final 30 minutes before delivery, selected because the prototype is intended to estimate risk at the end of the available intrapartum trace and every recording is at least 60 minutes long.
- Invalid FHR rule: values below 50 bpm or above 220 bpm are treated as missing.
- Gap rule: interpolate only internal gaps of at most 15 seconds (60 samples at 4 Hz). Leading, trailing, and longer gaps remain missing.
- Signal-quality fields: percentage missing before and after interpolation, longest missing gap before and after interpolation, and number of interpolated samples.
- In the final 30-minute window, median FHR loss is 20.49% before interpolation and 11.08% after interpolation.
- 71 recordings exceed 50% missing FHR in the final window, leaving 481 recordings for the planned modelling set. This is more restrictive than the full-recording Phase 1 count of 5 because the final window can contain a concentrated dropout.
- The final 30-minute window and its exclusion counts are written to `data/processed/metadata_preprocessed.csv`.

## Phase 3 feature definitions (2026-10-17)

- `baseline_fhr_bpm`: median valid FHR in the final 30-minute window.
- `fhr_std_bpm`: overall spread of valid FHR values.
- `short_term_variability_bpm`: mean absolute change between adjacent valid samples.
- `long_term_variability_bpm`: mean one-minute FHR range.
- Accelerations: FHR at least 15 bpm above baseline for at least 15 seconds; count and total duration.
- Decelerations: FHR at least 15 bpm below baseline for at least 15 seconds; count, mean depth, and mean duration.
- Contractions: UC peaks separated by at least 60 seconds; rate is reported per 10 minutes.
- Late decelerations: approximate only; a detected deceleration nadir occurring 10–90 seconds after a detected UC peak.
- These are simplified research features, not formal FIGO/NICE annotations. They are intentionally documented so the thresholds can be changed and sensitivity-tested later.

## Phase 4 modelling protocol (2026-10-17)

- Models use only CTG-derived features and signal-quality features. Outcome measurements and post-delivery metadata are excluded.
- Evaluation uses the 481 non-excluded recordings, repeated stratified 5-fold CV with 5 repeats.
- Logistic regression imputation and scaling are fitted inside each training fold. Logistic `C` and Random Forest depth/leaf size are tuned in a 3-fold inner CV loop.
- The rule baseline flags baseline FHR outside 110–160 bpm, one-minute range below 5 bpm, at least 3 decelerations, or at least 1 approximate late deceleration.
- Recall at 90% specificity is calculated using the 90th percentile of negative-class scores within each test fold.
- Results are saved in `data/processed/model_results.csv` and fold-level values in `data/processed/model_fold_results.csv`.
- Phase 4 results: for the primary pH < 7.15 label, logistic L2 achieved mean ROC-AUC 0.756 (95% empirical interval 0.665–0.876) and mean PR-AUC 0.441 (0.333–0.602); Random Forest achieved ROC-AUC 0.745 and PR-AUC 0.405. The prevalence PR-AUC baseline was 0.170. These are exploratory estimates, not clinical performance claims.
- Across thresholds, discrimination was strongest for the rare pH < 7.05 label by ROC-AUC but with wide intervals, and weaker as the positive class became broader. The rule baseline was intentionally simple and high-sensitivity; its binary scores are poorly calibrated, so Brier score should not be compared directly with probabilistic models.

## Phase 5 calibration and explanation (2026-10-17)

- Calibration uses repeated out-of-fold predictions for the primary pH < 7.15 label. For each outer test fold, the logistic model is fitted only on the outer training partition; Platt scaling is then fitted to inner out-of-fold training predictions, keeping the outer test fold untouched.
- Raw logistic scores had Brier score 0.203 and mean predicted risk 0.438, substantially above the observed prevalence of 0.170. Platt calibration reduced Brier score to 0.126 and mean predicted risk to 0.169. This improvement is exploratory and requires external validation.
- Standardized logistic coefficients provide the global explanation. The most stable features were acceleration duration (top-five in 25/25 folds), mean deceleration duration (23/25), FHR standard deviation (21/25), and deceleration count (14/25).
- The direction of some associations is not automatically clinical: acceleration duration and FHR spread were positively associated with the low-pH label in this small retrospective sample, while acceleration count was negative. These findings require clinical review and external validation rather than mechanistic interpretation.
- Three case-study traces are selected from out-of-fold predictions. An exploratory 0.20 calibrated-risk operating point is used only to make TP/FP/FN examples readable; it is not a recommended clinical threshold.

## Phase 6 and 7 write-up and demo (2026-10-18)

- Added `README.md` with the problem statement, data source and citation, method, results, figures, limitations, reproducibility instructions, and next steps for real hospital data.
- Added `run_pipeline.py` as the one-command rebuild entry point and pinned notebook execution dependencies in `requirements.txt`.
- Added `app.py`, a Streamlit demonstration showing a selected final-window FHR/UC trace, repeated out-of-fold calibrated research score, and five largest plain-language logistic contributions.
- Added `docs/rehearsal.md` with 30-second and two-minute explanations plus concise answers to likely interview questions.
- The Streamlit app passed a local health-check smoke test. The full pipeline rebuilt EDA, preprocessing, and feature outputs; the environment then reported an external Windows file lock on the pre-existing `data/processed/model_results.csv` during model-output replacement. The modelling and explainability stages remain independently validated by the full test suite and prior artifact checks.
- The demo UI was redesigned using the selected Frontend Design and Web Interface Guidelines skills: signal-first hierarchy, deep slate clinical workspace, sea-glass teal signal accent, amber caution state, visible focus treatment, cached data loading, and explicit separation between observed outcome and model estimate.

---

## CTG basics (from reading)

- **Baseline FHR:** normal 110–160 bpm; measured over ≥ 10 min stable segments
- **Variability (STV/LTV):** beat-to-beat fluctuations; reduced variability is a non-reassuring sign
- **Accelerations:** transient rise ≥ 15 bpm for ≥ 15 s; reassuring
- **Decelerations types:**
  - *Early:* uniform, mirror contractions — usually benign (head compression)
  - *Late:* nadir after contraction peak — suggests uteroplacental insufficiency (concerning)
  - *Variable:* abrupt, variable shape — cord compression (context-dependent)
- **FIGO classification:** normal / suspicious / pathological — based on baseline, variability, accels, decels
- **Decision time:** in this prototype, "decision time" = the last 30 min of available FHR before delivery (chosen in Phase 2)

---

## Papers skimmed

| Paper | Key finding | Reported AUC |
|---|---|---|
| Not yet reviewed | Not yet reviewed | Not yet reviewed |
| Not yet reviewed | Not yet reviewed | Not yet reviewed |

---

## Open questions

- What exact segment to use? (Last 30 min vs last 60 min vs full recording)
- How to handle recordings shorter than chosen window?
- Is UC signal reliable enough for late deceleration detection?
