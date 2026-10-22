# Interview rehearsal

## Thirty-second version

This project is an exploratory CTG risk prototype using 552 public
cardiotocography recordings. I froze the final 30 minutes before delivery,
cleaned implausible FHR values without filling long gaps, and extracted
clinically named measurements such as baseline, variability, accelerations,
decelerations, contractions, and signal quality. Leakage-safe repeated nested
cross-validation showed that regularized logistic regression discriminated
better than prevalence and rule baselines for the primary cord-pH label. I
then calibrated the score and explained it with standardized coefficients and
case studies. It is not a clinical tool: the dataset is small and single-site,
the annotations are simplified, and prospective external validation is still
required.

## Two-minute version

The question was whether a small set of interpretable measurements from the
last 30 minutes of a fetal heart-rate and uterine-contraction trace could
retrospectively identify low umbilical artery cord pH. I used the CTU-UHB
PhysioNet dataset, checked the metadata and WFDB files, and froze the label
definition before modelling: pH below 7.15 as primary, with 7.05 and 7.20 as
sensitivity analyses.

For preprocessing, FHR below 50 or above 220 bpm became missing. I interpolated
only internal gaps up to 15 seconds and preserved longer gaps. Recordings with
more than 50% missing FHR in the final window were excluded, leaving 481
records. The features were deliberately readable: baseline FHR, variability,
acceleration and deceleration summaries, approximate late decelerations,
contraction frequency, and signal-quality fields.

I compared prevalence and rule baselines with L1 and L2 logistic regression and
a class-weighted Random Forest. The evaluation used repeated stratified 5-fold
cross-validation with nested inner tuning, so imputation, scaling, and
hyperparameter selection stayed inside the training data. For pH below 7.15,
L2 logistic regression reached an exploratory mean ROC-AUC of 0.756 and PR-AUC
of 0.441, while the prevalence PR-AUC was 0.170.

The raw logistic score was overconfident, so I fitted Platt calibration only
to inner out-of-fold training predictions. Its Brier score improved from
0.203 to 0.126 internally. Standardized coefficients and repeated-fold
stability showed that acceleration duration, mean deceleration duration, FHR
spread, and deceleration count were frequent top features. I would not call
those causal or clinically definitive because this is a small retrospective
dataset.

The next real-world step would be a prospective, multi-site evaluation with a
predefined clinical decision, subgroup calibration checks, alarm-burden
analysis, comparison with current practice, and governance review.

In the demo, I would start with the “Highest model score” browsing view rather
than manually scanning all record IDs. The “Case studies” view provides the
true-positive, false-positive, and false-negative examples from Phase 5, while
the confirmed-low-pH view is explicitly retrospective because that outcome is
only known after delivery.

## Questions and answers

### Why did you choose the final 30 minutes?

It is a fixed, reproducible decision window close to delivery and is available
for every record. It is a modelling choice, not a claim that 30 minutes is the
correct clinical horizon.

### Why not use cord pH as a feature?

Cord pH is measured after delivery and is the outcome being predicted. Using
it, or related post-delivery variables such as Apgar scores or birth weight,
would leak future information into the model.

### Why are the feature definitions approximate?

The project is a transparent prototype, not a formal computerized CTG
annotation system. The thresholds are documented and testable, but clinical
deployment would need expert-reviewed annotations and guideline-aligned
definitions.

### Why report PR-AUC as well as ROC-AUC?

The low-pH outcome is imbalanced. PR-AUC makes performance relative to the
positive-class prevalence more visible; the prevalence baseline is about 0.17
for the primary label.

### Can the 0.126 Brier score be trusted?

It is an internal out-of-fold estimate, not external evidence. Calibration can
look better or worse in another hospital, time period, device, or subgroup.

### What is the most important limitation?

Generalisation. The dataset is relatively small and from one source, while
missingness and delivery practice may be site-specific. External prospective
validation is required before any clinical interpretation.

### What would you change first?

I would define the clinical decision and acceptable false-alarm burden with
clinicians, then evaluate on temporally separated multi-site data with
pre-specified calibration, subgroup, drift, and alarm-burden analyses.
