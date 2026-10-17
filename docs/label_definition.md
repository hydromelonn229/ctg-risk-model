# Label Definition

**Status:** FROZEN for the exploratory analysis — decided before modelling.

## Primary label

- **Definition:** umbilical artery cord pH < **7.15** = positive class ("poor outcome")
- **Rationale:** pH < 7.15 is a widely used threshold for fetal acidaemia in the obstetric literature. It is below the threshold for mild acidaemia (7.20) and above the threshold for severe acidaemia (7.05), balancing case count against clinical significance.
- **Column used:** `pH` from the `.hea` file comments
- **Units:** pH units (dimensionless)

## Sensitivity analyses (run after primary analysis is complete)

| Threshold | Clinical meaning |
|---|---|
| pH < 7.05 | Severe acidaemia |
| pH < 7.15 | **Primary label** |
| pH < 7.20 | Mild acidaemia |

## Data verification checklist

- [x] Confirmed column name in `.hea` file: `pH`
- [x] Checked units (pH units, not scaled)
- [x] Counted recordings with missing pH
- [x] Counted positive cases at each threshold
- [x] Checked for duplicate or implausible pH values (< 6.5 or > 8.0)

## Positive class counts (fill in after EDA)

| Threshold | Positive (n) | Total with pH | Prevalence |
|---|---|---|---|
| pH < 7.05 | 40 | 552 | 7.25% |
| pH < 7.15 | 105 | 552 | 19.02% |
| pH < 7.20 | 177 | 552 | 32.07% |

## Decision record

- Date decided: 2026-10-17
- Decided by: Project owner
- **Frozen before:** any model results — do not change during the exploratory analysis
