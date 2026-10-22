"""Streamlit demo for the exploratory CTG risk prototype."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loading import FS, resolve_data_dir
from src.explainability import local_contributions
from src.preprocessing import prepare_recording

PROCESSED = ROOT / "data" / "processed"
RAW_DIR = resolve_data_dir(ROOT / "data" / "raw")
PRIMARY_THRESHOLD = 7.15
CASE_DECISION_THRESHOLD = 0.20


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(
        PROCESSED / "ctg_features.csv",
        converters={"record": lambda value: str(value).split(".")[0]},
    )
    predictions = pd.read_csv(
        PROCESSED / "calibration_oof_predictions.csv",
        converters={"record": lambda value: str(value).split(".")[0]},
    )
    coefficients = pd.read_csv(PROCESSED / "logistic_coefficients.csv")
    return features, predictions, coefficients


@st.cache_data
def load_trace(record: str) -> tuple[np.ndarray, np.ndarray]:
    fhr, uc, _ = prepare_recording(record, RAW_DIR, fs=FS)
    return fhr, uc


st.set_page_config(
    page_title="CTG study | final-window review",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
      --ink: #102331;
      --slate: #193847;
      --panel: #21495a;
      --paper: #edf4f1;
      --teal: #66d5c7;
      --amber: #f1c46a;
      --muted: #abc4c8;
    }
    .stApp { background: var(--ink); color: var(--paper); }
    [data-testid="stHeader"] { background: var(--ink); }
    [data-testid="stSidebar"] { background: #0d1d29; border-right: 1px solid #2b5360; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--paper); }
    .brand {
      border-bottom: 1px solid #2b5360; padding: 0.25rem 0 1.3rem;
      margin-bottom: 1.5rem;
    }
    .brand-mark { color: var(--teal); font-size: 1.8rem; font-weight: 800; letter-spacing: -0.08em; }
    .brand-copy { color: var(--muted); font-size: 0.78rem; margin-left: 0.55rem; }
    .hero {
      background: var(--slate); border: 1px solid #2b5360; border-left: 5px solid var(--teal);
      padding: 1.5rem 1.7rem; margin-bottom: 1rem;
    }
    .hero h1 { color: var(--paper); font-size: 2.15rem; letter-spacing: -0.04em; margin: 0 0 .4rem; }
    .hero p { color: var(--muted); margin: 0; max-width: 42rem; }
    .notice {
      background: var(--amber); color: #182733; padding: .75rem 1rem;
      font-weight: 700; margin-bottom: 1.2rem;
    }
    .section-title {
      color: var(--paper); font-size: 1.2rem; font-weight: 700;
      border-bottom: 1px solid #2b5360; padding-bottom: .45rem; margin: 1.5rem 0 .8rem;
    }
    .score-panel {
      background: var(--panel); border-top: 3px solid var(--teal);
      padding: 1rem 1.2rem; min-height: 7.4rem;
    }
    .score-label { color: var(--muted); font-size: .78rem; }
    .score-value { color: var(--teal); font-size: 2.25rem; font-weight: 750; line-height: 1.2; }
    .score-note { color: var(--muted); font-size: .75rem; }
    .quality-good { color: var(--teal); font-weight: 700; }
    .quality-watch { color: var(--amber); font-weight: 700; }
    .stButton > button:focus, .stSelectbox [data-baseweb="select"]:focus-within {
      outline: 3px solid var(--amber); outline-offset: 2px;
    }
    [data-testid="stMetricValue"] { color: var(--paper); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

features, predictions, coefficients = load_tables()
usable = features.loc[~features["excluded"]].copy()
prediction_summary = (
    predictions.groupby("record", as_index=False)
    .agg(
        raw_score=("raw_score", "mean"),
        calibrated_score=("calibrated_score", "mean"),
        y_true=("y_true", "first"),
    )
)
records = usable.merge(prediction_summary, on="record", validate="one_to_one")
records["research_class"] = np.select(
    [
        (records["y_true"] == 1) & (records["calibrated_score"] >= CASE_DECISION_THRESHOLD),
        (records["y_true"] == 0) & (records["calibrated_score"] >= CASE_DECISION_THRESHOLD),
        (records["y_true"] == 1) & (records["calibrated_score"] < CASE_DECISION_THRESHOLD),
    ],
    ["True positive", "False positive", "False negative"],
    default="Other",
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">∿ CTG</span>'
        '<span class="brand-copy">research review</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Find a recording")
    browse_mode = st.radio(
        "Browse by",
        [
            "Highest model score",
            "Case studies",
            "Confirmed low pH (research)",
            "All recordings",
        ],
        index=0,
        help=(
            "These are research navigation aids. The score is retrospective, "
            "and the confirmed outcome is only known after delivery."
        ),
    )
    search_term = st.text_input(
        "Search record ID",
        placeholder="e.g. 1198",
        help="Enter part or all of a recording identifier.",
    ).strip()
    if browse_mode == "Highest model score":
        candidates = records.sort_values(
            ["calibrated_score", "record"], ascending=[False, True]
        )
    elif browse_mode == "Case studies":
        candidates = records.loc[records["research_class"] != "Other"].copy()
        candidates = candidates.sort_values(
            ["research_class", "calibrated_score"], ascending=[True, False]
        )
    elif browse_mode == "Confirmed low pH (research)":
        candidates = records.loc[records["y_true"] == 1].sort_values(
            ["calibrated_score", "record"], ascending=[False, True]
        )
    else:
        candidates = records.sort_values("record")
    if search_term:
        candidates = candidates.loc[candidates["record"].str.contains(search_term, na=False)]
    if candidates.empty:
        st.warning("No recordings match this view. Try clearing the search.")
        candidates = records.sort_values("record").head(1)
    st.caption(f"{len(candidates)} recording(s) in this view")
    selected_record = st.selectbox(
        "Recording",
        candidates["record"].tolist(),
        label_visibility="collapsed",
        help="Repeated out-of-fold estimate for the primary pH < 7.15 label.",
    )
    selected_row = candidates.loc[candidates["record"] == selected_record].iloc[0]
    if browse_mode == "Case studies":
        st.info(
            f"{selected_row['research_class']}: "
            f"{selected_row['calibrated_score']:.1%} research score"
        )
    else:
        st.caption(
            f"Top of list: {candidates.iloc[0]['record']} "
            f"({candidates.iloc[0]['calibrated_score']:.1%})"
        )
    st.divider()
    st.caption("Primary outcome")
    st.markdown("Umbilical artery cord pH **< 7.15**")
    st.caption("The score is a retrospective research estimate, not a diagnosis.")

case = records.loc[records["record"] == selected_record].iloc[0]
quality_class = (
    "quality-good" if case["pct_signal_missing"] < 20 else "quality-watch"
)
quality_label = "Good signal coverage" if case["pct_signal_missing"] < 20 else "Review signal coverage"

st.markdown(
    '<div class="hero"><h1>Reading the final 30 minutes</h1>'
    "<p>Explore the signal, the data quality, and the model's reasoning in that order.</p></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice">Exploratory research prototype — not for clinical use. '
    "Cord pH is shown for research inspection and was not used as a model input.</div>",
    unsafe_allow_html=True,
)
with st.expander("How to find an interesting recording"):
    st.write(
        "Start with **Highest model score** to review the records the model ranks "
        "highest. Use **Case studies** for the true-positive, false-positive, and "
        "false-negative examples from Phase 5. **Confirmed low pH (research)** "
        "uses the outcome recorded after delivery, so it is useful for retrospective "
        "learning only and would not be available in a live setting."
    )

st.markdown('<div class="section-title">Selected record</div>', unsafe_allow_html=True)
score_col, outcome_col, quality_col = st.columns([1.2, 1, 1])
with score_col:
    st.markdown(
        f'<div class="score-panel"><div class="score-label">Calibrated research estimate</div>'
        f'<div class="score-value">{case["calibrated_score"]:.1%}</div>'
        f'<div class="score-note">Primary label: pH &lt; {PRIMARY_THRESHOLD:.2f}</div></div>',
        unsafe_allow_html=True,
    )
with outcome_col:
    st.markdown(
        f'<div class="score-panel"><div class="score-label">Observed cord pH</div>'
        f'<div class="score-value">{case["pH"]:.2f}</div>'
        '<div class="score-note">Outcome shown after the model estimate</div></div>',
        unsafe_allow_html=True,
    )
with quality_col:
    st.markdown(
        f'<div class="score-panel"><div class="score-label">Signal coverage</div>'
        f'<div class="{quality_class}">{quality_label}</div>'
        f'<div class="score-note">{case["pct_signal_missing"]:.1f}% missing after cleaning</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Signal review</div>', unsafe_allow_html=True)
fhr, uc = load_trace(str(selected_record))
time_min = np.arange(len(fhr)) / FS / 60.0
fig, axes = plt.subplots(
    2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
)
fig.patch.set_facecolor("#193847")
for axis in axes:
    axis.set_facecolor("#193847")
    axis.tick_params(colors="#abc4c8", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#2b5360")
axes[0].plot(time_min, fhr, color="#66d5c7", linewidth=0.75)
axes[0].axhspan(110, 160, color="#66d5c7", alpha=0.08)
axes[0].set_ylabel("FHR (bpm)", color="#edf4f1")
axes[0].set_ylim(40, 230)
axes[1].plot(time_min, uc, color="#f1c46a", linewidth=0.75)
axes[1].set_ylabel("UC", color="#edf4f1")
axes[1].set_xlabel("Minutes in final analysis window", color="#edf4f1")
fig.tight_layout()
st.pyplot(fig, clear_figure=True, use_container_width=True)
st.caption(
    "The shaded FHR band marks 110–160 bpm as a visual reference only. "
    "Long gaps remain visible rather than being invented by interpolation."
)

st.markdown('<div class="section-title">Model reasoning</div>', unsafe_allow_html=True)
case_for_contributions = pd.DataFrame([case]).assign(case_type="selected")
contributions = local_contributions(case_for_contributions, features)
contributions = contributions.merge(coefficients[["feature", "direction"]], on="feature")
contributions["plain_language"] = contributions["feature"].map(
    {
        "acceleration_duration_s": "total time in accelerations",
        "deceleration_mean_duration_s": "average deceleration duration",
        "fhr_std_bpm": "overall FHR spread",
        "n_decelerations": "number of decelerations",
        "long_term_variability_bpm": "one-minute FHR variability",
        "n_late_decelerations": "approximate late decelerations",
    }
).fillna(contributions["feature"].str.replace("_", " ", regex=False))
contributions["effect"] = contributions["contribution"].map(
    lambda value: "increased score" if value > 0 else "decreased score"
)
st.dataframe(
    contributions[
        ["rank", "plain_language", "feature_value", "effect", "contribution"]
    ].rename(
        columns={
            "rank": "Rank",
            "plain_language": "Feature",
            "feature_value": "Observed value",
            "effect": "Model effect",
            "contribution": "Contribution",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Contributions describe association in this retrospective model. "
    "They do not establish causation or clinical importance."
)

with st.expander("About this view"):
    st.write(
        "The model was trained on CTG-derived measurements and signal-quality "
        "fields using repeated nested cross-validation. The displayed score is "
        "an averaged out-of-fold estimate. The exploratory 20% operating point "
        "used in the case studies is not a clinical threshold."
    )
