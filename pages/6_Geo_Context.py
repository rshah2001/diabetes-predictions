import numpy as np
import pandas as pd
import streamlit as st

from src.geo import (
    MEASURE_LABELS,
    TRAIN_PREVALENCE,
    adjust_probability_for_prevalence,
    county_context,
    load_places,
)

st.title("6) Geographic Risk Context")

st.caption(
    "Diabetes risk isn't just personal — it varies a lot by where you live. "
    "This page combines your model prediction with CDC PLACES county-level data "
    "(adult diabetes prevalence, obesity, inactivity, blood pressure, insurance access) "
    "to put the risk score in local context."
)


@st.cache_data(show_spinner="Downloading CDC PLACES county data (first run only)...")
def get_places() -> pd.DataFrame:
    return load_places()


try:
    places = get_places()
except Exception as e:
    st.error(
        f"Could not load CDC PLACES data ({e}). "
        "Check your internet connection and reload."
    )
    st.stop()

# --- County picker --------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    state_names = sorted(places["state_name"].unique())
    default_state = state_names.index("Arizona") if "Arizona" in state_names else 0
    state = st.selectbox("State", state_names, index=default_state)
with col2:
    counties = places[places["state_name"] == state].sort_values("county")
    county = st.selectbox("County", counties["county"].tolist())

fips = counties[counties["county"] == county].iloc[0]["county_fips"]
ctx = county_context(places, fips)

# --- County health profile -------------------------------------------------
st.subheader(f"{ctx['county']} County, {ctx['state']} — health profile")
st.caption(f"Adult population context for ~{ctx['population']:,} residents (CDC PLACES, 2024 release).")

cols = st.columns(len(MEASURE_LABELS))
for c, (key, m) in zip(cols, ctx["measures"].items()):
    delta = m["county_pct"] - m["national_median_pct"]
    c.metric(
        m["label"],
        f"{m['county_pct']:.1f}%",
        delta=f"{delta:+.1f}% vs national median",
        delta_color="inverse",
    )

st.caption(
    "Percentile rank among all US counties (higher = more burden): "
    + ", ".join(
        f"{m['label'].split(' (')[0]} {m['percentile']:.0f}th"
        for m in ctx["measures"].values()
    )
)

# --- National map ----------------------------------------------------------
st.subheader("County diabetes prevalence map")
st.caption(
    "Each dot is a county, colored by adult diabetes prevalence "
    "(green = low, red = high). The selected county is shown larger in blue."
)

map_df = places.dropna(subset=["lat", "lon"]).copy()
q = map_df["diabetes_pct"].rank(pct=True)
red = (q * 255).astype(int)
green = ((1 - q) * 255).astype(int)
map_df["color"] = [f"#{r:02x}{g:02x}40" for r, g in zip(red, green)]
map_df["size"] = 2500.0
sel = map_df["county_fips"] == ctx["county_fips"]
map_df.loc[sel, "color"] = "#1f77ff"
map_df.loc[sel, "size"] = 30000.0
st.map(map_df, latitude="lat", longitude="lon", color="color", size="size")

# --- Personal risk in local context ----------------------------------------
st.divider()
st.subheader("Your prediction in local context")

last = st.session_state.get("last_prediction")
if last is not None:
    st.caption("Using the most recent single-patient prediction from the Predict page.")
    prob = float(last["probability"])
else:
    st.caption(
        "No prediction found in this session — run one on the Predict page, "
        "or enter a probability manually."
    )
    prob = st.number_input(
        "Model probability (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01
    )

county_prev = ctx["measures"]["diabetes_pct"]["county_pct"] / 100.0
adjusted = adjust_probability_for_prevalence(prob, county_prev)
rel = adjusted / county_prev if county_prev else float("nan")

c1, c2, c3 = st.columns(3)
c1.metric("Model risk score", f"{prob:.1%}")
c2.metric(f"Adjusted to {ctx['county']} County base rate", f"{adjusted:.1%}")
c3.metric("vs average adult in this county", f"{rel:.1f}×")

st.caption(
    f"The model was trained on a cohort where {TRAIN_PREVALENCE:.0%} of patients had diabetes, "
    f"so its raw score is relative to that high-prevalence group. The adjusted figure rescales "
    f"the same evidence to {ctx['county']} County's adult prevalence of {county_prev:.1%} "
    "(standard prior correction on the odds). "
    "This is a screening-style illustration, not a medical diagnosis."
)
