import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("2) Insights (EDA)")

st.caption(
    "This page helps you understand what the data looks like before using machine learning. "
    "It shows how many people have diabetes vs not, how each health measurement is distributed, "
    "and how different measurements relate to each other."
)

if "data_pack" not in st.session_state:
    st.warning("Upload and prepare data first in the Upload page.")
    st.stop()

pack = st.session_state["data_pack"]
df_raw: pd.DataFrame = pack["df_raw"].copy()
target_col: str = pack["target_col"]
feature_cols = pack["feature_cols"]

# ---------------------------
# Dataset preview
# ---------------------------
st.subheader("Dataset preview")
st.caption(
    "This is a small preview of the dataset after basic cleaning (for example, handling missing values)."
)
st.dataframe(df_raw.head(25), use_container_width=True)

st.divider()

# ---------------------------
# Class balance
# ---------------------------
st.subheader("Class balance")
st.caption(
    "This chart shows how many people in the dataset have diabetes versus how many do not. "
    "Each bar represents a group of people. A taller bar means more people in that group."
)
st.caption("Here, 0 means no diabetes and 1 means diabetes.")

counts = df_raw[target_col].value_counts(dropna=False).sort_index()

fig = plt.figure()
plt.bar([str(x) for x in counts.index], counts.values)
plt.xlabel("Diabetes outcome (0 = No, 1 = Yes)")
plt.ylabel("Number of patients")
st.pyplot(fig, clear_figure=True)

st.divider()

# ---------------------------
# Feature distributions
# ---------------------------
st.subheader("Feature distributions")
st.caption(
    "This chart shows how values of one health measurement (for example, Age or Glucose) "
    "are spread across all patients in the dataset."
)
st.caption(
    "• The x-axis shows the actual values of the measurement (for example, age in years).\n"
    "• The y-axis shows how many patients fall into each value range.\n"
    "Taller bars mean more patients have values in that range."
)
st.info(
    "How to read these charts:\n"
    "• Wider spreads mean more variation among patients.\n"
    "• Tall bars show common values.\n"
    "• Extreme values at the edges may indicate outliers or unusual cases."
)

feature = st.selectbox("Choose a feature", feature_cols)

fig = plt.figure()
plt.hist(df_raw[feature].dropna(), bins=30)
plt.xlabel(f"{feature} (feature value)")
plt.ylabel("Number of patients")
st.pyplot(fig, clear_figure=True)

st.divider()

# ---------------------------
# Correlation
# ---------------------------
st.subheader("Correlation (numeric columns)")
st.caption(
    "Correlation shows how two measurements change together. "
    "A higher positive value means that when one measurement increases, "
    "the other tends to increase as well."
)
st.caption(
    "For example, if glucose and diabetes outcome have a high correlation, "
    "higher glucose levels are more commonly seen in patients with diabetes."
)
st.caption(
    "Values close to 0 mean little or no linear relationship, while values closer to 1 or -1 "
    "mean a stronger relationship."
)

num_df = df_raw[[c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]].copy()

if num_df.shape[1] >= 2:
    corr = num_df.corr(numeric_only=True)

    st.subheader("Correlation table")
    st.caption(
        "Each cell shows the relationship between two columns. "
        "Example: a higher value between Glucose and Outcome suggests glucose tends to be higher in diabetic patients."
    )
    st.dataframe(corr, use_container_width=True)

    st.subheader("Correlation heatmap")
    st.caption(
        "This heatmap is the same correlation table, but visual. Brighter colors indicate stronger relationships."
    )
    fig = plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar()
    st.pyplot(fig, clear_figure=True)
else:
    st.info("Not enough numeric columns for correlation.")
