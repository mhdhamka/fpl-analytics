import os

import pandas as pd
import streamlit as st

from src.config import PLAYERS_CLEANED_PATH
from src.model import load_feature_importances
from src.predict import ModelNotTrainedError, load_metadata, load_model, predict_points

# Use the local PNG for the browser tab icon, with an emoji fallback if missing
favicon_path = "assets/images/pl.png"
page_icon = favicon_path if os.path.exists(favicon_path) else "⚽"

st.set_page_config(
    page_title="Premier League | Player Stats & FPL Predictor",
    page_icon=page_icon,
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #1a0b2e; color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #26053b; border-right: 1px solid #3d0a5c; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #ffffff !important; }
    div[data-testid="stMetric"] {
        background-color: #280b3d; border: 1px solid #4a156e; padding: 15px;
        border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label { color: #0ff5c4 !important; font-weight: 600; }
    h3 { color: #0ff5c4 !important; border-bottom: 2px solid #37003c; padding-bottom: 5px; margin-top: 25px; }
    .stAlert { background-color: #280b3d !important; border: 1px solid #4a156e !important; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    if not os.path.exists(PLAYERS_CLEANED_PATH):
        return None
    df = pd.read_csv(PLAYERS_CLEANED_PATH)
    df["full_name"] = df["first_name"].fillna("") + " " + df["second_name"]
    return df


@st.cache_resource
def load_cached_model():
    """Loads the model persisted by src/model.py — trained once offline,
    not on every Streamlit session. Returns None if none exists yet.
    """
    try:
        return load_model()
    except ModelNotTrainedError:
        return None


df = load_data()
model = load_cached_model()
metadata = load_metadata()

col_logo, col_text = st.columns([0.08, 0.92])
with col_logo:
    if os.path.exists("assets/images/pl.png"):
        st.image("assets/images/pl.png", width=55)
    else:
        st.markdown("<div style='font-size: 2.5rem;'>⚽</div>", unsafe_allow_html=True)
with col_text:
    st.markdown(
        """
        <h1 style="color: #ffffff; margin: 0px 0px 4px 0px; font-weight: 800; font-size: 2.2rem;">Premier League Performance Hub</h1>
        <p style="color: #0ff5c4; margin: 0px; font-size: 1.0rem; font-weight: 500;">Player statistics, underlying metrics, and ML-powered FPL projections.</p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

if df is None or df.empty:
    st.error("Processed data not found. Run `python main.py` first to generate the dataset and model.")
    st.stop()

if model is None:
    st.warning(
        "No trained model found yet. Player stats below still work, but predictions won't. "
        "Run `python main.py` (or `python -m src.model`) to train one."
    )

tab_explore, tab_predict, tab_insights = st.tabs(["Explore Players", "Predict Points", "Model Insights"])

# ---------------------------------------------------------------- Explore --
with tab_explore:
    st.sidebar.markdown("## **Filter Hub**")
    st.sidebar.markdown("---")

    teams = sorted(df["team_name"].dropna().unique())
    selected_team = st.sidebar.selectbox("Select Club", ["All Clubs"] + teams)

    positions = sorted(df["position"].dropna().unique())
    selected_position = st.sidebar.selectbox("Select Position", ["All Positions"] + positions)

    filtered_df = df.copy()
    if selected_team != "All Clubs":
        filtered_df = filtered_df[filtered_df["team_name"] == selected_team]
    if selected_position != "All Positions":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]

    player_names = sorted(filtered_df["full_name"].tolist())
    st.sidebar.markdown("---")
    selected_player_name = st.sidebar.selectbox("Select Player", player_names) if player_names else None
    if not player_names:
        st.sidebar.warning("No players match the selected filters.")

    if selected_player_name:
        player_data = filtered_df[filtered_df["full_name"] == selected_player_name].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Points", int(player_data.get("total_points", 0)))
        with col2:
            st.metric("Goals Scored", int(player_data.get("goals_scored", 0)))
        with col3:
            st.metric("Expected Goals (xG)", f"{player_data.get('expected_goals', 0):.2f}")
        with col4:
            st.metric("Points per £M", f"{player_data.get('points_per_million', 0):.1f}")

        st.subheader(f"Player Profile: {selected_player_name}")
        stat_cols = ["team_name", "position", "minutes", "assists", "clean_sheets",
                     "expected_assists", "market_value_m", "goals_per_90", "minutes_share"]
        available_stats = {c.replace("_", " ").title(): player_data.get(c, "N/A") for c in stat_cols if c in player_data}
        st.dataframe(pd.DataFrame([available_stats]), hide_index=True)

        if model is not None:
            st.subheader("Machine Learning Prediction")
            try:
                pred = predict_points(model, pd.DataFrame([player_data]))
                st.success(f"**ML Predicted Total Points:** {pred.iloc[0]:.1f} pts (model: {metadata.get('model_name', 'n/a')})")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    with st.expander("View Complete Filtered Squad Table"):
        st.dataframe(filtered_df, use_container_width=True)

# ---------------------------------------------------------------- Predict --
with tab_predict:
    st.subheader("Predict Points for a Custom Player Profile")
    st.caption("Slide the underlying metrics to see how the model's fantasy-points forecast responds.")

    if model is None:
        st.info("Train a model first (`python main.py`) to use this tab.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            minutes = st.slider("Minutes played", 0, 3420, 2000)
            goals_scored = st.slider("Goals scored", 0, 40, 5)
            assists = st.slider("Assists", 0, 25, 4)
        with c2:
            clean_sheets = st.slider("Clean sheets", 0, 25, 0)
            expected_goals = st.slider("Expected Goals (xG)", 0.0, 35.0, 5.0)
            expected_assists = st.slider("Expected Assists (xA)", 0.0, 20.0, 4.0)
        with c3:
            market_value_m = st.slider("Market value (£m)", 4.0, 15.0, 7.0, step=0.1)
            position = st.selectbox("Position", sorted(df["position"].dropna().unique()))

        nineties = max(minutes, 1) / 90.0
        custom_row = pd.DataFrame([{
            "minutes": minutes,
            "goals_scored": goals_scored,
            "assists": assists,
            "clean_sheets": clean_sheets,
            "expected_goals": expected_goals,
            "expected_assists": expected_assists,
            "market_value_m": market_value_m,
            "goals_per_90": goals_scored / nineties,
            "assists_per_90": assists / nineties,
            "xg_per_90": expected_goals / nineties,
            "xa_per_90": expected_assists / nineties,
            "points_per_million": 0.0,  # unknown ahead of prediction; not circular since it's a minor feature
            "minutes_share": min(minutes / (38 * 90), 1.0),
            "position": position,
        }])

        pred = predict_points(model, custom_row)
        st.success(f"**Predicted Total Points:** {pred.iloc[0]:.1f} pts")

# --------------------------------------------------------------- Insights --
with tab_insights:
    st.subheader("Model Performance")
    if not metadata:
        st.info("No model metadata found yet. Train a model first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", metadata.get("model_name", "n/a"))
        c2.metric("Hold-out R²", f"{metadata.get('test_r2', 0):.3f}")
        c3.metric("Hold-out MAE", f"{metadata.get('test_mae', 0):.2f} pts")
        c4.metric("Trained On", f"{metadata.get('n_training_rows', 0)} players")

        st.caption(f"Last trained: {metadata.get('trained_at_utc', 'unknown')}")

        st.subheader("Feature Importance")
        try:
            importances = load_feature_importances(model)
            st.bar_chart(importances.set_index("feature").head(15))
        except Exception as exc:
            st.warning(f"Could not load feature importances: {exc}")

        st.subheader("Top Value Picks (Points per £Million)")
        value_df = df[df["minutes"] >= 450].sort_values("points_per_million", ascending=False).head(15)
        st.dataframe(
            value_df[["full_name", "team_name", "position", "market_value_m", "total_points", "points_per_million"]],
            hide_index=True, use_container_width=True,
        )