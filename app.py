import os
import streamlit as st
import pandas as pd
import joblib
from src.config import PLAYERS_CLEANED_PATH, ML_FEATURES
from src.model import train_points_model

# Page Configuration with local project icon path
st.set_page_config(
    page_title="Premier League | Player Stats & FPL Predictor",
    page_icon="assets/images/pl.png",
    layout="wide"
)

# Custom CSS for Official Premier League Themeing & Clean Column Alignment
st.markdown("""
    <style>
    /* Main background & font styling */
    .stApp {
        background-color: #1a0b2e;
        color: #ffffff;
        font-family: 'Premier Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #26053b;
        color: white;
        border-right: 1px solid #3d0a5c;
    }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #280b3d;
        border: 1px solid #4a156e;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #0ff5c4 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
    }

    /* Subheaders */
    h3 {
        color: #0ff5c4 !important;
        border-bottom: 2px solid #37003c;
        padding-bottom: 5px;
        margin-top: 25px;
    }
    
    /* Success / Info Boxes */
    .stAlert {
        background-color: #280b3d !important;
        border: 1px solid #4a156e !important;
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
@st.cache_data
def load_data():
    if not os.path.exists(PLAYERS_CLEANED_PATH):
        return None
    df = pd.read_csv(PLAYERS_CLEANED_PATH)
    df["full_name"] = df["first_name"].fillna('') + " " + df["second_name"]
    return df

@st.cache_resource
def load_or_train_model():
    return train_points_model()

df = load_data()
model = load_or_train_model()

# Perfectly Aligned Header Layout (Clean columns without spacer hacks)
col_logo, col_text = st.columns([0.07, 0.93])
with col_logo:
    if os.path.exists("assets/images/pl.png"):
        st.image("assets/images/pl.png", width=60)
with col_text:
    st.markdown("""
        <h1 style="color: #ffffff; margin: 0px 0px 4px 0px; font-weight: 800; font-size: 2.2rem; line-height: 1.2;">Premier League Performance Hub</h1>
        <p style="color: #0ff5c4; margin: 0px; font-size: 1.0rem; font-weight: 500;">Official player statistics, underlying performance metrics ($xG$), and machine learning-powered FPL projections.</p>
    """, unsafe_allow_html=True)

st.markdown("---")

if df is None or df.empty:
    st.error("Processed data not found! Please run `python main.py` first to generate the dataset.")
else:
    # Sidebar Filters
    st.sidebar.markdown("## **Filter Hub**")
    st.sidebar.markdown("---")
    
    # 1. Team Filter
    teams = sorted(df["team_name"].unique()) if "team_name" in df.columns else []
    selected_team = st.sidebar.selectbox("Select Club", ["All Clubs"] + list(teams))

    # 2. Position Filter
    positions = df["position"].unique() if "position" in df.columns else []
    selected_position = st.sidebar.selectbox("Select Position", ["All Positions"] + list(positions))
    
    # Apply Filters
    filtered_df = df.copy()
    if selected_team != "All Clubs":
        filtered_df = filtered_df[filtered_df["team_name"] == selected_team]
    if selected_position != "All Positions":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]

    # Player Selection
    player_names = sorted(filtered_df["full_name"].tolist()) if "full_name" in filtered_df.columns else []
    
    st.sidebar.markdown("---")
    if len(player_names) > 0:
        selected_player_name = st.sidebar.selectbox("Select Player", player_names)
    else:
        selected_player_name = None
        st.sidebar.warning("No players match the selected filters.")

    if selected_player_name:
        player_data = filtered_df[filtered_df["full_name"] == selected_player_name].iloc[0]

        # Main Dashboard Layout - Key Metrics Grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Points", int(player_data.get("total_points", 0)))
        with col2:
            st.metric("Goals Scored", int(player_data.get("goals_scored", 0)))
        with col3:
            st.metric("Expected Goals (xG)", f"{player_data.get('expected_goals', 0):.2f}")

        st.subheader(f"Player Profile: {selected_player_name}")
        
        # Display player stats table/grid using correct columns
        stat_cols = ["team_name", "position", "minutes", "assists", "clean_sheets", "expected_assists", "market_value_m"]
        available_stats = {col.replace('_', ' ').title(): player_data.get(col, "N/A") for col in stat_cols if col in player_data}
        
        st.dataframe(pd.DataFrame([available_stats]), hide_index=True)

        # Live Prediction Section using trained ML models
        st.subheader(" Machine Learning Prediction Model")
        st.info("Powered by regression algorithms analyzing underlying metrics and historical performance indicators.")
        
        try:
            X_player = pd.DataFrame([player_data[ML_FEATURES]])
            predicted_points = model.predict(X_player)[0]
            st.success(f"**ML Predicted Total Points:** {predicted_points:.1f} pts")
        except Exception as model_err:
            minutes = player_data.get("minutes", 0)
            goals = player_data.get("goals_scored", 0)
            assists = player_data.get("assists", 0)
            predicted_points_est = (minutes * 0.05) + (goals * 4) + (assists * 3)
            st.warning("Model prediction unavailable. Showing fallback estimation index.")
            st.success(f"**Projected Fantasy Value Index:** {predicted_points_est:.1f} pts")

    # Data Explorer View
    st.markdown("---")
    with st.expander("View Complete Filtered Squad Table"):
        st.dataframe(filtered_df, use_container_width=True)