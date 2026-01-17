import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import streamlit as st
from pathlib import Path

# --- Page Config ---
st.set_page_config(page_title="NFL Win Probability Tracker", layout="wide")
st.title("🏈 NFL Play-by-Play Win Probability")

# --- Model Selection Logic ---
def get_available_models(model_dir="models"):
    """Scans the models folder for .pkl files."""
    path = Path(model_dir)
    if not path.exists():
        # Fallback if folder doesn't exist yet
        return ["xgboost_wp.pkl"] 
    return [f.name for f in path.glob("*.pkl")]

@st.cache_resource
def load_selected_model(model_name):
    return joblib.load(f'models/{model_name}')

# --- Sidebar Selectors ---
st.sidebar.header("Settings")

# 1. Select Model
available_models = get_available_models()
selected_model_name = st.sidebar.selectbox("Select Model", options=available_models)
wp_model = load_selected_model(selected_model_name)

st.sidebar.divider() # Visual separation
st.sidebar.header("Select Game")

# wp_model = load_model()

# --- Sidebar Selectors ---
st.sidebar.header("Select Game")

# 1. Select Year
selected_year = st.sidebar.selectbox("Year", options=range(2023, 2019, -1))

# 2. Fetch Data (Cached to avoid re-downloading on every click)
@st.cache_data
def get_year_data(year):
    return nfl.import_pbp_data([year])

with st.spinner(f"Loading {selected_year} data..."):
    year_data = get_year_data(selected_year)

# 3. Select Team
all_teams = sorted(year_data['home_team'].unique())
selected_team = st.sidebar.selectbox("Team", options=all_teams)

# 4. Select Game
team_games = year_data[
    (year_data['home_team'] == selected_team) | 
    (year_data['away_team'] == selected_team)
].copy()

# Create a readable label for the dropdown
team_games['game_label'] = team_games['game_id'] + " (" + team_games['away_team'] + " @ " + team_games['home_team'] + ")"
game_options = team_games['game_label'].unique()
selected_game_label = st.sidebar.selectbox("Game", options=game_options)

# Extract the actual game_id
selected_game_id = selected_game_label.split(" (")[0]

# --- Processing ---
game_df = year_data[year_data['game_id'] == selected_game_id].copy()

# Feature Engineering
game_df['score_diff'] = np.where(
    game_df['possession_team'] == game_df['home_team'],
    game_df['total_home_score'] - game_df['total_away_score'],
    game_df['total_away_score'] - game_df['total_home_score']
)

# Clean for graph
graph_df = game_df.dropna(subset=['down']).copy()
features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']

# Run Inference
graph_df['possession_wp'] = wp_model.predict_proba(graph_df[features])[:, 1]

# Calculate WP for the SELECTED team specifically
graph_df['team_wp'] = np.where(
    graph_df['possession_team'] == selected_team,
    graph_df['possession_wp'],
    1 - graph_df['possession_wp']
)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(graph_df['game_seconds_remaining'], graph_df['team_wp'], color="#247CE1", linewidth=2)
ax.fill_between(graph_df['game_seconds_remaining'], 0.5, graph_df['team_wp'], 
                where=(graph_df['team_wp'] >= 0.5), color='green', alpha=0.1)
ax.fill_between(graph_df['game_seconds_remaining'], 0.5, graph_df['team_wp'], 
                where=(graph_df['team_wp'] < 0.5), color='red', alpha=0.1)

ax.invert_xaxis()
ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
ax.set_title(f"{selected_team} Win Probability vs {selected_game_id}", fontsize=16)
ax.set_ylabel(f"{selected_team} Win %")
ax.set_xlabel("Seconds Remaining")
plt.grid(alpha=0.3)

# Display in Streamlit
st.pyplot(fig)

# Show Play-by-Play Table
st.subheader("Play-by-Play Details")
st.dataframe(graph_df[['game_seconds_remaining', 'down', 'ydstogo', 'desc', 'team_wp']].sort_values('game_seconds_remaining', ascending=False))