import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
import plotly.express as px

# Import both services
from data_provider.service_nflreadpy import fetch_nfl_data
from data_provider.service_espnapi import fetch_espn_pbp, get_scoreboard

# --- Page Config ---
st.set_page_config(page_title="NFL Win Probability Tracker", layout="wide")
st.title("🏈 NFL Play-by-Play Win Probability")

# --- Model Selection Logic ---
def get_available_models(model_dir="models"):
    path = Path(model_dir)
    if not path.exists(): return ["xgboost_wp.pkl"] 
    return [f.name for f in path.glob("*.pkl")]

@st.cache_resource
def load_selected_model(model_name):
    return joblib.load(f'models/{model_name}')

# --- Sidebar: Settings ---
st.sidebar.header("Global Settings")
selected_model_name = st.sidebar.selectbox("Select Model", options=get_available_models())
wp_model = load_selected_model(selected_model_name)

st.sidebar.divider()
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Source", ["NFLReadPy (Historical)", "ESPN API (Live/Recent)"])

# --- Logic: Handle Source-Specific Selectors ---
pbp_struct = None

if data_source == "NFLReadPy (Historical)":
    selected_year = st.sidebar.selectbox("Year", options=range(2025, 2019, -1))
    
    @st.cache_data
    def get_cached_nfl_pbp(year):
        return fetch_nfl_data(year)
    
    with st.spinner("Fetching NFLReadPy Data..."):
        pbp_struct = get_cached_nfl_pbp(selected_year)

else:
    # # ESPN Source Logic
    # col1, col2 = st.sidebar.columns(2)
    # with col1:
    #     e_year = st.selectbox("Year", options=[2025, 2024], index=0)
    # with col2:
    #     e_week = st.selectbox("Week", options=range(1, 19), index=0)
        
    @st.cache_data
    def get_cached_espn_scoreboard():
        return get_scoreboard()

    games = get_cached_espn_scoreboard()
    selected_game = st.sidebar.selectbox(
        "Select Game", 
        options=games, 
        format_func=lambda x: x['name']
    )

    if selected_game:
        with st.spinner("Fetching ESPN Live Data..."):
            pbp_struct = fetch_espn_pbp(selected_game['id'])

# --- Main App Logic (Shared for both sources) ---
if pbp_struct:
    year_data = pbp_struct.raw_df
    team_map = pbp_struct.team_map
    all_team_colors = pbp_struct.color_map

    # 1. Select Team
    # Handle cases where year_data might be empty or missing columns
    if 'posteam' not in year_data.columns and 'home_team' in year_data.columns:
         # Some ESPN games might need posteam derived if not in raw_df
         pass 

    all_teams = sorted(year_data['home_team'].unique())
    selected_team = st.sidebar.selectbox("Team", options=all_teams)

    # 2. Filter Game Data
    # For NFLReadPy, we filter by game_id. For ESPN, we already have the specific game.
    if data_source == "NFLReadPy (Historical)":
        team_games = year_data[(year_data['home_team'] == selected_team) | (year_data['away_team'] == selected_team)].copy()
        team_games['game_label'] = team_games['game_id'] + " (" + team_games['away_team'] + " @ " + team_games['home_team'] + ")"
        game_options = team_games['game_label'].unique()
        selected_game_label = st.sidebar.selectbox("Game", options=game_options)
        selected_game_id = selected_game_label.split(" (")[0]
        game_df = year_data[year_data['game_id'] == selected_game_id].copy()
    else:
        # ESPN data is already game-specific
        game_df = year_data.copy()

    if 'score_diff' not in game_df.columns:
        st.error("Data source failed to provide 'score_diff'. Check service implementation. Columns: " + ", ".join(game_df.columns))
        st.stop()

    # Inference Prep
    graph_df = game_df.dropna(subset=['down', 'posteam', 'score_diff']).copy()
    features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']
    
    # Run Prediction
    graph_df['possession_wp'] = wp_model.predict_proba(graph_df[features])[:, 1]
    graph_df['team_wp'] = np.where(
        graph_df['posteam'] == selected_team,
        graph_df['possession_wp'],
        1 - graph_df['possession_wp']
    )

    # --- Visualization ---
    primary_color = all_team_colors.get(selected_team, {}).get('team_color', '#247CE1')
    
    # Text Setup
    home_abbr = game_df['home_team'].iloc[0] if 'home_team' in game_df.columns else "Home"
    away_abbr = game_df['away_team'].iloc[0] if 'away_team' in game_df.columns else "Away"
    vs_text = "v" if selected_team == home_abbr else "@"
    opponent_abbr = away_abbr if selected_team == home_abbr else home_abbr
    
    clean_title = f"{team_map.get(selected_team, selected_team)} Win Probability {vs_text} {team_map.get(opponent_abbr, opponent_abbr)}"

    fig = px.line(graph_df, x='game_seconds_remaining', y='team_wp', title=clean_title, custom_data=['desc'])
    fig.update_traces(line_color=primary_color, hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        yaxis_tickformat='.0%', yaxis_range=[0, 1],
        xaxis=dict(tickvals=[3600, 2700, 1800, 900, 0], ticktext=['Start', 'Q2', 'Half', 'Q4', 'Final'], autorange="reversed"),
        hovermode="closest"
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Play-by-Play Details")
    st.dataframe(graph_df[['game_seconds_remaining', 'down', 'ydstogo', 'desc', 'team_wp']].sort_values('game_seconds_remaining', ascending=False))

else:
    st.info("Please select a game from the sidebar to view Win Probability.")