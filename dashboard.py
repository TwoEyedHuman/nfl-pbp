import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
import plotly.express as px

# Import both services
from data_provider.service_nflreadpy import fetch_nfl_data
from data_provider.service_espnapi import GameNotStartedError, fetch_espn_pbp, get_scoreboard

# --- Page Config ---
st.set_page_config(page_title="NFL Win Probability Tracker", layout="wide")

# --- Helper Functions ---
@st.cache_data
def get_cached_nfl_pbp(year):
    return fetch_nfl_data(year)

@st.cache_data
def get_cached_espn_scoreboard():
    return get_scoreboard()

@st.cache_resource
def load_selected_model(model_name):
    return joblib.load(f'models/{model_name}')

def get_available_models(model_dir="models"):
    path = Path(model_dir)
    if not path.exists(): return ["xgboost_wp.pkl"] 
    return [f.name for f in path.glob("*.pkl")]

def update_url_params():
    params_to_sync = ['source', 'team', 'year', 'game_id']
    for param in params_to_sync:
        if param in st.session_state:
            st.query_params[param] = st.session_state[param]

# --- Initialize State from URL ---
if "source" not in st.session_state:
    st.session_state.source = st.query_params.get("source", "NFLReadPy (Historical)")
if "team" not in st.session_state:
    st.session_state.team = st.query_params.get("team", None)

def main():
    st.title("🏈 NFL Play-by-Play Win Probability")

    # --- Sidebar: Settings & Persistence ---
    st.sidebar.header("Global Settings")
    selected_model_name = st.sidebar.selectbox("Select Model", options=get_available_models())
    wp_model = load_selected_model(selected_model_name)

    st.sidebar.divider()
    st.sidebar.header("Data Source")
    
    # 1. Define 'source' first
    source = st.sidebar.radio(
        "Select Source", 
        ["NFLReadPy (Historical)", "ESPN API (Live/Recent)"],
        key="source",
        on_change=update_url_params
    )

    pbp_struct = None
    selected_team = None

    if source == "NFLReadPy (Historical)":
        # 1. Year Selection
        saved_year = int(st.query_params.get("year", 2025))
        year_options = list(range(2025, 2019, -1))
        year_idx = year_options.index(saved_year) if saved_year in year_options else 0
        
        selected_year = st.sidebar.selectbox(
            "Year", options=year_options, index=year_idx, key="year", on_change=update_url_params
        )
        
        with st.spinner("Fetching NFLReadPy Data..."):
            pbp_struct = get_cached_nfl_pbp(selected_year)

        # 2. Team Selection
        if pbp_struct:
            all_teams = sorted(pbp_struct.raw_df['home_team'].unique())
            saved_team = st.query_params.get("team")
            team_idx = all_teams.index(saved_team) if saved_team in all_teams else 0
            
            selected_team = st.sidebar.selectbox(
                "Team", options=all_teams, index=team_idx, key="team", on_change=update_url_params
            )

    else: # ESPN Source
        games = get_cached_espn_scoreboard()
        game_ids = [g['id'] for g in games]
        
        saved_game_id = st.query_params.get("game_id")
        game_idx = game_ids.index(saved_game_id) if saved_game_id in game_ids else 0
        
        selected_game_obj = st.sidebar.selectbox(
            "Select Game", 
            options=games,
            index=game_idx,
            format_func=lambda x: x['name'],
            key="game_id_obj"
        )

        if selected_game_obj:
            st.session_state.game_id = selected_game_obj['id']
            update_url_params()
            
            with st.spinner("Fetching ESPN Live Data..."):
                try:
                    pbp_struct = fetch_espn_pbp(selected_game_obj['id'])
                except GameNotStartedError as e:
                    st.sidebar.warning(str(e))
                    pbp_struct = None
            
            if pbp_struct:
                all_teams = sorted(pbp_struct.raw_df['home_team'].unique())
                default_team = pbp_struct.raw_df['home_team'].iloc[0]
                saved_team = st.query_params.get("team", default_team)
                team_idx = all_teams.index(saved_team) if saved_team in all_teams else 0
                
                selected_team = st.sidebar.selectbox(
                    "Team", options=all_teams, index=team_idx, key="team", on_change=update_url_params
                )

    # --- Main App Logic (Shared) ---
    if pbp_struct and selected_team:
        year_data = pbp_struct.raw_df
        team_map = pbp_struct.team_map
        all_team_colors = pbp_struct.color_map

        # Filter Game Data
        if source == "NFLReadPy (Historical)":
            team_games = year_data[(year_data['home_team'] == selected_team) | (year_data['away_team'] == selected_team)].copy()
            
            if not team_games.empty:
                team_games['game_label'] = team_games['game_id'] + " (" + team_games['away_team'] + " @ " + team_games['home_team'] + ")"
                game_options = sorted(team_games['game_label'].unique())
                
                saved_gid = st.query_params.get("game_id")
                game_idx = 0
                if saved_gid:
                    for i, opt in enumerate(game_options):
                        if opt.startswith(saved_gid):
                            game_idx = i
                            break
                
                selected_game_label = st.sidebar.selectbox(
                    "Game", 
                    options=game_options, 
                    index=game_idx,
                    key="nfl_game_select"
                )
                
                if selected_game_label:
                    selected_game_id = selected_game_label.split(" (")[0]
                    st.session_state.game_id = selected_game_id
                    update_url_params()
                    game_df = year_data[year_data['game_id'] == selected_game_id].copy()
                else:
                    st.warning(f"No specific game selected for {selected_team}.")
                    st.stop()
            else:
                st.warning(f"No games found for {selected_team} in {selected_year}.")
                st.stop()
        else:
            # ESPN data is already game-specific
            game_df = year_data.copy()

        if 'score_diff' not in game_df.columns:
            st.error("Data source failed to provide 'score_diff'.")
            st.stop()

        game_df['score_diff'] = np.where(
            game_df['posteam'] == game_df['home_team'],
            game_df['home_score'] - game_df['away_score'],
            game_df['away_score'] - game_df['home_score']
        )

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
        
        home_abbr = game_df['home_team'].iloc[0] if 'home_team' in game_df.columns else "Home"
        away_abbr = game_df['away_team'].iloc[0]
        vs_text = "v" if selected_team == home_abbr else "@"
        opponent_abbr = away_abbr if selected_team == home_abbr else home_abbr
        
        clean_title = f"{team_map.get(selected_team, selected_team)} Win Probability {vs_text} {team_map.get(opponent_abbr, opponent_abbr)}"

        fig = px.line(graph_df, x='game_seconds_remaining', y='team_wp', title=clean_title, custom_data=['desc'])
        fig.update_traces(line_color=primary_color, hovertemplate="%{customdata[0]}<extra></extra>")
        fig.update_layout(
            yaxis_tickformat='.0%', yaxis_range=[0, 1],
            xaxis=dict(tickvals=[3600, 2700, 1800, 900, 0], ticktext=['Start', 'Q2', 'Half', 'Q4', 'Final'], autorange="reversed"),
            hovermode="closest",
            xaxis_title="",
            yaxis_title=""
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Play-by-Play Details")
        graph_df.rename(columns={'desc': 'Play',
                                  'game_seconds_remaining': 'Seconds Left',
                                  'down': 'Down',
                                  'ydstogo': 'Distance',
                                  'team_wp': 'Win Probability'}, inplace=True)
        st.dataframe(graph_df[['Seconds Left', 'Down', 'Distance', 'Play', 'Win Probability']].sort_values('Seconds Left', ascending=False))

    else:
        st.info("Please select a game from the sidebar to view Win Probability.")

if __name__ == "__main__":
    main()