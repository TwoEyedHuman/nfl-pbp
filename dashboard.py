import nfl_data_py as nfl
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
import plotly.express as px

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

# 1. Select Year
selected_year = st.sidebar.selectbox("Year", options=range(2023, 2019, -1))

# 2. Fetch Data (Cached to avoid re-downloading on every click)
@st.cache_data(persist="disk")
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

@st.cache_data(persist="disk")
def get_team_map():
    teams = nfl.import_team_desc()
    return dict(zip(teams['team_abbr'], teams['team_nick']))

team_map = get_team_map()

week = int(game_df['week'].iloc[0])
home_abbr = game_df['home_team'].iloc[0]
away_abbr = game_df['away_team'].iloc[0]

if selected_team == home_abbr:
    opponent_abbr = away_abbr
    vs_text = "v"
else:
    opponent_abbr = home_abbr
    vs_text = "@"

selected_nick = team_map.get(selected_team, selected_team)
opponent_nick = team_map.get(opponent_abbr, opponent_abbr)

clean_title = f"{selected_nick} Win Probability - Week {week}, {selected_year} {vs_text} {opponent_nick}"

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

@st.cache_data
def get_team_colors_map():
    teams = nfl.import_team_desc()
    # Create a dictionary mapping abbreviation to a dictionary of colors
    return teams.set_index('team_abbr')[['team_color', 'team_color2']].to_dict('index')

all_team_colors = get_team_colors_map()
primary_color = all_team_colors.get(selected_team, {}).get('team_color', '#247CE1')
secondary_color = all_team_colors.get(selected_team, {}).get('team_color2', '#FFFFFF')

def hex_to_rgba(hex_code, opacity=0.1):
    hex_code = hex_code.lstrip('#')
    lv = len(hex_code)
    rgb = tuple(int(hex_code[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})'

bg_color = hex_to_rgba(secondary_color, opacity=0.15)

# --- Interactive Visualization with Plotly ---
# Create the interactive line chart
fig = px.line(
    graph_df, 
    x='game_seconds_remaining', 
    y='team_wp',
    title=clean_title,
    custom_data=['desc']
)

fig.update_traces(
    line=dict(color=primary_color, width=4),
    hovertemplate="%{customdata[0]}<extra></extra>")

fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    yaxis_tickformat='.0%',
    yaxis_range=[0, 1],
    plot_bgcolor=bg_color,
    xaxis=dict(
        tickmode='array',
        tickvals=[3600, 2700, 1800, 900, 0],
        ticktext=['Start', 'End Q1', 'Half', 'End Q3', 'Final'],
        autorange="reversed"
    ),
    hovermode="closest"
)

# Add the 50% baseline
fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)

# Display in Streamlit
st.plotly_chart(fig, width='stretch')

# Show Play-by-Play Table
st.subheader("Play-by-Play Details")
st.dataframe(graph_df[['game_seconds_remaining', 'down', 'ydstogo', 'desc', 'team_wp']].sort_values('game_seconds_remaining', ascending=False))