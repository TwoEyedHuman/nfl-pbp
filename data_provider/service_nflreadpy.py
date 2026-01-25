import nflreadpy as nfl
import pandas as pd
from typing import NamedTuple
import numpy as np

# Our "Struct" for play-by-play data
class PBPData(NamedTuple):
    raw_df: pd.DataFrame
    team_map: dict
    color_map: dict

def fetch_nfl_data(year: int) -> PBPData:
    """
    Sourcing method that encapsulates all nflreadpy calls.
    Returns a PBPData struct containing the DataFrame and metadata.
    """
    REQUIRED_COLUMNS = [
        'game_id', 'home_team', 'away_team', 'week', 'posteam',
        'total_home_score', 'total_away_score', 'game_seconds_remaining',
        'down', 'ydstogo', 'yardline_100', 'desc'
    ]

    full_df = nfl.load_pbp(year).to_pandas()
    available_cols = [c for c in REQUIRED_COLUMNS if c in full_df.columns]
    df = full_df[available_cols].copy()

    # 1. Normalize Names immediately
    df = df.rename(columns={
        'total_home_score': 'home_score', 
        'total_away_score': 'away_score'
    })
    
    # 2. Calculate score_diff (Must happen before returning)
    df['score_diff'] = np.where(
        df['posteam'] == df['home_team'],
        df['home_score'] - df['away_score'],
        df['away_score'] - df['home_score']
    )

    teams_df = nfl.load_teams().to_pandas()
    
    team_map = dict(zip(teams_df['team_abbr'], teams_df['team_nick']))
    color_map = teams_df.set_index('team_abbr')[['team_color', 'team_color2']].to_dict('index')

    # log out the column names

    print(f"Fetched NFLReadPy data for {year} with columns: {df.columns.tolist()}")
    
    return PBPData(
        raw_df=df.copy(),
        team_map=team_map,
        color_map=color_map
    )