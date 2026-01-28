import nflreadpy as nfl
import pandas as pd
import numpy as np

cols = ['game_id', 'posteam', 'home_team', 'away_team', 'total_home_score',
        'total_away_score', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'result']


def load_training_data():
    pbp = nfl.load_pbp([2020,2021,2022,2023]).select(cols).to_pandas()


    # 2. Data Cleaning
    # Remove kickoffs, timeouts, and end of quarters (only keep plays with a 'down')
    pbp = pbp.dropna(subset=['down'])

    # Calculate Score Differential from the perspective of the possession team
    pbp['score_diff'] = np.where(
        pbp['posteam'] == pbp['home_team'],
        pbp['total_home_score'] - pbp['total_away_score'],
        pbp['total_away_score'] - pbp['total_home_score']
    )

    # Create Target: Did the possession team win?
    # 'result' is (home_score - away_score). 
    # If result > 0 and home has ball OR result < 0 and away has ball, team wins.
    pbp['posteam_win'] = np.where(
        ((pbp['posteam'] == pbp['home_team']) & (pbp['result'] > 0)) |
        ((pbp['posteam'] == pbp['away_team']) & (pbp['result'] < 0)),
        1, 0
    )
    
    features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']
    X = pbp[features]
    y = pbp['posteam_win']

    return X, y