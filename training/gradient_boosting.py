import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import joblib

cols = ['game_id', 'posteam', 'home_team', 'away_team', 'total_home_score', 
        'total_away_score', 'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'result']
pbp = nfl.import_pbp_data([2023], columns=cols)

# 2. Data Cleaning
# Remove kickoffs, timeouts, and end of quarters (only keep plays with a 'down')
pbp = pbp.dropna(subset=['down'])

# Calculate Score Differential from the perspective of the possession team
pbp['score_diff'] = np.where(
    pbp['posteam'] == pbp['home_team'],
    pbp['total_home_score'] - pbp['total_away_score'],
    pbp['total_away_score'] - pbp['total_home_score']
)

pbp['posteam_win'] = np.where(
    ((pbp['posteam'] == pbp['home_team']) & (pbp['result'] > 0)) |
    ((pbp['posteam'] == pbp['away_team']) & (pbp['result'] < 0)),
    1, 0
)

# 3. Model Training
features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']
X = pbp[features]
y = pbp['posteam_win']

wp_model = GradientBoostingClassifier()
wp_model.fit(X, y)

print("Model Trained on 2023 Season Data!")

joblib.dump(wp_model, 'xgboost_wp.pkl')