import nfl_data_py as nfl
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# Create Target: Did the possession team win?
# 'result' is (home_score - away_score). 
# If result > 0 and home has ball OR result < 0 and away has ball, team wins.
pbp['posteam_win'] = np.where(
    ((pbp['posteam'] == pbp['home_team']) & (pbp['result'] > 0)) |
    ((pbp['posteam'] == pbp['away_team']) & (pbp['result'] < 0)),
    1, 0
)

# 3. Model Training
features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']
X = pbp[features]
y = pbp['posteam_win']

wp_model = LogisticRegression()
wp_model.fit(X, y)

print("Model Trained on 2023 Season Data!")

# Save the model to a file
joblib.dump(wp_model, 'nfl_wp_model.pkl')

# Later, in a different script, you can load it instantly:
# wp_model = joblib.load('nfl_wp_model.pkl')