import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mplcursors

# 1. Fetch data for a specific game (2023 Super Bowl ID: 2023_22_SF_KC)
game_id = '2023_13_CAR_TB'
game_data = nfl.import_pbp_data([2023])
game_df = game_data[game_data['game_id'] == game_id].copy()

game_df['score_diff'] = np.where(
    game_df['possession_team'] == game_df['home_team'],
    game_df['total_home_score'] - game_df['total_away_score'],
    game_df['total_away_score'] - game_df['total_home_score']
)

# Filter out non-plays (kickoffs, timeouts) for a cleaner graph
graph_df = game_df.dropna(subset=['down']).copy()

features = ['game_seconds_remaining', 'down', 'ydstogo', 'yardline_100', 'score_diff']
wp_model = joblib.load('nfl_wp_model.pkl')
graph_df['possession_wp'] = wp_model.predict_proba(graph_df[features])[:, 1]

graph_df['panthers_wp'] = np.where(
    graph_df['possession_team'] == 'CAR',
    graph_df['possession_wp'],
    1 - graph_df['possession_wp']
)
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(graph_df['game_seconds_remaining'], graph_df['panthers_wp'], color="#247CE1", linewidth=1.5, alpha=0.7)


sc = ax.scatter(graph_df['game_seconds_remaining'], graph_df['panthers_wp'], 
                color='#247CE1', s=20, edgecolors='white', linewidth=0.5, zorder=3)

cursor = mplcursors.cursor(sc, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    play_desc = graph_df.iloc[idx]['desc']
    wp_val = graph_df.iloc[idx]['panthers_wp']
    
    wrapped_desc = "\n".join([play_desc[i:i+50] for i in range(0, len(play_desc), 50)])
    
    sel.annotation.set_text(f"WP: {wp_val:.1%}\n---\n{wrapped_desc}")
    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round")
ax.invert_xaxis()
ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
ax.set_title("Panthers Win Probability (Hover for Play Details)", fontsize=14)
ax.set_ylabel("Panthers Win %")
ax.set_xlabel("Seconds Remaining")
plt.grid(alpha=0.2)

plt.show()