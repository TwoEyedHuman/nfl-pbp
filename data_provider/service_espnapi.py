import time
import requests
import pandas as pd
from typing import Any, Dict, List
from .service_nflreadpy import PBPData
import numpy as np

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

LIVE_TTL = 60
COMPLETED_TTL = 300

_cache: dict = {}  # {key: (data, timestamp)}
_completed_keys: set = set()


def _get_cached(key: str, ttl: float) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    data, timestamp = entry
    if time.time() - timestamp < ttl:
        return data
    return None


def _set_cached(key: str, data: Any) -> None:
    _cache[key] = (data, time.time())


def cache_age_seconds(key: str) -> float | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    _, timestamp = entry
    return time.time() - timestamp

def calculate_seconds(play: dict) -> int:
    """
    Converts ESPN period and clock string to total game seconds remaining.
    """
    period = play.get('period', {}).get('number', 1)
    clock_str = play.get('clock', {}).get('displayValue', "00:00")
    
    try:
        minutes, seconds = map(int, clock_str.split(':'))
    except (ValueError, AttributeError):
        minutes, seconds = 0, 0

    quarter_seconds = (minutes * 60) + seconds

    if period <= 4:
        game_seconds_remaining = ((4 - period) * 900) + quarter_seconds
    else:
        game_seconds_remaining = 0
        
    return max(0, game_seconds_remaining)

def get_scoreboard() -> List[Dict]:
    key = "scoreboard"
    cached = _get_cached(key, LIVE_TTL)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/scoreboard"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    events = []
    for event in data.get('events', []):
        events.append({
            'id': event['id'],
            'name': event['name'],
            'shortName': event['shortName']
        })

    _set_cached(key, events)
    return events

# Custom error for game not started
class GameNotStartedError(Exception):
    pass

def fetch_espn_pbp(event_id: str) -> PBPData:
    key = f"pbp_{event_id}"
    ttl = COMPLETED_TTL if key in _completed_keys else LIVE_TTL
    cached = _get_cached(key, ttl)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/summary?event={event_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Safely navigate the nested list
    header = data.get('header', {})
    competitions = header.get('competitions', [{}])
    game_status = competitions[0].get('status', {}).get('type', {}).get('name', '')

    if game_status == "STATUS_SCHEDULED":
        raise GameNotStartedError("The selected game has not started yet.")

    # 1. Identify Home Team Metadata first
    # This avoids the NameError by defining home_id before the loop
    competitors = data.get('boxscore', {}).get('teams', [])
    home_team_info = next(c['team'] for c in competitors if c['homeAway'] == 'home')
    home_id = home_team_info.get('id')
    home_abbr = home_team_info.get('abbreviation')
    away_team_info = next(c['team'] for c in competitors if c['homeAway'] == 'away')
    away_abbr = away_team_info.get('abbreviation')

    # 2. Extract Team Metadata for the struct
    team_map = {}
    color_map = {}
    id_to_abbr = {} # Helper to map numeric IDs to strings (e.g., "17" -> "NE")
    
    for comp in competitors:
        team = comp.get('team', {})
        t_id = str(team.get('id'))
        t_abbr = team.get('abbreviation')
        
        id_to_abbr[t_id] = t_abbr
        team_map[t_abbr] = team.get('nickname', team.get('name'))
        color_map[t_abbr] = {
            'team_color': f"#{team.get('color', '000000')}",
            'team_color2': f"#{team.get('alternateColor', 'FFFFFF')}"
        }

    # 3. Extract Plays
    all_plays = []
    drives = data.get('drives', {}).get('previous', [])
    
    for drive in drives:
        for play in drive.get('plays', []):
            if play.get('type').get('text') == 'Official Timeout':
                continue
            posteam_id = str(play.get('start', {}).get('team', {}).get('id'))

            play_row = {
                'game_id': event_id,
                'desc': play.get('text'),
                'home_score': play.get('homeScore'),
                'away_score': play.get('awayScore'),
                'period': play.get('period', {}).get('number'),
                'game_seconds_remaining': calculate_seconds(play),
                'down': play.get('start', {}).get('down'),
                'ydstogo': play.get('start', {}).get('distance'),
                'yardline_100': play.get('start', {}).get('yardsToEndzone'),
                'home_team': home_abbr,
                'away_team': away_abbr,
                'posteam': id_to_abbr.get(posteam_id)
            }
            all_plays.append(play_row)
            
    df = pd.DataFrame(all_plays)

    # 4. Calculate score_diff (Now that posteam and home_team are strings)
    if not df.empty:
        # ESPN scores are post-play; nflfastR training data uses pre-play scores.
        # Shift by 1 so each play sees the score *entering* the play, matching training.
        df['home_score'] = df['home_score'].shift(1).fillna(0).astype(int)
        df['away_score'] = df['away_score'].shift(1).fillna(0).astype(int)

        df['score_diff'] = np.where(
            df['posteam'] == df['home_team'],
            df['home_score'] - df['away_score'],
            df['away_score'] - df['home_score']
        )

    result = PBPData(
        raw_df=df,
        team_map=team_map,
        color_map=color_map
    )

    if game_status.startswith("STATUS_FINAL"):
        _completed_keys.add(key)
    _set_cached(key, result)
    return result