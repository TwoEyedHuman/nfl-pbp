import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import nflreadpy as nfl
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache"))


# Our "Struct" for play-by-play data
class PBPData(NamedTuple):
    raw_df: pd.DataFrame
    team_map: dict
    color_map: dict


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(season: int) -> Path:
    """Return the parquet path for a given season."""
    return CACHE_DIR / f"pbp_{season}.parquet"


def cache_is_stale(season: int, max_age_hours: int = 24) -> bool:
    """Return True if the cache file is missing or older than max_age_hours."""
    path = cache_path(season)
    if not path.exists():
        return True
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds > max_age_hours * 3600


def fetch_and_cache_season(season: int) -> None:
    """Fetch play-by-play data via nflreadpy and write to parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(season)

    logger.info("Fetching nflreadpy PBP for season %d …", season)
    start = time.time()

    pbp = _fetch_nfl_data_raw(season)
    pbp.raw_df.to_parquet(path, index=False)

    elapsed = time.time() - start
    logger.info("Cached %s in %.1fs", path, elapsed)


def load_game(season: int, game_id: str) -> pd.DataFrame:
    """Read the season parquet and return rows for a single game."""
    df = pd.read_parquet(cache_path(season))
    return df[df["game_id"] == game_id].reset_index(drop=True)


def available_games(season: int) -> list[dict]:
    """
    Read the season parquet and return one dict per game with keys:
    game_id, home_team, away_team, week, home_score, away_score.
    Final scores are taken from the last play of each game.
    """
    df = pd.read_parquet(cache_path(season))

    # Last row per game has the highest (final) scores
    last = (
        df.groupby("game_id", sort=False)
        .last()
        .reset_index()
    )

    cols = [c for c in ["game_id", "home_team", "away_team", "week", "home_score", "away_score"] if c in last.columns]
    return last[cols].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Internal fetch (original logic, unchanged)
# ---------------------------------------------------------------------------

def _fetch_nfl_data_raw(year: int) -> PBPData:
    REQUIRED_COLUMNS = [
        'game_id', 'home_team', 'away_team', 'week', 'posteam',
        'total_home_score', 'total_away_score', 'game_seconds_remaining',
        'down', 'ydstogo', 'yardline_100', 'desc'
    ]

    full_df = nfl.load_pbp(year).to_pandas()
    available_cols = [c for c in REQUIRED_COLUMNS if c in full_df.columns]
    df = full_df[available_cols].copy()

    df = df.rename(columns={
        'total_home_score': 'home_score',
        'total_away_score': 'away_score'
    })

    df['score_diff'] = np.where(
        df['posteam'] == df['home_team'],
        df['home_score'] - df['away_score'],
        df['away_score'] - df['home_score']
    )

    teams_df = nfl.load_teams().to_pandas()
    team_map = dict(zip(teams_df['team_abbr'], teams_df['team_nick']))
    color_map = teams_df.set_index('team_abbr')[['team_color', 'team_color2']].to_dict('index')

    logger.info("Fetched NFLReadPy data for %d with columns: %s", year, df.columns.tolist())

    return PBPData(
        raw_df=df.copy(),
        team_map=team_map,
        color_map=color_map
    )


def fetch_nfl_data(year: int) -> PBPData:
    """
    Public API: returns PBPData for the given year.
    Reads from parquet cache when available and fresh; falls back to live fetch.
    """
    if cache_is_stale(year):
        fetch_and_cache_season(year)

    df = pd.read_parquet(cache_path(year))
    teams_df = nfl.load_teams().to_pandas()
    team_map = dict(zip(teams_df['team_abbr'], teams_df['team_nick']))
    color_map = teams_df.set_index('team_abbr')[['team_color', 'team_color2']].to_dict('index')

    return PBPData(raw_df=df, team_map=team_map, color_map=color_map)
