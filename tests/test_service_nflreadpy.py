"""
Unit tests for data_provider.service_nflreadpy cache functions.
No real nflreadpy or filesystem I/O — all mocked.
"""
import io
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.read()


SAMPLE_DF = pd.DataFrame([
    {
        "game_id": "2024_01_KC_LV",
        "home_team": "KC",
        "away_team": "LV",
        "week": 1,
        "posteam": "KC",
        "home_score": 7,
        "away_score": 0,
        "game_seconds_remaining": 3000,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 50,
        "desc": "some play",
        "score_diff": 7,
    },
    {
        "game_id": "2024_01_KC_LV",
        "home_team": "KC",
        "away_team": "LV",
        "week": 1,
        "posteam": "LV",
        "home_score": 7,
        "away_score": 7,
        "game_seconds_remaining": 0,
        "down": 4,
        "ydstogo": 1,
        "yardline_100": 30,
        "desc": "final play",
        "score_diff": 0,
    },
    {
        "game_id": "2024_02_BUF_MIA",
        "home_team": "BUF",
        "away_team": "MIA",
        "week": 2,
        "posteam": "BUF",
        "home_score": 24,
        "away_score": 17,
        "game_seconds_remaining": 0,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 50,
        "desc": "final play",
        "score_diff": 7,
    },
])


# ---------------------------------------------------------------------------
# Tests: cache_is_stale
# ---------------------------------------------------------------------------

class TestCacheIsStale(unittest.TestCase):

    @patch("data_provider.service_nflreadpy.cache_path")
    def test_stale_when_file_missing(self, mock_path):
        p = MagicMock(spec=Path)
        p.exists.return_value = False
        mock_path.return_value = p

        from data_provider.service_nflreadpy import cache_is_stale
        self.assertTrue(cache_is_stale(2024))

    @patch("data_provider.service_nflreadpy.cache_path")
    def test_fresh_when_recent(self, mock_path):
        p = MagicMock(spec=Path)
        p.exists.return_value = True
        # mtime = now → age = 0 → not stale
        p.stat.return_value = MagicMock(st_mtime=time.time())
        mock_path.return_value = p

        from data_provider.service_nflreadpy import cache_is_stale
        self.assertFalse(cache_is_stale(2024))

    @patch("data_provider.service_nflreadpy.cache_path")
    def test_stale_when_old(self, mock_path):
        p = MagicMock(spec=Path)
        p.exists.return_value = True
        # mtime = 25 hours ago
        p.stat.return_value = MagicMock(st_mtime=time.time() - 25 * 3600)
        mock_path.return_value = p

        from data_provider.service_nflreadpy import cache_is_stale
        self.assertTrue(cache_is_stale(2024, max_age_hours=24))


# ---------------------------------------------------------------------------
# Tests: available_games
# ---------------------------------------------------------------------------

class TestAvailableGames(unittest.TestCase):

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_returns_one_entry_per_game(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import available_games
        games = available_games(2024)
        self.assertEqual(len(games), 2)

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_team_names_correct(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import available_games
        games = available_games(2024)
        game_ids = {g["game_id"] for g in games}
        self.assertIn("2024_01_KC_LV", game_ids)
        self.assertIn("2024_02_BUF_MIA", game_ids)

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_week_numbers_correct(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import available_games
        games = {g["game_id"]: g for g in available_games(2024)}
        self.assertEqual(games["2024_01_KC_LV"]["week"], 1)
        self.assertEqual(games["2024_02_BUF_MIA"]["week"], 2)

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_final_scores_from_last_play(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import available_games
        games = {g["game_id"]: g for g in available_games(2024)}
        # KC vs LV — last play had 7-7
        self.assertEqual(games["2024_01_KC_LV"]["home_score"], 7)
        self.assertEqual(games["2024_01_KC_LV"]["away_score"], 7)
        # BUF vs MIA — 24-17
        self.assertEqual(games["2024_02_BUF_MIA"]["home_score"], 24)
        self.assertEqual(games["2024_02_BUF_MIA"]["away_score"], 17)


# ---------------------------------------------------------------------------
# Tests: load_game (cache read)
# ---------------------------------------------------------------------------

class TestLoadGame(unittest.TestCase):

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_filters_to_correct_game(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import load_game
        df = load_game(2024, "2024_01_KC_LV")
        self.assertTrue((df["game_id"] == "2024_01_KC_LV").all())
        self.assertEqual(len(df), 2)

    @patch("data_provider.service_nflreadpy.pd.read_parquet")
    def test_unknown_game_returns_empty(self, mock_read):
        mock_read.return_value = SAMPLE_DF.copy()

        from data_provider.service_nflreadpy import load_game
        df = load_game(2024, "2024_99_XX_YY")
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
