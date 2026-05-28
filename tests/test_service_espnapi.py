import json
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub out streamlit before importing the module under test
import sys
st_stub = types.ModuleType("streamlit")
st_stub.cache_data = lambda **kw: (lambda f: f)
sys.modules.setdefault("streamlit", st_stub)

import data_provider.service_espnapi as svc
from data_provider.service_espnapi import fetch_espn_pbp, LIVE_TTL, COMPLETED_TTL

SCORING_PLAY_RESPONSE = {
    "header": {
        "competitions": [{"status": {"type": {"name": "STATUS_IN_PROGRESS"}}}]
    },
    "boxscore": {
        "teams": [
            {
                "homeAway": "home",
                "team": {"id": "1", "abbreviation": "KC", "nickname": "Chiefs", "color": "E31837", "alternateColor": "FFB81C"}
            },
            {
                "homeAway": "away",
                "team": {"id": "2", "abbreviation": "LV", "nickname": "Raiders", "color": "000000", "alternateColor": "A5ACAF"}
            }
        ]
    },
    "drives": {
        "previous": [
            {
                "plays": [
                    {
                        "type": {"text": "Touchdown"},
                        "text": "P.Mahomes 5 yard TD pass",
                        "homeScore": 7,
                        "awayScore": 0,
                        "period": {"number": 1},
                        "clock": {"displayValue": "10:00"},
                        # start.* reflects pre-snap state (correct)
                        "start": {
                            "team": {"id": "1"},
                            "down": 1,
                            "distance": 5,
                            "yardsToEndzone": 5
                        },
                        # end.* reflects post-play state — ESPN returns -1/0 on scores
                        "end": {
                            "team": {"id": "1"},
                            "down": -1,
                            "distance": 0,
                            "yardsToEndzone": 0
                        }
                    }
                ]
            }
        ]
    }
}


class TestFetchEspnPbpStartFields(unittest.TestCase):
    def _mock_response(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = SCORING_PLAY_RESPONSE
        return resp

    @patch("data_provider.service_espnapi.requests.get")
    def test_scoring_play_down_not_minus_one(self, mock_get):
        mock_get.return_value = self._mock_response()
        result = fetch_espn_pbp("test_event")
        df = result.raw_df
        self.assertFalse((df["down"] == -1).any(), "down should never be -1 (end.* leaking into output)")

    @patch("data_provider.service_espnapi.requests.get")
    def test_scoring_play_yardline_not_zero(self, mock_get):
        mock_get.return_value = self._mock_response()
        result = fetch_espn_pbp("test_event")
        df = result.raw_df
        self.assertFalse((df["yardline_100"] == 0).any(), "yardline_100 should never be 0 on a scoring play")

    @patch("data_provider.service_espnapi.requests.get")
    def test_scoring_play_down_in_valid_range(self, mock_get):
        mock_get.return_value = self._mock_response()
        result = fetch_espn_pbp("test_event")
        df = result.raw_df
        valid = df["down"].between(1, 4)
        self.assertTrue(valid.all(), f"all downs should be 1-4, got: {df['down'].tolist()}")

    @patch("data_provider.service_espnapi.requests.get")
    def test_scoring_play_yardline_in_valid_range(self, mock_get):
        mock_get.return_value = self._mock_response()
        result = fetch_espn_pbp("test_event")
        df = result.raw_df
        valid = df["yardline_100"].between(1, 99)
        self.assertTrue(valid.all(), f"yardline_100 should be 1-99, got: {df['yardline_100'].tolist()}")

    @patch("data_provider.service_espnapi.requests.get")
    def test_away_team_present_and_non_null(self, mock_get):
        mock_get.return_value = self._mock_response()
        result = fetch_espn_pbp("test_event")
        df = result.raw_df
        self.assertIn("away_team", df.columns, "away_team column must be present")
        self.assertFalse(df["away_team"].isnull().any(), "away_team must be non-null on every row")
        self.assertTrue((df["away_team"] == "LV").all(), "away_team should be 'LV'")


class TestEspnApiCache(unittest.TestCase):
    def setUp(self):
        svc._cache.clear()
        svc._completed_keys.clear()

    def _mock_response(self, response_data=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = response_data or SCORING_PLAY_RESPONSE
        return resp

    @patch("data_provider.service_espnapi.requests.get")
    def test_cache_hit_second_call_skips_api(self, mock_get):
        mock_get.return_value = self._mock_response()
        fetch_espn_pbp("evt1")
        fetch_espn_pbp("evt1")
        self.assertEqual(mock_get.call_count, 1)

    @patch("data_provider.service_espnapi.requests.get")
    def test_cache_miss_after_live_ttl(self, mock_get):
        mock_get.return_value = self._mock_response()
        fetch_espn_pbp("evt2")
        # Backdate the cached timestamp past LIVE_TTL
        data, _ = svc._cache["pbp_evt2"]
        svc._cache["pbp_evt2"] = (data, time.time() - LIVE_TTL - 1)
        fetch_espn_pbp("evt2")
        self.assertEqual(mock_get.call_count, 2)

    @patch("data_provider.service_espnapi.requests.get")
    def test_completed_game_uses_longer_ttl(self, mock_get):
        completed_response = {
            "header": {
                "competitions": [{"status": {"type": {"name": "STATUS_FINAL"}}}]
            },
            "boxscore": SCORING_PLAY_RESPONSE["boxscore"],
            "drives": SCORING_PLAY_RESPONSE["drives"],
        }
        mock_get.return_value = self._mock_response(completed_response)
        fetch_espn_pbp("evt3")
        # Backdate past LIVE_TTL but within COMPLETED_TTL
        data, _ = svc._cache["pbp_evt3"]
        svc._cache["pbp_evt3"] = (data, time.time() - LIVE_TTL - 1)
        fetch_espn_pbp("evt3")
        # Should still be cached (game completed, COMPLETED_TTL not yet elapsed)
        self.assertEqual(mock_get.call_count, 1)

    @patch("data_provider.service_espnapi.requests.get")
    def test_cache_age_seconds_correct(self, mock_get):
        mock_get.return_value = self._mock_response()
        fetch_espn_pbp("evt4")
        age = svc.cache_age_seconds("pbp_evt4")
        self.assertIsNotNone(age)
        self.assertLess(age, 2.0)

    def test_cache_age_seconds_returns_none_for_missing_key(self):
        self.assertIsNone(svc.cache_age_seconds("nonexistent_key"))


if __name__ == "__main__":
    unittest.main()
