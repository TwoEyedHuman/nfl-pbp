import json
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub out streamlit before importing the module under test
import sys
st_stub = types.ModuleType("streamlit")
st_stub.cache_data = lambda **kw: (lambda f: f)
sys.modules.setdefault("streamlit", st_stub)

from data_provider.service_espnapi import fetch_espn_pbp

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


if __name__ == "__main__":
    unittest.main()
