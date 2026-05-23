# NFL Play-by-Play Win Probability — Refactor Plan

> Fix data leakage and schema mismatches between the ESPN live feed and historical training data so model inputs are in-distribution for every play.

---

## Table of Contents

1. [Problem Summary](#problem-summary)
2. [Architecture Overview](#architecture-overview)
3. [Root Causes](#root-causes)
4. [Refactor Strategy](#refactor-strategy)
5. [Implementation Stories](#implementation-stories)
6. [Definition of Done](#definition-of-done)

---

## Problem Summary

The dashboard has two data sources — `service_espnapi.py` (live games) and a historical NFLReadPy source (training/replay). Both return a `PBPData` NamedTuple, but the DataFrames inside have different semantics. The model was trained exclusively on historical data. Several ESPN-specific sentinel values survive into model input, producing out-of-distribution (OOD) predictions on live games.

**Observed symptoms:**
- Win probability spikes or collapses immediately after touchdowns
- Model behavior on scoring plays differs from equivalent historical plays
- `away_team` KeyErrors caught only by a band-aid guard in `dashboard.py`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  dashboard.py                   │
│   - renders win probability chart               │
│   - calls one of two services based on mode     │
└────────────┬────────────────────┬───────────────┘
             │                    │
             ▼                    ▼
  service_espnapi.py     service_historical.py
  (live ESPN API)        (NFLReadPy / nflverse)
             │                    │
             └────────┬───────────┘
                      ▼
             normalize_play_df()         ← NEW: single contract
                      │
                      ▼
              PBPData(DataFrame)
                      │
                      ▼
             model.predict(features)
```

### Key Design Decisions

- **Single normalization contract:** Both services call `normalize_play_df(df)` before returning. `dashboard.py` should never need to patch up missing columns or sentinel values.
- **start.\* fields over end.\* fields (ESPN):** ESPN `end.*` fields represent post-play state; historical fields represent pre-snap state. Switching to `start.*` aligns semantics and eliminates the `-1`/`0` sentinels on scoring plays at the source.
- **NaN as the universal missing value:** Historical convention is NaN for non-scrimmage plays. All services must conform. `-1`, `0`, and other sentinels are illegal in the normalized output.

---

## Root Causes

| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | ESPN uses `end.down` (post-play); scoring plays set it to `-1` | `service_espnapi.py:96–97` | `-1` survives `dropna` (it's an int), hits model OOD |
| 2 | ESPN uses `end.yardsToEndzone`; scoring plays set it to `0` | `service_espnapi.py` | `yardline_100 = 0` is OOD; model trained only on `1–99` |
| 3 | No shared normalization step | Both services + `dashboard.py` | Schema diverges silently; new services will repeat the same bugs |
| 4 | `away_team` not populated by ESPN service | `service_espnapi.py` | `dashboard.py:134` guard is a band-aid, not a fix |
| 5 | `end.*` vs `start.*` field semantics differ across sources | `service_espnapi.py` | Every play (not just scoring) has a semantic mismatch |

---

## Refactor Strategy

Three targeted changes, in dependency order:

1. **Story 1 — Switch ESPN to `start.*` fields.** Fixes bugs 1, 2, and 5 simultaneously. `start.down` and `start.yardsToEndzone` reflect pre-snap state on all plays, matching historical convention. Scoring plays now show the correct down and yardline rather than `-1`/`0`.

2. **Story 2 — Add `away_team` to ESPN output.** Scoreboard data is already fetched. Derive `away_team` from it and populate the column. Removes the `dashboard.py` band-aid entirely.

3. **Story 3 — Add `normalize_play_df(df)`.** Single post-processing function both services call before returning `PBPData`. Enforces: replace any remaining sentinel values (`-1` downs, `0` yardline) with `NaN`; assert required columns are present. This is the defensive layer — it catches regressions introduced by future service changes.

**Do not touch the model or training pipeline** in this refactor. The goal is to make live inputs match the distribution the model already expects.

---

## Implementation Stories

Each story is one Claude Code session. Keep them tight.

---

### EPIC 1 — Fix ESPN Data at the Source

**Epic Goal:** ESPN service returns pre-snap field values on all plays, including scoring plays. No sentinel values reach the model.

---

#### Story 1.1 — Switch ESPN service to `start.*` fields

**Context:** `service_espnapi.py` currently reads `end.down` and `end.yardsToEndzone` from the ESPN play object. This produces `-1` and `0` on scoring plays and mismatches historical field semantics on every play.

**Assumptions:**
- ESPN play objects expose both `start.*` and `end.*` sub-objects
- `start.down` and `start.yardsToEndzone` are populated on all scrimmage plays
- Test fixtures (if any) use the ESPN JSON shape

**Tasks:**
- In `service_espnapi.py`, replace all reads of `end.down` with `start.down`
- Replace all reads of `end.yardsToEndzone` with `start.yardsToEndzone`
- Audit the file for any other `end.*` field reads; replace with `start.*` equivalents where the historical convention is pre-snap state
- Update or add a unit test that feeds a mock scoring-play JSON and asserts `down` is not `-1` and `yardline_100` is not `0`

**Out of Scope:** `normalize_play_df` (Story 1.3), `away_team` (Story 1.2), any changes to `dashboard.py`.

**Acceptance Criteria:**
- [ ] Feed a mock ESPN touchdown play JSON through the service → `down` is a valid integer (1–4) or `NaN`, never `-1`
- [ ] Feed a mock ESPN touchdown play JSON → `yardline_100` is in range `1–99` or `NaN`, never `0`
- [ ] All existing unit tests pass
- [ ] Run the dashboard against a live or recorded ESPN game with a scoring play → no probability spike immediately after the score

---

#### Story 1.2 — Populate `away_team` in ESPN service

**Context:** Story 1.1 complete. `service_espnapi.py` already fetches scoreboard data that includes both team identifiers. `dashboard.py:190` has a guard `if 'away_team' in game_df.columns else "Away"` that exists only because this column is missing.

**Assumptions:**
- The scoreboard API response includes an away team identifier accessible at a known key path (verify before implementing)
- `away_team` in the historical DataFrame contains the team abbreviation string (e.g. `"KC"`)

**Tasks:**
- In `service_espnapi.py`, extract the away team identifier from the already-fetched scoreboard data
- Add `away_team` as a column to the returned DataFrame, populated with that identifier for every row
- Remove the band-aid guard from `dashboard.py:190` — replace with a direct column read
- Add an assertion in the ESPN service's unit test that `away_team` is present and non-null

**Out of Scope:** Adding `home_team` if not already present (do it only if it unblocks a failing test).

**Acceptance Criteria:**
- [ ] `service_espnapi.py` returns a DataFrame with an `away_team` column on every row
- [ ] `dashboard.py:190` no longer contains `if 'away_team' in game_df.columns`
- [ ] Unit test asserts `away_team` is populated
- [ ] Dashboard renders correctly for a live ESPN game (away team label is correct, not `"Away"`)

---

#### Story 1.3 — Add `normalize_play_df(df)` shared contract

**Context:** Stories 1.1 and 1.2 complete. Both services are closer to correct, but there is still no single place that enforces the schema contract. Future service changes could re-introduce sentinel values silently.

**Assumptions:**
- Both `service_espnapi.py` and `service_historical.py` (or equivalent) return a `PBPData` NamedTuple containing a DataFrame
- `dashboard.py` calls these services and then passes the DataFrame downstream to the model

**Tasks:**
- Create `normalization.py` (or add to an existing `utils.py`) with a `normalize_play_df(df: pd.DataFrame) -> pd.DataFrame` function
- Function must: replace `down == -1` with `NaN`; replace `yardline_100 == 0` with `NaN`; assert required columns (`down`, `yardline_100`, `ydstogo`, `score_differential`, `game_seconds_remaining`) are present, raising a descriptive `ValueError` if not
- Call `normalize_play_df` as the final step in both `service_espnapi.py` and `service_historical.py` before constructing `PBPData`
- Remove any equivalent patchwork from `dashboard.py` (e.g. the `dropna` call that was papering over `-1` surviving)
- Add unit tests: pass a DataFrame with `-1` down → assert it becomes `NaN`; pass a DataFrame missing a required column → assert `ValueError`

**Out of Scope:** Changes to model features, retraining, or adding new columns.

**Acceptance Criteria:**
- [ ] `normalize_play_df` is the single place that enforces sentinel → NaN replacement
- [ ] Both services call it before returning
- [ ] Unit tests pass for sentinel replacement and missing-column error
- [ ] `dashboard.py` contains no inline sentinel patching or column-presence guards for fields that `normalize_play_df` now guarantees
- [ ] Run full dashboard end-to-end against a game with scoring plays → no OOD model inputs in logs

---

### EPIC 1 Integration Gate

Before closing the refactor, verify manually against a live or recorded game that includes at least one touchdown:

- [ ] Win probability does **not** spike or collapse immediately after a scoring play
- [ ] `away_team` label in dashboard is the correct team abbreviation, not `"Away"`
- [ ] No `-1` or `0` values appear in the `down` or `yardline_100` columns logged at prediction time
- [ ] `python -m pytest` → all tests pass
- [ ] Revert Story 1.1 temporarily → normalization layer catches the `-1` and converts it (defense-in-depth check), then re-apply

---

## Definition of Done

A story is complete when:

- [ ] All acceptance criteria pass
- [ ] No sentinel values (`-1` down, `0` yardline) can reach `model.predict()` from either service
- [ ] New logic has unit tests with at least one positive and one negative case
- [ ] `dashboard.py` contains no column-presence guards for fields that the services now guarantee
- [ ] Epic integration gate passes before the refactor is considered done
