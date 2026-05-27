# NFL Win Probability

A Streamlit dashboard that displays play-by-play win probability for historical and live NFL games. Supports two selectable models (Logistic Regression and XGBoost). Historical data sourced via nflreadpy; live game data fetched from the ESPN API with in-memory caching.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Technology Stack](#technology-stack)
4. [Environment Strategy](#environment-strategy)
5. [Scale-to-Zero Caveat](#scale-to-zero-caveat)
6. [Pre-Flight Checklist](#pre-flight-checklist)
7. [Implementation Stories](#implementation-stories)
8. [Seasonal Operations](#seasonal-operations)
9. [Definition of Done](#definition-of-done)

---

## Architecture Overview

```
Browser
  │
  ▼
Fly.io Machine
  │
  ▼
dashboard.py (Streamlit entry point)
  │
  ├── data_provider/
  │     ├── service_espnapi.py   ← live games (ESPN API, memory cache + TTL)
  │     └── service_nflreadpy.py ← historical games (nflreadpy, parquet cache)
  │
  ├── models/
  │     ├── logistic_regression_wp.pkl  ← selectable by user
  │     └── xgboost_wp.pkl             ← selectable by user
  │
  └── Scheduler (APScheduler, in-process)
        └── Nightly refresh of nflreadpy parquet cache
```

### Key Design Decisions

- **`data_provider/` as the data layer** — `dashboard.py` is a pure entry point; all ESPN API and nflreadpy logic lives in the service modules. This separation already exists and should be preserved.
- **Two selectable models** — both `.pkl` files are baked into the Docker image. The user selects the model via the dashboard UI; the selected model scores each play at render time.
- **Two cache strategies** — historical data cached as parquet (refreshed nightly by APScheduler running in-process); live data cached in-memory with a 60-second TTL.
- **Scale-to-zero with seasonal override** — see [Scale-to-Zero Caveat](#scale-to-zero-caveat).
- **Pipfile → requirements.txt** — Docker builds use `requirements.txt`. A `make freeze` target exports Pipfile deps to `requirements.txt` to keep them in sync.

---

## Repository Structure

```
nfl-pbp/
├── README.md
├── Makefile                        ← update: add freeze, build, run, down targets
├── Pipfile                         ← exists
├── Pipfile.lock                    ← exists
├── requirements.txt                ← generated from Pipfile via `make freeze`
├── Dockerfile                      ← new
├── fly.toml                        ← new
├── .streamlit/
│   └── config.toml                 ← new: disable toolbar, dark theme
├── dashboard.py                    ← exists: add scheduler init, model loader
├── data_provider/
│   ├── __init__.py                 ← exists
│   ├── service_espnapi.py          ← exists: add TTL cache + cache_age_seconds()
│   └── service_nflreadpy.py        ← exists: add parquet cache + cache_is_stale()
├── models/
│   ├── logistic_regression_wp.pkl  ← exists
│   └── xgboost_wp.pkl             ← exists
├── training/                       ← exists: untouched
├── tests/
│   ├── __init__.py                 ← exists
│   └── test_service_espnapi.py     ← exists: extend with cache tests
├── sample_data/                    ← exists: used in tests only
├── Issues                          ← exists: untouched
└── cache/
    └── .gitkeep                    ← new: parquet files written here at runtime
```

---

## Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| App framework | Streamlit | Already in use |
| Charts | Plotly | Interactive win probability line chart |
| Historical data | nflreadpy | Play-by-play, cached as parquet |
| Live data | ESPN API (public) | Scores + live play-by-play, no auth needed |
| Models | scikit-learn / XGBoost (pkl) | Already trained, baked into image |
| Cache (historical) | Parquet files (local FS) | Fast reads, no DB needed |
| Cache (live) | In-memory dict with TTL | Simple, cleared on restart |
| Scheduler | APScheduler (in-process) | No extra container needed |
| Dependency mgmt | Pipenv + requirements.txt | Pipenv for local dev, requirements.txt for Docker |
| Hosting | Fly.io (scale to zero) | Docker-native, configurable min machines |
| CI/CD | GitHub Actions | Auto-deploy on push to `main` |

---

## Environment Strategy

| | Local | Production (Fly.io) |
|---|---|---|
| Run command | `make dev` | Docker via Fly Machine |
| URL | `http://localhost:8501` | `https://nfl.brandonlocke.xyz` |
| Cache dir | `./cache/` | `/app/cache/` |
| Models dir | `./models/` | `/app/models/` |
| Min machines | N/A | 0 (off-season) / 1 (in-season) |

---

## Scale-to-Zero Caveat

APScheduler runs inside the Streamlit process. When the Fly Machine scales to zero, the scheduler stops and the nflreadpy cache goes stale.

**Mitigation:**
- **Off-season** (`min_machines_running = 0`): historical data doesn't change week-to-week. Stale cache is acceptable. On cold start, if cache is missing or older than 24 hours, a cache rebuild runs before the app is marked healthy.
- **In-season** (`min_machines_running = 1`): keep one machine alive so the nightly cron runs. Costs ~$2-3/month.

See [Seasonal Operations](#seasonal-operations) for the exact commands.

---

## Pre-Flight Checklist

```bash
# Deps installed
pipenv install

# App runs locally
make dev

# Tests pass
make test

# ESPN API reachable
curl "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# Docker builds
make build
make run

# Fly CLI
fly version
fly auth whoami
```

---

## Implementation Stories

Stories are ordered by dependency. Each is scoped for a single Claude CLI session.

---

### EPIC 1 — Caching & Scheduler Layer

**Epic Goal:** Both data providers have robust caching. APScheduler keeps historical cache fresh. App handles a cold cache gracefully on startup.

---

#### Story 1.1 — nflreadpy Parquet Cache

**Context:** `service_nflreadpy.py` exists and fetches data. Adding a parquet cache layer so data is not re-fetched on every Streamlit rerun, and so the scheduler has something to refresh.

**Assumptions:**
- `cache/` directory exists (`.gitkeep` committed)
- `CACHE_DIR` env var configures cache path (default: `./cache`)
- `pyarrow` or `fastparquet` added to Pipfile for parquet support

**Tasks:**
- Add to `service_nflreadpy.py`:
  - `cache_path(season: int) -> Path` — returns `{CACHE_DIR}/pbp_{season}.parquet`
  - `cache_is_stale(season: int, max_age_hours: int = 24) -> bool` — true if file missing or older than threshold
  - `fetch_and_cache_season(season: int)` — fetches via nflreadpy, writes parquet, logs duration
  - `load_game(season: int, game_id: str) -> pd.DataFrame` — reads from parquet, filters to game
  - `available_games(season: int) -> list[dict]` — reads parquet, returns game metadata (id, teams, week, final score)
- Existing fetch logic preserved — new functions wrap it
- New unit tests: `cache_is_stale`, `available_games`, cache read

**Out of Scope:** Scheduler (Story 1.2), dashboard wiring (Story 1.3).

**Acceptance Criteria:**
- [ ] `cache_is_stale()` returns `True` when `cache/` is empty
- [ ] `fetch_and_cache_season(2024)` writes `cache/pbp_2024.parquet`
- [ ] `load_game()` returns correct rows for a known game ID
- [ ] `available_games()` returns list with correct team names and week numbers
- [ ] `make test` — all existing and new tests pass
- [ ] No Streamlit imports in `service_nflreadpy.py`

---

#### Story 1.2 — ESPN API TTL Cache

**Context:** Story 1.1 complete. `service_espnapi.py` exists. Adding TTL cache to avoid hammering the ESPN API on rapid Streamlit reruns.

**Assumptions:**
- Live cache stored as module-level dict — reset on container restart, acceptable
- `LIVE_TTL = 60` seconds for live games; `COMPLETED_TTL = 300` for finished games
- Sample data in `sample_data/` used for all test mocking — no real API calls in tests

**Tasks:**
- Add to `service_espnapi.py`:
  - `_cache: dict` — module-level store `{key: (data, timestamp)}`
  - `_get_cached(key, ttl) -> Any | None`
  - `_set_cached(key, data)`
  - `cache_age_seconds(key) -> float | None`
  - Wrap existing fetch functions to check cache before hitting API
- Extend `test_service_espnapi.py`:
  - Cache hit: second call does not hit API (assert mock call count == 1)
  - Cache miss after TTL: API called again
  - All responses mocked from `sample_data/`

**Acceptance Criteria:**
- [ ] Two rapid calls → second uses cache (verified via mock assert)
- [ ] After TTL expiry → cache miss, API called again
- [ ] `cache_age_seconds()` returns correct value
- [ ] `make test` passes including new cache tests
- [ ] No real ESPN API calls during test suite run

---

#### Story 1.3 — APScheduler In-Process Refresh

**Context:** Stories 1.1 and 1.2 complete. Cache layer exists. Adding scheduled nightly refresh.

**Assumptions:**
- `APScheduler` added to Pipfile and `requirements.txt` (`make freeze` run after)
- Scheduler started once per process via `st.session_state` guard
- Logs go to Python `logging` — visible in terminal and `docker logs`

**Tasks:**
- Create `scheduler.py` at repo root (or inline in `dashboard.py` if minimal):
  - `start_scheduler()` — `BackgroundScheduler`, registers jobs, starts, returns instance
  - Job 1: on startup — `fetch_and_cache_season(current_season)` if `cache_is_stale()`
  - Job 2: nightly 2am ET — `fetch_and_cache_season(current_season)`
  - `atexit.register(scheduler.shutdown)` for clean shutdown
- In `dashboard.py`:
  ```python
  if "scheduler" not in st.session_state:
      st.session_state.scheduler = start_scheduler()
  ```
- Sidebar: "Historical data last updated: {timestamp}" or "Building cache..." if stale
- All job runs logged at INFO level

**Acceptance Criteria:**
- [ ] `make dev` — scheduler startup log visible in terminal
- [ ] Delete `cache/*.parquet` → restart → cache rebuilds, log visible
- [ ] Page refresh → no duplicate scheduler (verify via log — only one startup message)
- [ ] Cache status indicator shows correct last-updated timestamp
- [ ] `Ctrl+C` → atexit shutdown log visible

---

### EPIC 1 Integration Gate

- [ ] `make dev` — app loads, historical games available from cache
- [ ] Empty `cache/` → restart → rebuilds automatically within 90 seconds
- [ ] Rapid page reruns → ESPN API called at most once per TTL (verify via logs)
- [ ] Scheduler does not duplicate on Streamlit reruns
- [ ] `make test` — all tests pass

---

### EPIC 2 — Dashboard Polish

**Epic Goal:** Model selector properly wired, win probability chart is annotated and clear, live game UX is responsive, historical navigation is intuitive.

---

#### Story 2.1 — Model Selector Cleanup

**Context:** Epic 1 complete. Both `.pkl` models exist. Dashboard may already have a model selector — verify it loads once and switches cleanly.

**Assumptions:**
- Both models accept the same feature set (verify before starting)
- Model loaded once into `st.session_state` to avoid reloading on every rerun

**Tasks:**
- Sidebar: radio buttons — "Logistic Regression" | "XGBoost"
- Load model once:
  ```python
  if "model" not in st.session_state or st.session_state.model_name != selected:
      st.session_state.model = load_model(selected)
      st.session_state.model_name = selected
  ```
- `load_model(name: str) -> object` — loads correct `.pkl` from `models/`
- Chart title includes active model name
- Switching models reruns chart

**Acceptance Criteria:**
- [ ] Both models selectable in sidebar
- [ ] Switching models updates the chart
- [ ] Model loads once per selection — verify via log (not on every rerun)
- [ ] Chart title reflects active model name
- [ ] Both models output valid 0-100% probabilities for a known historical game

---

#### Story 2.2 — Win Probability Chart Polish

**Context:** Story 2.1 complete.

**Tasks:**
- X axis: game clock (preferred); fall back to play number if clock unavailable
- Y axis: 0-100% home team win probability
- Team colors from ESPN data; fallback to generic NFL blue/red if unavailable
- Key play annotations: TDs, turnovers, field goals — marker on line with hover label
- Quarter/OT shaded regions (subtle alternating background)
- Hover tooltip: down, distance, yard line, play description, win probability
- Completed game: final score in chart title
- Live game: current score in chart title

**Acceptance Criteria:**
- [ ] Chart renders for any historical game without errors
- [ ] Key play annotations visible without overlapping at 800px width
- [ ] Hover tooltip shows all fields
- [ ] Quarter shading visible
- [ ] Chart title shows correct score for both historical and live games

---

#### Story 2.3 — Live Game UX & Historical Navigation

**Context:** Story 2.2 complete.

**Tasks:**
- Sidebar sections:
  - **Live** (shown only when games in progress): game picker, auto-refresh toggle, cache age indicator, manual refresh button
  - **Historical**: season dropdown, week filter, game picker
- Auto-refresh: `st.rerun()` every 60 seconds when toggled on
- Manual refresh button: clears live cache entry, re-fetches, reruns
- ESPN API error: warning banner with last-good-data timestamp; do not crash
- "No live games": hide live section, show informational message
- Historical defaults: most recent cached season, most recent week

**Acceptance Criteria:**
- [ ] Live section hidden when ESPN returns no in-progress games
- [ ] Auto-refresh reruns every ~60 seconds when enabled
- [ ] Manual refresh clears cache and fetches fresh data
- [ ] ESPN API error → warning banner, last cached data still displayed
- [ ] Historical defaults to most recent season/week on load
- [ ] Week filter scopes game list correctly

---

### EPIC 2 Integration Gate

- [ ] Full historical flow: load → season → week → game → chart with annotations
- [ ] Model switch: change selection → chart updates with new probabilities
- [ ] Live flow (test on game day or with mocked response): banner → game → refresh → chart updates
- [ ] No Python exceptions during normal navigation
- [ ] App renders correctly at 800px wide

---

### EPIC 3 — Streamlit Config & Dockerization

**Epic Goal:** App styled for iframe embed, runs identically in Docker, image lean and healthy.

---

#### Story 3.1 — Streamlit Config & Makefile

**Context:** Epic 2 complete. No `.streamlit/config.toml` yet.

**Tasks:**
- `.streamlit/config.toml`:
  ```toml
  [server]
  headless = true
  address = "0.0.0.0"
  port = 8501

  [client]
  toolbarMode = "minimal"
  showSidebarNavigation = false

  [theme]
  base = "dark"
  primaryColor = "#013369"
  backgroundColor = "#1a1a1a"
  secondaryBackgroundColor = "#2d2d2d"
  textColor = "#ffffff"
  ```
- Update `Makefile`:
  ```makefile
  freeze:
      pipenv requirements > requirements.txt

  dev:
      pipenv run streamlit run dashboard.py

  build:
      docker build -t nfl-win-probability .

  run:
      docker run -p 8501:8501 --env CACHE_DIR=/app/cache nfl-win-probability

  test:
      pipenv run pytest tests/

  down:
      docker stop $$(docker ps -q --filter ancestor=nfl-win-probability)
  ```
- Run `make freeze` and commit `requirements.txt`

**Acceptance Criteria:**
- [ ] No Streamlit toolbar visible at 800px viewport
- [ ] NFL blue theme applied consistently
- [ ] `make freeze` produces valid `requirements.txt`
- [ ] `make dev` and `make test` work correctly

---

#### Story 3.2 — Dockerfile

**Context:** Story 3.1 complete. `requirements.txt` is current.

**Assumptions:**
- `models/` baked into image — pkl files don't change without retraining
- `cache/` created in image, writable by app user, populated at runtime
- `sample_data/`, `training/`, `tests/`, `Issues` excluded from image

**Tasks:**
- Multi-stage Dockerfile — builder installs deps; runner copies only runtime files
- Non-root user (`appuser`) owns `/app/cache`
- `HEALTHCHECK --start-period=90s` to allow cache build time
- `ENV CACHE_DIR=/app/cache`
- `.dockerignore` excludes: `.git`, `__pycache__`, `training/`, `sample_data/`, `tests/`, `Issues`, `Pipfile`, `Pipfile.lock`, `cache/*.parquet`

**Acceptance Criteria:**
- [ ] `make build` succeeds
- [ ] `make run` — app at `http://localhost:8501`, cache builds within 90s
- [ ] `docker exec <container> whoami` → `appuser`
- [ ] `docker exec <container> ls /app/cache` → parquet file present after build
- [ ] Image under 600MB
- [ ] Healthcheck healthy after 90 seconds
- [ ] `training/` and `sample_data/` absent from image
- [ ] `make down` stops container cleanly

---

### EPIC 3 Integration Gate

- [ ] `make build && make run` — full app works, models selectable, chart renders
- [ ] Cache rebuilds automatically inside container within 90 seconds
- [ ] Healthcheck healthy
- [ ] `make down` → exits within 15 seconds (scheduler shuts down cleanly)
- [ ] Fresh `make run` after stop → clean start, no stale state

---

### EPIC 4 — Fly.io Deployment & CI/CD

**Epic Goal:** App live at `https://nfl.brandonlocke.xyz`, scales to zero off-season, auto-deploys on push to `main`.

---

#### Story 4.1 — Fly.io App Setup

**Context:** Epic 3 complete. Docker image healthy locally.

**Assumptions:**
- `fly` CLI installed and authenticated
- Pattern established from Bobiverse and F1 deployments
- `min_machines_running` toggled manually per [Seasonal Operations](#seasonal-operations)

**Tasks:**
- `fly launch` — name `nfl-win-probability`, nearest region, no managed DB
- `fly.toml` with `CACHE_DIR=/app/cache`, health check grace period 90s, `min_machines_running = 0`
- `fly deploy`
- `fly certs add nfl.brandonlocke.xyz`
- Cloudflare DNS CNAME: `nfl` → `nfl-win-probability.fly.dev`
- Update personal site `projects.yaml` — activate NFL entry (Story 6.2 of personal site refactor README)

**Acceptance Criteria:**
- [ ] `fly deploy` succeeds
- [ ] `https://nfl-win-probability.fly.dev` — app loads, cache builds within 90s
- [ ] `https://nfl.brandonlocke.xyz` — loads after DNS propagation
- [ ] `fly logs` — scheduler startup and cache build logged
- [ ] Health check passes during cache build (grace period sufficient)
- [ ] Cold start: `fly scale count 0` → wait 5 min → visit URL → wakes within 10s

---

#### Story 4.2 — GitHub Actions CI/CD

**Context:** Story 4.1 complete.

**Tasks:**
- `.github/workflows/deploy.yml`: test job + deploy job (depends on test); push to `main`
- `.github/workflows/pr-check.yml`: test only on PRs
- Test job: `pip install -r requirements.txt && pytest tests/`
- `FLY_API_TOKEN` in GitHub repository secrets
- Add note in CI README: run `make freeze` before pushing if Pipfile changed

**Acceptance Criteria:**
- [ ] Push to `main` → deploy within 5 minutes
- [ ] Failing test → deploy blocked
- [ ] PR → test only, no deploy
- [ ] Post-deploy: `https://nfl.brandonlocke.xyz` reflects change

---

### EPIC 4 Integration Gate

- [ ] Cold start: scale to zero → visit personal site NFL embed → spinner → app loads
- [ ] Push change → GitHub Actions → live within 5 minutes
- [ ] Both models selectable and correct in production
- [ ] `fly logs` clean during normal use
- [ ] Personal site NFL card no longer shows "Coming Soon"
- [ ] Monthly cost: ~$0 off-season, ~$2-3 in-season (confirm in dashboard)

---

## Seasonal Operations

```bash
# Start of NFL season (early September)
fly scale count 1 -a nfl-win-probability
fly logs -a nfl-win-probability        # verify scheduler running

# End of NFL season (after Super Bowl, ~mid February)
fly scale count 0 -a nfl-win-probability
fly status -a nfl-win-probability      # verify stopped
```

Set calendar reminders for both. Forgetting to scale down costs ~$2-3/month unnecessarily.

---

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] `make dev` works locally
- [ ] `make build && make run` works identically
- [ ] `make test` passes
- [ ] `requirements.txt` current (`make freeze` run before last commit)
- [ ] No secrets committed
- [ ] Epic integration gate passes before moving to next epic
- [ ] Seasonal operations calendar reminders set
