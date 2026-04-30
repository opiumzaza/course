# Telegram Botfarm Detector

## Project overview

Telegram Botfarm Detector is an unsupervised graph-analysis pipeline for finding coordinated comment behavior in public Telegram news channels. It collects recent post comments, builds a user co-activity graph, engineers behavioral and graph features, detects anomalous users with machine learning, and renders an interactive HTML investigation dashboard. The project is designed as a coursework prototype that runs the same way on macOS Apple Silicon and Windows x64. It does not require GPU libraries or supervised labels.

## System requirements

- Python 3.10, 3.11, or 3.12
- macOS Apple Silicon M2, arm64
- Windows x64
- Telegram API credentials from https://my.telegram.org/apps

## Installation

### macOS M2

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Windows x64

```bat
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

## Telegram API setup

1. Open https://my.telegram.org/apps and sign in with the Telegram account that will collect public-channel comments.
2. Create a Telegram application if you do not already have one.
3. Copy the generated `api_id` and `api_hash`.
4. Copy `.env.example` to `.env` in the project root.
5. Fill in `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `TELEGRAM_PHONE`.
6. Optionally set `TELEGRAM_CHANNELS=ukraine_now,suspilne_news,pravda_ua` or pass `--channels` at runtime.
7. On first collection, Telethon may prompt for a login code sent by Telegram.

## Usage

Run the full pipeline with the default 200-post limit:

```bash
python pipeline.py --all
```

Run everything with explicit channels and a larger post cap:

```bash
python pipeline.py --all --limit 500 --channels ukraine_now,suspilne_news,pravda_ua
```

Run only feature engineering after raw comments already exist:

```bash
python pipeline.py --features
```

Expected output includes progress lines such as:

```text
[A] Collected 1234 comments from ukraine_now
✅ Module A complete — data/raw_comments.csv
[B] Feature matrix shape: 842 users × 7 features
✅ Module B complete — data/features.csv
[C] Total users: 842
[C] Bot candidates detected: 42 (5.0%)
[C] Bot farm clusters found: 3
✅ Module C complete — data/ml_results.csv
[D] Dashboard saved → output/botfarm_graph.html
✅ Module D complete — output/botfarm_graph.html
```

## Output files

| Filename | Location | Description |
| --- | --- | --- |
| `raw_comments.csv` | `data/` | Raw collected comments with user, post, reply, channel, and timestamp fields. |
| `features.csv` | `data/` | Seven-feature per-user matrix used by the ML engine. |
| `graph.graphml` | `data/` | Weighted co-activity graph where users are connected by near-simultaneous same-post commenting. |
| `ml_results.csv` | `data/` | User anomaly scores, bot labels, and bot-farm cluster IDs. |
| `botfarm_graph.html` | `output/` | Interactive PyVis dashboard for visual inspection of suspected bot farms. |

## How it works

**Module A — collector.py:** The collector connects to Telegram with Telethon credentials loaded from `.env`, iterates over recent posts in configurable public channels, and saves non-anonymous comments to `data/raw_comments.csv`. It retries transient network and RPC failures, handles Telegram flood waits, and prints per-channel collection progress.

**Module B — features.py:** The feature builder loads raw comments, constructs a weighted co-activity graph with NetworkX, and connects users who comment on the same post within the configured time window. It computes reaction speed, posting rate, duplicate-content ratio, graph degree, clustering coefficient, and channel diversity for each user.

**Module C — ml_engine.py:** The ML engine imputes missing values, scales all seven features, and uses Isolation Forest to identify anomalous users without requiring labeled training data. It then clusters bot candidates with DBSCAN and falls back to KMeans when density-based clustering cannot identify multiple groups.

**Module D — visualizer.py:** The visualizer combines the GraphML graph and ML results into an interactive PyVis HTML dashboard. Normal users are grey, unclustered bot candidates are orange, clustered bot-farm members are red, and edge width reflects repeated co-activity.

## Interpreting results

Open `output/botfarm_graph.html` in a browser after running the pipeline. Grey nodes are users classified as normal, orange nodes are isolated bot candidates, and red nodes are anomalous users assigned to a cluster. Red clusters indicate groups of accounts that share suspicious timing, content, and co-activity patterns; they should be treated as investigation leads rather than automatic proof of malicious automation.
