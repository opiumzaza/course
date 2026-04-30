"""Central configuration for Telegram bot-farm detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from os import getenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
RAW_COMMENTS_PATH = DATA_DIR / "raw_comments.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
GRAPH_PATH = DATA_DIR / "graph.graphml"
ML_RESULTS_PATH = DATA_DIR / "ml_results.csv"
DASHBOARD_PATH = OUTPUT_DIR / "botfarm_graph.html"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"

DEFAULT_CHANNELS = ("ukraine_now", "suspilne_news", "pravda_ua")
DEFAULT_POST_LIMIT = 200
MIN_POST_LIMIT = 100
MAX_POST_LIMIT = 500
CO_ACTIVITY_WINDOW_SECONDS = 120
TELEGRAM_REQUEST_RETRIES = 3
RATE_LIMIT_BASE_DELAY_SECONDS = 2

VIZ_MIN_EDGE_WEIGHT = 2
VIZ_MAX_NODES = 1500
VIZ_STABILIZATION_ITERATIONS = 200

FEATURE_COLUMNS = (
    "reaction_speed_mean",
    "reaction_speed_std",
    "comment_count_per_day",
    "duplicate_ratio",
    "graph_degree",
    "clustering_coefficient",
    "unique_channels",
)

load_dotenv(BASE_DIR / ".env")

TELEGRAM_API_ID = getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = getenv("TELEGRAM_PHONE")
_CHANNELS_ENV = getenv("TELEGRAM_CHANNELS")
TELEGRAM_CHANNELS = tuple(
    channel.strip().lstrip("@")
    for channel in (_CHANNELS_ENV.split(",") if _CHANNELS_ENV else DEFAULT_CHANNELS)
    if channel.strip()
)


def ensure_directories() -> None:
    """Create runtime output directories when they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def has_directory_contents(path: Path) -> bool:
    """Return whether a directory exists and contains at least one entry."""
    return path.exists() and any(path.iterdir())


def confirm_overwrite_runtime_outputs() -> None:
    """Pause for confirmation before overwriting existing data or output files."""
    populated = [path for path in (DATA_DIR, OUTPUT_DIR) if has_directory_contents(path)]
    if not populated:
        return
    locations = ", ".join(str(path) for path in populated)
    response = input(
        f"The following runtime directories already contain files: {locations}. "
        "Overwrite/update generated outputs? [y/N]: "
    )
    if response.strip().lower() not in {"y", "yes"}:
        raise RuntimeError("Stopped before overwriting existing data/output files.")


def validate_telegram_credentials() -> None:
    """Validate that Telegram credentials are available and well formed."""
    missing = [
        name
        for name, value in (
            ("TELEGRAM_API_ID", TELEGRAM_API_ID),
            ("TELEGRAM_API_HASH", TELEGRAM_API_HASH),
            ("TELEGRAM_PHONE", TELEGRAM_PHONE),
        )
        if not value or value.startswith("your_")
    ]
    if missing:
        raise ValueError(
            "Missing Telegram credentials in .env: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill values from https://my.telegram.org/apps."
        )
    try:
        int(str(TELEGRAM_API_ID))
    except ValueError as exc:
        raise ValueError("TELEGRAM_API_ID must be an integer.") from exc


def get_api_id() -> int:
    """Return the Telegram API ID as an integer after validation."""
    validate_telegram_credentials()
    return int(str(TELEGRAM_API_ID))


def configure_logging() -> None:
    """Configure consistent logging for warnings and errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
