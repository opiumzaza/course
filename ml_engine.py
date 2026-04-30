"""Module C: detect anomalous Telegram users and cluster bot candidates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config


LOGGER = logging.getLogger(__name__)
NO_CLUSTER_ID = -1


def load_features(input_path: Path = config.FEATURES_PATH) -> pd.DataFrame:
    """Load engineered features from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str})
    required = {"user_id", *config.FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing required columns: {sorted(missing)}")
    return frame


def preprocess_features(features: pd.DataFrame) -> np.ndarray:
    """Impute missing feature values and apply standard scaling."""
    matrix = features.loc[:, config.FEATURE_COLUMNS].to_numpy(dtype=float)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    return StandardScaler().fit_transform(imputed)


def run_isolation_forest(scaled_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run IsolationForest and return labels plus anomaly scores."""
    if scaled_features.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    labels = model.fit_predict(scaled_features)
    scores = model.decision_function(scaled_features)
    return labels, scores


def count_non_noise_clusters(labels: np.ndarray) -> int:
    """Count DBSCAN-style clusters while excluding noise label -1."""
    return len({int(label) for label in labels if int(label) != NO_CLUSTER_ID})


def cluster_bot_candidates(bot_features: np.ndarray) -> np.ndarray:
    """Cluster anomalous users with DBSCAN and KMeans fallback."""
    n_candidates = bot_features.shape[0]
    if n_candidates == 0:
        return np.array([], dtype=int)
    if n_candidates < 3:
        LOGGER.warning("Fewer than 3 bot candidates; saving without cluster assignments.")
        return np.full(n_candidates, NO_CLUSTER_ID, dtype=int)

    dbscan_labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(bot_features)
    if count_non_noise_clusters(dbscan_labels) >= 2:
        return dbscan_labels.astype(int)

    n_clusters = min(5, n_candidates)
    try:
        return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(bot_features).astype(int)
    except ValueError as exc:
        LOGGER.warning("KMeans fallback failed: %s. Saving results without cluster_id.", exc)
        return np.full(n_candidates, NO_CLUSTER_ID, dtype=int)


def detect_bot_farms() -> tuple[Path, dict[str, float]]:
    """Run anomaly detection, cluster bot candidates, and save ML results."""
    config.configure_logging()
    config.ensure_directories()
    features = load_features()
    scaled = preprocess_features(features)
    labels, scores = run_isolation_forest(scaled)
    is_bot = labels == -1
    cluster_ids = np.full(features.shape[0], NO_CLUSTER_ID, dtype=int)
    bot_indices = np.flatnonzero(is_bot)
    cluster_ids[bot_indices] = cluster_bot_candidates(scaled[bot_indices])

    results = pd.DataFrame(
        {
            "user_id": features["user_id"],
            "anomaly_score": scores,
            "is_bot": is_bot.astype(int),
            "cluster_id": cluster_ids,
        }
    )
    config.ML_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.ML_RESULTS_PATH, index=False)

    total_users = int(features.shape[0])
    bot_count = int(is_bot.sum())
    bot_percentage = (bot_count / total_users * 100) if total_users else 0.0
    cluster_count = count_non_noise_clusters(cluster_ids[bot_indices])

    print(f"[C] Total users: {total_users}")
    print(f"[C] Bot candidates detected: {bot_count} ({bot_percentage:.1f}%)")
    print(f"[C] Bot farm clusters found: {cluster_count}")
    return config.ML_RESULTS_PATH, {
        "total_users": float(total_users),
        "bot_count": float(bot_count),
        "bot_percentage": bot_percentage,
        "cluster_count": float(cluster_count),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the Module C command-line parser."""
    parser = argparse.ArgumentParser(description="Detect bot candidates and bot-farm clusters.")
    return parser


def main() -> None:
    """Run ML detection from the command line."""
    build_parser().parse_args()
    output_path, _ = detect_bot_farms()
    print(f"Module C output saved → {output_path}")


if __name__ == "__main__":
    main()
