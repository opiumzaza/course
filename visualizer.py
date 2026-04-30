"""Module D: render bot-farm detection results as an interactive PyVis dashboard."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

import config


LOGGER = logging.getLogger(__name__)
NO_CLUSTER_ID = -1


def load_graph(input_path: Path = config.GRAPH_PATH) -> nx.Graph:
    """Load the co-activity graph from GraphML."""
    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")
    return nx.read_graphml(input_path)


def load_results(input_path: Path = config.ML_RESULTS_PATH) -> pd.DataFrame:
    """Load ML bot-detection results from CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"ML results file not found: {input_path}")
    frame = pd.read_csv(input_path, dtype={"user_id": str})
    required = {"user_id", "anomaly_score", "is_bot", "cluster_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"ML results file is missing required columns: {sorted(missing)}")
    return frame


def result_lookup(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Convert ML results to a lookup keyed by user ID."""
    return results.set_index("user_id").to_dict(orient="index")


def node_style(user_id: str, lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    """Return PyVis style attributes for a graph node."""
    result = lookup.get(user_id, {})
    is_bot = int(result.get("is_bot", 0)) == 1
    cluster_id = int(result.get("cluster_id", NO_CLUSTER_ID))
    anomaly_score = float(result.get("anomaly_score", 0.0))

    if is_bot and cluster_id != NO_CLUSTER_ID:
        return {
            "label": f"{user_id}\ncluster {cluster_id}",
            "color": "rgba(220, 38, 38, 1.0)",
            "size": 20,
            "title": f"user_id={user_id}<br>cluster_id={cluster_id}<br>anomaly_score={anomaly_score:.4f}",
        }
    if is_bot:
        return {
            "label": user_id,
            "color": "rgba(249, 115, 22, 1.0)",
            "size": 14,
            "title": f"user_id={user_id}<br>bot candidate<br>anomaly_score={anomaly_score:.4f}",
        }
    return {
        "label": user_id,
        "color": "rgba(107, 114, 128, 0.5)",
        "size": 6,
        "title": f"user_id={user_id}<br>normal user<br>anomaly_score={anomaly_score:.4f}",
    }


def build_network(graph: nx.Graph, results: pd.DataFrame) -> Network:
    """Build a PyVis network with bot-specific node and edge styling."""
    network = Network(height="900px", width="100%", bgcolor="#111827", font_color="#f9fafb")
    network.barnes_hut(gravity=-30_000, central_gravity=0.3, spring_length=120)
    lookup = result_lookup(results)

    for user_id in graph.nodes:
        network.add_node(str(user_id), **node_style(str(user_id), lookup))

    for source, target, attributes in graph.edges(data=True):
        weight = float(attributes.get("weight", 1))
        network.add_edge(
            str(source),
            str(target),
            value=min(weight, 5),
            width=min(weight, 5),
            title=f"co-activity weight={weight:g}",
            color="rgba(156, 163, 175, 0.45)",
        )
    return network


def legend_html() -> str:
    """Return the HTML legend overlay inserted into the dashboard."""
    return """
<div style="
  position: fixed;
  top: 18px;
  right: 18px;
  z-index: 9999;
  padding: 14px 16px;
  color: #f9fafb;
  background: rgba(17, 24, 39, 0.92);
  border: 1px solid rgba(249, 250, 251, 0.18);
  border-radius: 10px;
  font-family: Arial, sans-serif;
  font-size: 14px;
  line-height: 1.45;
">
  <strong>Botfarm Detector</strong><br>
  <span style="color:#9ca3af;">●</span> Normal user<br>
  <span style="color:#f97316;">●</span> Bot candidate<br>
  <span style="color:#dc2626;">●</span> Clustered bot farm<br>
  <small>Edge width = co-activity weight, capped at 5</small>
</div>
"""


def inject_legend(html_path: Path) -> None:
    """Insert the legend panel into the generated PyVis HTML file."""
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>\n{legend_html()}", 1)
    html_path.write_text(html, encoding="utf-8")


def render_dashboard() -> Path:
    """Render the graph dashboard to output/botfarm_graph.html."""
    config.configure_logging()
    config.ensure_directories()
    graph = load_graph()
    results = load_results()
    network = build_network(graph, results)
    config.DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    network.save_graph(str(config.DASHBOARD_PATH))
    inject_legend(config.DASHBOARD_PATH)
    print("[D] Dashboard saved → output/botfarm_graph.html")
    return config.DASHBOARD_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the Module D command-line parser."""
    parser = argparse.ArgumentParser(description="Render bot-farm graph dashboard.")
    return parser


def main() -> None:
    """Run dashboard rendering from the command line."""
    build_parser().parse_args()
    config.configure_logging()
    config.confirm_overwrite_runtime_outputs()
    output_path = render_dashboard()
    print(f"Module D output saved → {output_path}")


if __name__ == "__main__":
    main()
