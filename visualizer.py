"""Module D: render bot-farm detection results as an interactive PyVis dashboard."""

from __future__ import annotations

import argparse
import json
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


def filter_graph_for_viz(
    graph: nx.Graph,
    lookup: dict[str, dict[str, object]],
    min_edge_weight: int = config.VIZ_MIN_EDGE_WEIGHT,
    max_nodes: int = config.VIZ_MAX_NODES,
) -> nx.Graph:
    """Return a graph reduced to a renderable subset.

    Drops weak edges (weight < `min_edge_weight`) so the dashboard isn't
    swamped by noise. Then prioritises nodes for inclusion until the
    `max_nodes` cap is reached, in this order: clustered bots → unclustered
    bot candidates → users connected to any bot → remaining users.
    Always keeps every bot-related node so the analysis surface is intact.
    """
    if min_edge_weight > 1:
        weak = [
            (u, v) for u, v, attrs in graph.edges(data=True)
            if float(attrs.get("weight", 1)) < min_edge_weight
        ]
        if weak:
            graph = graph.copy()
            graph.remove_edges_from(weak)

    def category(node: str) -> int:
        result = lookup.get(node, {})
        if int(result.get("is_bot", 0)) == 1:
            return 0 if int(result.get("cluster_id", NO_CLUSTER_ID)) != NO_CLUSTER_ID else 1
        return 3

    bot_nodes = {n for n in graph.nodes if category(n) <= 1}
    bot_neighbors = {nbr for n in bot_nodes for nbr in graph.neighbors(n)} - bot_nodes
    remaining = set(graph.nodes) - bot_nodes - bot_neighbors

    ordered: list[str] = []
    ordered.extend(sorted(n for n in bot_nodes if category(n) == 0))
    ordered.extend(sorted(n for n in bot_nodes if category(n) == 1))
    ordered.extend(sorted(bot_neighbors))
    ordered.extend(sorted(remaining))

    if len(ordered) <= max_nodes:
        keep = set(ordered)
    else:
        keep = set(ordered[:max_nodes])
        LOGGER.warning(
            "Dashboard capped at %s of %s nodes for browser performance "
            "(all bot-related nodes are kept; raise VIZ_MAX_NODES in config.py to show more).",
            max_nodes,
            len(ordered),
        )

    subgraph = graph.subgraph(keep).copy()
    isolated_normals = [
        n for n in list(subgraph.nodes)
        if subgraph.degree(n) == 0 and category(n) == 3
    ]
    subgraph.remove_nodes_from(isolated_normals)
    return subgraph


def physics_options() -> str:
    """Return JSON physics options that stabilise quickly and stop simulating."""
    return json.dumps({
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.3,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.6,
                "avoidOverlap": 0.2,
            },
            "stabilization": {
                "enabled": True,
                "iterations": config.VIZ_STABILIZATION_ITERATIONS,
                "fit": True,
            },
            "minVelocity": 0.75,
            "timestep": 0.5,
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "keyboard": True,
        },
    })


def build_network(graph: nx.Graph, results: pd.DataFrame) -> Network:
    """Build a PyVis network with bot-specific node and edge styling."""
    network = Network(height="900px", width="100%", bgcolor="#111827", font_color="#f9fafb")
    network.set_options(physics_options())
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


def legend_html(node_count: int, edge_count: int, total_nodes: int, total_edges: int) -> str:
    """Return the HTML legend overlay inserted into the dashboard."""
    return f"""
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
  <small>Edge width = co-activity weight, capped at 5</small><br>
  <small>Showing {node_count} / {total_nodes} nodes, {edge_count} / {total_edges} edges</small>
</div>
"""


def inject_legend(html_path: Path, legend: str) -> None:
    """Insert the legend panel into the generated PyVis HTML file."""
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<body>", f"<body>\n{legend}", 1)
    html_path.write_text(html, encoding="utf-8")


def render_cluster_dashboards(
    graph: nx.Graph,
    results: pd.DataFrame,
    output_dir: Path = config.CLUSTERS_DIR,
) -> list[Path]:
    """Render one drill-down HTML per bot-farm cluster."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_paths: list[Path] = []
    bot_results = results[(results["is_bot"] == 1) & (results["cluster_id"] != NO_CLUSTER_ID)]
    if bot_results.empty:
        return cluster_paths

    for cluster_id, cluster_users in bot_results.groupby("cluster_id"):
        members = {str(uid) for uid in cluster_users["user_id"]}
        members_in_graph = members.intersection(graph.nodes)
        if not members_in_graph:
            continue
        # Include direct neighbours so a cluster's coordination context is visible.
        neighbours = {nbr for m in members_in_graph for nbr in graph.neighbors(m)}
        subgraph = graph.subgraph(members_in_graph | neighbours).copy()
        sub_results = results[results["user_id"].isin(subgraph.nodes)]
        network = build_network(subgraph, sub_results)
        cluster_path = output_dir / f"cluster_{int(cluster_id)}.html"
        network.save_graph(str(cluster_path))
        legend = legend_html(
            node_count=subgraph.number_of_nodes(),
            edge_count=subgraph.number_of_edges(),
            total_nodes=subgraph.number_of_nodes(),
            total_edges=subgraph.number_of_edges(),
        )
        inject_legend(cluster_path, legend)
        cluster_paths.append(cluster_path)
    return cluster_paths


def render_dashboard() -> Path:
    """Render the graph dashboard to output/botfarm_graph.html."""
    config.configure_logging()
    config.ensure_directories()
    graph = load_graph()
    results = load_results()
    lookup = result_lookup(results)

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    filtered = filter_graph_for_viz(graph, lookup)

    network = build_network(filtered, results)
    config.DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    network.save_graph(str(config.DASHBOARD_PATH))
    legend = legend_html(
        node_count=filtered.number_of_nodes(),
        edge_count=filtered.number_of_edges(),
        total_nodes=total_nodes,
        total_edges=total_edges,
    )
    inject_legend(config.DASHBOARD_PATH, legend)
    print(
        f"[D] Dashboard saved → output/botfarm_graph.html "
        f"({filtered.number_of_nodes()}/{total_nodes} nodes, "
        f"{filtered.number_of_edges()}/{total_edges} edges shown)"
    )

    cluster_paths = render_cluster_dashboards(graph, results)
    for cluster_path in cluster_paths:
        print(f"[D] Cluster dashboard saved → {cluster_path.relative_to(config.BASE_DIR)}")

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
