"""Road-network helpers using GraphML and standard library only."""

from __future__ import annotations

import heapq
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .geo import haversine_km


@dataclass(frozen=True)
class Node:
    node_id: str
    lng: float
    lat: float


class RoadNetwork:
    def __init__(self, nodes: dict[str, Node], adjacency: dict[str, list[tuple[str, float]]]):
        self.nodes = nodes
        self.adjacency = adjacency

    @classmethod
    def from_graphml(cls, path: Path, max_nodes: int | None = None) -> "RoadNetwork":
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        root = ET.parse(path).getroot()
        keys = {k.attrib["id"]: k.attrib.get("attr.name") for k in root.findall("g:key", ns)}
        graph = root.find("g:graph", ns)
        if graph is None:
            raise ValueError("GraphML does not contain a graph element")

        nodes: dict[str, Node] = {}
        for elem in graph.findall("g:node", ns):
            if max_nodes and len(nodes) >= max_nodes:
                break
            data = {keys.get(d.attrib["key"], d.attrib["key"]): d.text for d in elem.findall("g:data", ns)}
            if data.get("x") and data.get("y"):
                nodes[elem.attrib["id"]] = Node(elem.attrib["id"], float(data["x"]), float(data["y"]))

        adjacency: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in nodes}
        for edge in graph.findall("g:edge", ns):
            source = edge.attrib["source"]
            target = edge.attrib["target"]
            if source not in nodes or target not in nodes:
                continue
            data = {keys.get(d.attrib["key"], d.attrib["key"]): d.text for d in edge.findall("g:data", ns)}
            try:
                length = float(data.get("length") or 0) / 1000
            except ValueError:
                length = 0
            if length <= 0:
                a, b = nodes[source], nodes[target]
                length = haversine_km(a.lng, a.lat, b.lng, b.lat)
            adjacency.setdefault(source, []).append((target, length))
            adjacency.setdefault(target, []).append((source, length))
        return cls(nodes, adjacency)

    def nearest_node(self, lng: float, lat: float) -> tuple[Node, float]:
        best: tuple[Node, float] | None = None
        for node in self.nodes.values():
            distance = haversine_km(lng, lat, node.lng, node.lat)
            if best is None or distance < best[1]:
                best = (node, distance)
        if best is None:
            raise ValueError("Road network is empty")
        return best

    def shortest_distance_km(self, start_id: str, end_id: str) -> float | None:
        queue = [(0.0, start_id)]
        seen: dict[str, float] = {}
        while queue:
            distance, node_id = heapq.heappop(queue)
            if node_id in seen:
                continue
            seen[node_id] = distance
            if node_id == end_id:
                return distance
            for nxt, weight in self.adjacency.get(node_id, []):
                if nxt not in seen:
                    heapq.heappush(queue, (distance + weight, nxt))
        return None


def eta_from_points(graphml: Path, start: tuple[float, float], end: tuple[float, float], speed_kmh: float = 25.0) -> dict[str, object]:
    net = RoadNetwork.from_graphml(graphml)
    start_node, start_snap = net.nearest_node(*start)
    end_node, end_snap = net.nearest_node(*end)
    road_km = net.shortest_distance_km(start_node.node_id, end_node.node_id)
    if road_km is None:
        road_km = haversine_km(start[0], start[1], end[0], end[1])
        method = "fallback_haversine"
    else:
        method = "graph_shortest_path"
    eta_min = road_km / max(speed_kmh, 1.0) * 60
    return {
        "method": method,
        "start_node": start_node.node_id,
        "end_node": end_node.node_id,
        "start_snap_km": round(start_snap, 4),
        "end_snap_km": round(end_snap, 4),
        "distance_km": round(road_km, 3),
        "speed_kmh": speed_kmh,
        "eta_min": round(eta_min, 1),
    }


def write_eta_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
