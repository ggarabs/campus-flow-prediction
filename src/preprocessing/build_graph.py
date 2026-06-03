from pathlib import Path
import pickle
import networkx as nx
import torch
from torch_geometric.utils import from_networkx

OUTPUT_DIR = Path("data/processed")
DISTANCE_GRAPH_PATH = "data/raw/graph/distances-graph.graphml"
WIDTH_GRAPH_PATH = "data/raw/graph/width-graph.graphml"

G_distance = nx.read_graphml(DISTANCE_GRAPH_PATH)
G_width = nx.read_graphml(WIDTH_GRAPH_PATH)

G = nx.Graph()

for u, v, data in G_distance.edges(data=True):
    distance = float(data.get("distance", 0.0))

    width_data = G_width.get_edge_data(u, v, default={})

    width = float(width_data.get("width", 1.0))

    edge = tuple(sorted((u, v)))

    u, v = edge

    G.add_edge(
        u,
        v,
        distance=distance,
        width=width,
)

LG = nx.line_graph(G)
line_nodes = [
    tuple(sorted(edge))
    for edge in LG.nodes()
]

edge_to_idx = {
    edge: idx
    for idx, edge in enumerate(line_nodes)
}
idx_to_edge = {
    idx: edge
    for edge, idx in edge_to_idx.items()
}

pyg_graph = from_networkx(LG)
edge_index = pyg_graph.edge_index

static_features = []
for edge in line_nodes:
    u, v = edge

    data = G[u][v]

    distance = float(data.get("distance", 0.0))
    width = float(data.get("width", 1.0))

    static_features.append([
        distance,
        width,
    ])

static_features = torch.tensor(
    static_features,
    dtype=torch.float32
)

print("static feature shape:", static_features.shape)

with open(OUTPUT_DIR / "line_graph.pkl", "wb") as f:
    pickle.dump(LG, f)

torch.save(
    edge_index,
    OUTPUT_DIR / "edge_index.pt"
)

torch.save(
    static_features,
    OUTPUT_DIR / "static_features.pt"
)

with open(OUTPUT_DIR / "edge_to_idx.pkl", "wb") as f:
    pickle.dump(edge_to_idx, f)

with open(OUTPUT_DIR / "idx_to_edge.pkl", "wb") as f:
    pickle.dump(idx_to_edge, f)

print("Files saved.")