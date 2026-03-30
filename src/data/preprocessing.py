import networkx as nx
import pandas as pd

def normalize_node(n):
    return str(n)

def edge_key(u, v):
    return tuple(sorted((normalize_node(u), normalize_node(v))))

G = nx.read_graphml("data/raw/distances-graph.graphml")
G1 = nx.read_graphml("data/raw/width-graph.graphml")

current_state = {}

for u, v, data in G.edges(data=True):
    key = edge_key(u, v)
    current_state[key] = {
        "in_transit": 0,
        "queue": 0,
        "entered": 0,
        "distance": data["label"],
        "width": 0  
    }

for u, v, data in G1.edges(data=True):
    key = edge_key(u, v)

    if key in current_state:
        current_state[key]["width"] = data["label"]
    else:
        print("aresta no width não existe:", key)

df = pd.read_csv("data/raw/flow.csv")

snapshots = []

for t, group in df.groupby("timestamp"):

    for _, row in group.iterrows():
        key = edge_key(row["u"], row["v"])

        if key not in current_state:
            print("aresta do CSV não existe:", key)
            continue

        current_state[key]["in_transit"] = row["in_transit"]
        current_state[key]["queue"] = row["in_queue"]
        current_state[key]["entered"] = row["entered_this_step"]

    for (u, v), data in current_state.items():
        snapshots.append({
            "time": t,
            "u": u,
            "v": v,
            **data
        })

df_final = pd.DataFrame(snapshots)
df_final.to_csv("/data/processed/data1.csv", index=False)