from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

for file_idx in range(12, 69):
    FLOW_CSV = f'data/processed/flow/data{file_idx}.csv'
    PROCESSED_DIR = Path("data/processed/graph")

    with open(PROCESSED_DIR / "edge_to_idx.pkl", "rb") as f:
        edge_to_idx = pickle.load(f)

    NUM_EDGES = len(edge_to_idx)
    static_features = torch.load(
        PROCESSED_DIR / "static_features.pt"
    )

    df = pd.read_csv(FLOW_CSV)
    timestamp = df["time"].unique()

    NUM_DYNAMIC_FEATURES = 3
    NUM_STATIC_FEATURES = 2

    NUM_FEATURES = (
        NUM_DYNAMIC_FEATURES +
        NUM_STATIC_FEATURES
    )

    all_snapshots = []

    for i, t in enumerate(timestamp):
        if i % 1000 == 0:
            print(f"{i}/{len(timestamp)}")

        df_t = df[df["time"] == t]

        dynamic = np.zeros(
            (NUM_EDGES, NUM_DYNAMIC_FEATURES),
            dtype=np.float32
        )

        for _, row in df_t.iterrows():
            u = row["u"]
            v = row["v"]

            edge = tuple(sorted((u, v)))
            idx = edge_to_idx[edge]

            dynamic[idx] = [
                float(row["in_transit"]),
                float(row["queue"]),
                float(row["entered"]),
            ]

        dynamic_tensor = torch.tensor(
            dynamic,
            dtype=torch.float32
        )

        snapshot = torch.cat(
            [
                dynamic_tensor,
                static_features,
            ],
            dim=1
        )

        all_snapshots.append(snapshot)

    X = torch.stack(all_snapshots)

    torch.save(
        X,
        PROCESSED_DIR / f'dynamic_features-{file_idx}.pt'
    )