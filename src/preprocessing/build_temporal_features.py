from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

LOCAL_RUN = False

if LOCAL_RUN:
    PROCESSED_DIR = Path("data/processed/features")
    GRAPH_DIR = Path("data/processed/graph")
    FLOW_DIR = Path("data/processed/flow")
    N_JOBS = 1
else:    
    DRIVE_DIR = Path("/content/drive/MyDrive")
    GRAPH_DIR = DRIVE_DIR / "data/processed/graph"
    PROCESSED_DIR = DRIVE_DIR / "data/processed/features"
    FLOW_DIR = DRIVE_DIR / "data/processed/flow"
    N_JOBS = -1


with open(GRAPH_DIR / "edge_to_idx.pkl", "rb") as f:
    edge_to_idx = pickle.load(f)

NUM_EDGES = len(edge_to_idx)
static_features = torch.load(GRAPH_DIR / "static_features.pt")
NUM_DYNAMIC_FEATURES = 3

def process_single_file(file_idx):
    flow_csv = FLOW_DIR / f'data{file_idx}.csv'
    output_path = PROCESSED_DIR / f'dynamic_features-{file_idx}.pt'
    
    if not flow_csv.exists():
        print(f"File {flow_csv.name} not found.")
        return

    df = pd.read_csv(flow_csv)
    all_snapshots = []
    
    grouped = df.groupby("time")
    
    for t, df_t in grouped:
        dynamic = np.zeros((NUM_EDGES, NUM_DYNAMIC_FEATURES), dtype=np.float32)
        
        for _, row in df_t.iterrows():
            u = row["u"]
            v = row["v"]
            edge = tuple(sorted((u, v)))
            
            if edge in edge_to_idx:
                idx = edge_to_idx[edge]
                dynamic[idx] = [
                    float(row["in_transit"]),
                    float(row["queue"]),
                    float(row["entered"]),
                ]
        
        dynamic_tensor = torch.tensor(dynamic, dtype=torch.float32)
        snapshot = torch.cat([dynamic_tensor, static_features], dim=1)
        all_snapshots.append(snapshot)
    
    if all_snapshots:
        X = torch.stack(all_snapshots)
        torch.save(X, output_path)
        print(f"File {file_idx} saved successfully!")


if __name__ == "__main__":
    file_indices = range(12, 69)
    
    if N_JOBS == 1:
        for idx in file_indices:
            process_single_file(idx)
    else:
        Parallel(n_jobs=N_JOBS)(
            delayed(process_single_file)(idx) for idx in file_indices
        )
        
    print("Job finished!")