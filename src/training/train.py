from pathlib import Path
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
)

from dataset import TemporalGraphDataset
from model import SimpleTemporalGNN

PROCESSED_DIR_FEATURES = Path("data/processed/features")
PROCESSED_DIR_GRAPH = Path("data/processed/graph")

WINDOW_SIZE = 12
BATCH_SIZE = 8
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 20

train_days = []
val_days = []
test_days = []

for day in range(8):
    X_day = torch.load(
        PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
    )

    train_days.append(X_day)

for day in range(8, 10):
    X_day = torch.load(
        PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
    )

    val_days.append(X_day)

for day in range(10, 12):
    X_day = torch.load(
        PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
    )

    test_days.append(X_day)

edge_index = torch.load(
    PROCESSED_DIR_GRAPH / "edge_index.pt"
)

X_train_all = torch.cat(
    train_days,
    dim=0
)

mean = X_train_all.mean(
    dim=(0, 1),
    keepdim=True
)

std = X_train_all.std(
    dim=(0, 1),
    keepdim=True
)

train_days = [
    (X - mean) / (std + 1e-8)
    for X in train_days
]

val_days = [
    (X - mean) / (std + 1e-8)
    for X in val_days
]

test_days = [
    (X - mean) / (std + 1e-8)
    for X in test_days
]

train_datasets = [
    TemporalGraphDataset(
        X_day,
        window_size=WINDOW_SIZE,
        target_feature_idx=0,
    )
    for X_day in train_days
]

val_datasets = [
    TemporalGraphDataset(
        X_day,
        window_size=WINDOW_SIZE,
        target_feature_idx=0,
    )
    for X_day in val_days
]

train_dataset = ConcatDataset(
    train_datasets
)

val_dataset = ConcatDataset(
    val_datasets
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

num_features = train_days[0].shape[-1]

model = SimpleTemporalGNN(
    num_features=num_features,
    hidden_dim=HIDDEN_DIM,
    window_size=WINDOW_SIZE,
).to("cpu")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.MSELoss()

edge_index = edge_index.to("cpu")

for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0

    for x, y in train_loader:
        x = x.to("cpu")
        y = y.to("cpu")

        optimizer.zero_grad()

        pred = model(x, edge_index)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to("cpu")
            y = y.to("cpu")

            pred = model(x, edge_index)

            loss = criterion(pred, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train={train_loss:.4f} | "
        f"val={val_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "temporal_gnn.pt"
)

print("Model saved.")