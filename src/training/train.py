from pathlib import Path

import torch
import torch.nn as nn
from dataset import TemporalGraphDataset
from model import TemporalGCN
from torch.utils.data import ConcatDataset, DataLoader

DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/flow-prediction-model/checkpoints")
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = DRIVE_OUTPUT_DIR / "temporal_gnn_checkpoint.pt"
FINAL_MODEL_PATH = DRIVE_OUTPUT_DIR / "temporal_gnn_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

kwargs_loader = {"pin_memory": True} if torch.cuda.is_available() else {}

PROCESSED_DIR_FEATURES = (
    Path("/content/drive/MyDrive/data/processed/features")
    if device == torch.device("cuda")
    else Path("data/processed/features")
)

PROCESSED_DIR_GRAPH = (
    Path("/content/drive/MyDrive/data/processed/graph")
    if device == torch.device("cuda")
    else Path("data/processed/graph")
)

WINDOW_SIZE = 12
BATCH_SIZE = 32
HIDDEN_DIM = 64
FORECAST_HORIZON = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EPOCHS = 20
PATIENCE = 5

train_days = []
val_days = []
test_days = []

print("Normalizing data...")

for day in range(42):
    try:
        X_day = torch.load(PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt")
        train_days.append(X_day)
    except Exception as e:
        print(e)

for day in range(42, 51):
    try:
        X_day = torch.load(PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt")
        val_days.append(X_day)
    except Exception as e:
        print(e)

for day in range(51, 60):
    try:
        X_day = torch.load(PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt")
        test_days.append(X_day)
    except Exception as e:
        print(e)

edge_index = torch.load(PROCESSED_DIR_GRAPH / "edge_index.pt")

X_train_all = torch.cat(train_days, dim=0)

mean = X_train_all.mean(dim=(0, 1), keepdim=True)
std = X_train_all.std(dim=(0, 1), keepdim=True)

train_days = [(X - mean) / (std + 1e-8) for X in train_days]
val_days = [(X - mean) / (std + 1e-8) for X in val_days]
test_days = [(X - mean) / (std + 1e-8) for X in test_days]

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

train_dataset = ConcatDataset(train_datasets)
val_dataset = ConcatDataset(val_datasets)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, **kwargs_loader
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, **kwargs_loader
)

num_features = train_days[0].shape[-1]

model = TemporalGCN(
    num_features=num_features,
    hidden_dim=HIDDEN_DIM,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON,
    dropout=DROPOUT,
).to(device)

print("Initializing optimizer...")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

criterion = nn.MSELoss()
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
edge_index = edge_index.to(device)

start_epoch = 0
best_val_loss = float("inf")
patience_counter = 0

if CHECKPOINT_PATH.exists():
    print(f"Checkpoint found at {CHECKPOINT_PATH}. Loading progress...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resuming training from Epoch {start_epoch + 1}")
else:
    print("No checkpoint found. Initializing training from scratch.")

print("Start training...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = model(x, edge_index).squeeze(-1)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x, edge_index).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    pred = model(x, edge_index).squeeze(-1)
                    loss = criterion(pred, y)
            else:
                pred = model(x, edge_index).squeeze(-1)
                loss = criterion(pred, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS} | train={train_loss:.10f} | val={val_loss:.10f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"--> Validation loss improved to {val_loss:.8f}. Saving best checkpoint...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
            CHECKPOINT_PATH,
        )
    else:
        patience_counter += 1
        print(f"--> No validation improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at Epoch {epoch + 1}!")
            break

if CHECKPOINT_PATH.exists():
    best_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    CHECKPOINT_PATH.unlink()
    print("Temporary checkpoint removed.")

torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Best final model saved successfully to {FINAL_MODEL_PATH}.")
