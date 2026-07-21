from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, ConcatDataset)

from dataset import TemporalGraphDataset
from model import TemporalGCN

DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/flow-prediction-model/checkpoints")
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = DRIVE_OUTPUT_DIR / "temporal_gnn_checkpoint.pt"
FINAL_MODEL_PATH = DRIVE_OUTPUT_DIR / "temporal_gnn_final.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running in device: {device}")

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
FORECAST_HORIZON = 1
LR = 1e-3
EPOCHS = 20
SAVE_EVERY_EPOCHS = 2

train_days = []
val_days = []
test_days = []

for day in range(42):
    try:
        X_day = torch.load(
            PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
        )
        train_days.append(X_day)
    except Exception as e:
        print(e)

for day in range(42, 51):
    try:
        X_day = torch.load(
            PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
        )
        val_days.append(X_day)
    except Exception as e:
        print(e)

for day in range(51, 60):
    try:
        X_day = torch.load(
            PROCESSED_DIR_FEATURES / f"dynamic_features-{day}.pt"
        )
        test_days.append(X_day)
    except Exception as e:
        print(e)

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
    num_workers=2,
    **kwargs_loader
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    **kwargs_loader
)

num_features = train_days[0].shape[-1]

model = TemporalGCN(
    num_features=num_features,
    hidden_dim=HIDDEN_DIM,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

criterion = nn.MSELoss()

scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

edge_index = edge_index.to(device)

start_epoch = 0

if CHECKPOINT_PATH.exists():
    print(f"Checkopoint finded at {CHECKPOINT_PATH}. Loading progress...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming training from Époch {start_epoch + 1}')
else:
    print("No checkpoint finded. Initializing training from begin.")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
                    with torch.amp.autocast('cuda'):
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

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred = model(x, edge_index).squeeze(-1)
                    loss = criterion(pred, y)
            else:
                pred = model(x, edge_index).squeeze(-1)
                loss = criterion(pred, y)
                
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"train={train_loss:.10f} | "
        f"val={val_loss:.10f}"
    )

    if (epoch + 1) % SAVE_EVERY_EPOCHS == 0 or (epoch + 1) == EPOCHS:
        print(f"Saving Epoch {epoch+1} checkpoint in Drive...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, CHECKPOINT_PATH)

print("Model saved successfully.")
torch.save(model.state_dict(), FINAL_MODEL_PATH)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("Temporary checkpoint removed. Final model saved.")