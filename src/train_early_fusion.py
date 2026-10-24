import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset
from models import EarlyFusionResNet50

model_name = "EarlyFusionResNet50"
dataset_variant = os.getenv("DATASET_VARIANT", "original")
default_dirs = {
    "original": "Data/ufc101",
    "noleakage": "Data/ucf101_noleakage",
}
data_dir = os.getenv("DATA_DIR", default_dirs[dataset_variant])

BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0001 ##learning rate.
WEIGHTS = ResNet50_Weights.DEFAULT ##pretrained imagenet weights;
SEEDS = [0, 42, 666]
N_FRAMES = 10  # must match n_sampled_frames in FrameVideoDataset


#Training
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = nn.functional.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def train_one_run(seed, phase="final"):
    started = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    config = {
        "model_name": model_name,
        "dataset_variant": dataset_variant,
        "data_dir": data_dir,
        "seed": seed,
        "phase": phase,
        "num_classes": 10,
        "weights": WEIGHTS.name,
        "frame_aggregation": "channel_stacking",
        "n_sampled_frames": N_FRAMES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "device": device,
    }
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("experiments") / model_name / dataset_variant / phase / f"{run_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    print("Saving run to:", run_dir)

    transform = WEIGHTS.transforms()
    train_ds = FrameVideoDataset(data_dir, split='train', transform=transform, stack_frames=True)
    val_ds = FrameVideoDataset(data_dir, split='val', transform=transform, stack_frames=True)
    assert train_ds.n_sampled_frames == N_FRAMES, "N_FRAMES must match the dataset"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    model = EarlyFusionResNet50(num_frames=N_FRAMES, num_classes=config["num_classes"], weights=WEIGHTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_acc, best_epoch = -1.0, 0
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch + 1, "loss": tr_loss, "train_video_accuracy": tr_acc,
            "val_loss": va_loss, "val_video_accuracy": va_acc, "learning_rate": LR,
        })
        pd.DataFrame(history).to_csv(run_dir / "stats.csv", index=False)
        print(f'epoch {epoch+1:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} '
              f'| val loss {va_loss:.3f} acc {va_acc:.3f}')
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "epoch": best_epoch,
                "val_accuracy": best_acc,
            }, run_dir / "best_model.pt")

    pd.DataFrame(history).plot(x="epoch", subplots=True, figsize=(8, 10))
    plt.tight_layout()
    plt.savefig(run_dir / "plots.png")
    plt.close()

    summary = {
        "seed": seed, "best_epoch": best_epoch,
        "best_val_video_accuracy": best_acc,
        "run_seconds": time.perf_counter() - started,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "train_summary.csv", index=False)
    print(f'seed {seed}: best val acc {best_acc:.3f} (epoch {best_epoch})')


if __name__ == '__main__':
    for seed in SEEDS:
        train_one_run(seed)