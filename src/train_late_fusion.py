import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset
from models import LateFusionResNet50
from validation_reports import save_validation_reports, summarize_validation_reports

model_name = "LateFusionResNet50"
dataset_variant = os.getenv("DATASET_VARIANT", "original")
default_dirs = {
    "original": "Data/ufc101",
    "noleakage": "Data/ucf101_noleakage",
}
data_dir = os.getenv("DATA_DIR", default_dirs[dataset_variant])


def train_one_run(seed, phase, overrides=None):
    started = time.perf_counter()
    run = wandb.init(project="idlcv-project2", config={
        "model_name": model_name,
        "dataset_variant": dataset_variant,
        "data_dir": data_dir,
        "seed": seed,
        "phase": phase,
        "num_classes": 10,
        "weights": ResNet50_Weights.DEFAULT.name,
        "rgb_trainable": "layer4,classifier",
        "resize": [224, 224],
        "n_sampled_frames": 10,
        "frame_aggregation": "mean_features",
        "epochs": 100,
        "patience": 10,
        "batch_size": 32,
        "learning_rate": 0.00003,
        "weight_decay": 0.0001,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 3,
        "scheduler_min_lr": 1e-6,
        **(overrides or {}),
    })

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    run.config.update({
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda" else str(device),
        "torch_version": str(torch.__version__),
    })
    config = dict(run.config)
    torch.manual_seed(int(config["seed"]))

    run_dir = (
        Path("experiments") / model_name / config["dataset_variant"]
        / config["phase"] / f"{run.id}_seed{config['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    print("Saving run to:", run_dir)

    weights = ResNet50_Weights[config["weights"]]
    transform = weights.transforms()
    batch_size = int(config["batch_size"])

    train_dataset = FrameVideoDataset(
        root_dir=config["data_dir"], split="train",
        transform=transform, stack_frames=True,
    )
    val_dataset = FrameVideoDataset(
        root_dir=config["data_dir"], split="val",
        transform=transform, stack_frames=True,
    )
    assert train_dataset.n_sampled_frames == config["n_sampled_frames"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = LateFusionResNet50(
        num_classes=config["num_classes"], weights=weights,
    ).to(device)
    # fine-tune only layer4 of the shared frame backbone; the classifier head stays trainable
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = name.startswith("layer4.")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(config["epochs"]):
        model.train()
        model.backbone.eval()          # frozen layers keep their BatchNorm statistics
        model.backbone.layer4.train()  # classifier stays in train mode: dropout active

        learning_rate = optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        correct = total = 0

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(videos)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += current_batch_size

        model.eval()
        val_correct = val_total = 0

        with torch.inference_mode():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                val_correct += (
                    model(videos).argmax(dim=1) == labels
                ).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        scheduler.step(val_accuracy)
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / total,
            "train_video_accuracy": correct / total,
            "val_video_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        })
        pd.DataFrame(history).to_csv(run_dir / "stats.csv", index=False)
        run.log({
            **history[-1],
            "best_val_video_accuracy": max(best_val_accuracy, val_accuracy),
        })
        print(
            f"Epoch {epoch + 1} | loss {total_loss / total:.4f} | "
            f"train {100 * correct / total:.1f}% | "
            f"validation {100 * val_accuracy:.1f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "epoch": best_epoch,
                "val_accuracy": best_val_accuracy,
            }, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break

    pd.DataFrame(history).plot(x="epoch", subplots=True, figsize=(8, 10))
    plt.tight_layout()
    plt.savefig(run_dir / "plots.png")
    plt.close()

    summary = {
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_val_video_accuracy": best_val_accuracy,
        "run_seconds": time.perf_counter() - started,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "train_summary.csv", index=False)
    run.summary.update(summary)

    validation_report = None
    if config["phase"] == "final":
        validation_report = save_validation_reports(
            model, val_loader, run_dir, device, clip_model=True,
        )

    run.finish()
    return validation_report


if __name__ == "__main__":
    if os.getenv("WANDB_SWEEP_ID"):
        train_one_run(0, "search")
    else:
        search_dir = (
            Path("experiments") / model_name / dataset_variant / "search"
        )
        best_stats = max(
            sorted(search_dir.glob("*/stats.csv")),
            key=lambda path: pd.read_csv(path)["val_video_accuracy"].max(),
        )
        best_config = json.loads(
            (best_stats.parent / "config.json").read_text(encoding="utf-8")
        )
        overrides = {
            key: best_config[key]
            for key in ("learning_rate", "weight_decay", "batch_size")
        }
        print("Selected search run:", best_stats.parent)
        print("Selected hyperparameters:", overrides)
        validation_reports = [train_one_run(seed, "final", overrides) for seed in [0, 42, 666]]
        summarize_validation_reports(validation_reports)