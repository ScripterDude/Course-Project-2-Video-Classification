
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import functional as TF

from datasets import FrameVideoDataset


model_name = "ResNet3D"
dataset_variant = os.getenv("DATASET_VARIANT", "original")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

default_dirs = {
    "original": "Data/ufc101",
    "noleakage": "Data/ucf101_noleakage",
}


data_dir = os.getenv(
    "DATA_DIR",
    str(PROJECT_ROOT / default_dirs[dataset_variant])
)


class VideoFrameTransform:
    """
    Preprocess each RGB frame for pretrained R3D-18.

    FrameVideoDataset is expected to apply this transform
    separately to each frame before stacking.
    """

    def __call__(self, frame):
        if not isinstance(frame, torch.Tensor):
            frame = TF.to_tensor(frame)
        else:
            frame = TF.convert_image_dtype(
                frame, torch.float32
            )

        frame = TF.resize(frame, [128, 171])
        frame = TF.center_crop(frame, [112, 112])

        frame = TF.normalize(
            frame,
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        )

        return frame


def build_model(num_classes, weights, trainable):
    model = r3d_18(weights=weights)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    if trainable == "layer4,fc":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(
                ("layer4.", "fc.")
            )

    elif trainable != "all":
        raise ValueError(
            f"Unknown trainable setting: {trainable}"
        )

    return model


def train_one_run(seed, phase, overrides=None):
    started = time.perf_counter()

    run = wandb.init(
        project="idlcv-project2",
        config={
            "model_name": model_name,
            "dataset_variant": dataset_variant,
            "data_dir": data_dir,
            "seed": seed,
            "phase": phase,
            "num_classes": 10,
            "weights": R3D_18_Weights.DEFAULT.name,
            "trainable": "layer4,fc",
            "resize": [112, 112],
            "n_sampled_frames": 10,
            "epochs": 100,
            "patience": 10,
            "batch_size": 4,
            "learning_rate": 0.00003,
            "weight_decay": 0.0001,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "scheduler_min_lr": 1e-6,
            **(overrides or {}),
        },
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    run.config.update({
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "torch_version": str(torch.__version__),
    })

    config = dict(run.config)

    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    run_dir = (
        PROJECT_ROOT
        / "experiments"
        / model_name
        / config["dataset_variant"]
        / config["phase"]
        / f"{run.id}_seed{config['seed']}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("Saving run to:", run_dir)

    weights = R3D_18_Weights[config["weights"]]
    transform = VideoFrameTransform()
    batch_size = int(config["batch_size"])

    train_dataset = FrameVideoDataset(
        root_dir=config["data_dir"],
        split="train",
        transform=transform,
        stack_frames=True,
    )

    val_dataset = FrameVideoDataset(
        root_dir=config["data_dir"],
        split="val",
        transform=transform,
        stack_frames=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(
        num_classes=config["num_classes"],
        weights=weights,
        trainable=config["trainable"],
    ).to(device)

    optimizer = torch.optim.Adam(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    history = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(int(config["epochs"])):

        # TRAINING
        model.train()

        if config["trainable"] == "layer4,fc":
            model.eval()
            model.layer4.train()
            model.fc.train()

        learning_rate = optimizer.param_groups[0]["lr"]

        total_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            # Expected [B, 3, T, H, W]
            if videos.ndim != 5 or videos.shape[1] != 3:
                raise ValueError(
                    f"Expected [B,3,T,H,W], got {videos.shape}"
                )

            optimizer.zero_grad()

            logits = model(videos)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            current_batch_size = labels.size(0)

            total_loss += (
                loss.item() * current_batch_size
            )

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += current_batch_size

        # VALIDATION
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                logits = model(videos)
                predictions = logits.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        train_accuracy = correct / total
        val_accuracy = val_correct / val_total
        avg_loss = total_loss / total

        scheduler.step(val_accuracy)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_video_accuracy": train_accuracy,
            "val_video_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        })

        pd.DataFrame(history).to_csv(
            run_dir / "stats.csv",
            index=False,
        )

        run.log({
            **history[-1],
            "best_val_video_accuracy": max(
                best_val_accuracy,
                val_accuracy,
            ),
        })

        print(
            f"Epoch {epoch + 1} | "
            f"loss {avg_loss:.4f} | "
            f"train {100 * train_accuracy:.1f}% | "
            f"validation {100 * val_accuracy:.1f}%"
        )

        # SAVE BEST MODEL
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

            if (
                epochs_without_improvement
                >= config["patience"]
            ):
                print("Early stopping triggered.")
                break

    # SAVE TRAINING PLOTS
    pd.DataFrame(history).plot(
        x="epoch",
        subplots=True,
        figsize=(8, 10),
    )

    plt.tight_layout()
    plt.savefig(run_dir / "plots.png")
    plt.close()

    summary = {
        "seed": config["seed"],
        "best_epoch": best_epoch,
        "best_val_video_accuracy": best_val_accuracy,
        "run_seconds": time.perf_counter() - started,
    }

    pd.DataFrame([summary]).to_csv(
        run_dir / "train_summary.csv",
        index=False,
    )

    run.summary.update(summary)
    run.finish()

    return summary


if __name__ == "__main__":

    if os.getenv("WANDB_SWEEP_ID"):
        train_one_run(0, "search")

    else:
        search_dir = (
            PROJECT_ROOT
            / "experiments"
            / model_name
            / dataset_variant
            / "search"
        )

        search_stats = sorted(
            search_dir.glob("*/stats.csv")
        )

        overrides = {}

        if search_stats:
            best_stats = max(
                search_stats,
                key=lambda path: pd.read_csv(
                    path
                )["val_video_accuracy"].max(),
            )

            best_config = json.loads(
                (
                    best_stats.parent / "config.json"
                ).read_text(encoding="utf-8")
            )

            overrides = {
                key: best_config[key]
                for key in (
                    "learning_rate",
                    "weight_decay",
                    "batch_size",
                )
            }

            print(
                "Selected search run:",
                best_stats.parent,
            )
            print(
                "Selected hyperparameters:",
                overrides,
            )

        else:
            print(
                "No hyperparameter search found. "
                "Using default hyperparameters."
            )

        summaries = [
            train_one_run(seed, "final", overrides)
            for seed in [0, 42, 666]
        ]

        summary_df = pd.DataFrame(summaries)

        final_dir = (
            PROJECT_ROOT
            / "experiments"
            / model_name
            / dataset_variant
            / "final"
        )

        summary_df.to_csv(
            final_dir / "train_per_seed.csv",
            index=False,
        )

        print("\nFinal validation results:")
        print(summary_df)
        print(
            "Mean validation accuracy:",
            summary_df[
                "best_val_video_accuracy"
            ].mean(),
        )
