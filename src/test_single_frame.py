import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset
from models import ResNet50Stream

dataset_variant = os.getenv("DATASET_VARIANT", "noleakage")
model_dir = (
    Path("experiments") / "SingleFrameResNet50" / dataset_variant / "final"
)
seeds = [0, 42, 666]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def save_cm(cm, path, title, percent=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.imshow(
        100 * cm if percent else cm,
        cmap="Blues", vmin=0, vmax=100 if percent else None,
    )
    ax.set(
        title=title, xlabel="Predicted class", ylabel="True class",
        xticks=range(len(cm)), yticks=range(len(cm)),
    )
    fig.colorbar(heatmap, ax=ax, label="Videos (%)" if percent else "Videos")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


runs = {}
for config_path in model_dir.glob("*/config.json"):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs[config["seed"]] = (config_path.parent, config)

seed_results = []
for seed in seeds:
    experiment_dir, config = runs[seed]
    checkpoint = torch.load(
        experiment_dir / "best_model.pt",
        map_location=device, weights_only=True,
    )
    model = ResNet50Stream(
        num_classes=config["num_classes"], weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = ResNet50_Weights[config["weights"]].transforms()
    dataset = FrameVideoDataset(
        root_dir=os.getenv("DATA_DIR", config["data_dir"]),
        split="test", transform=transform, stack_frames=True,
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False,
    )
    video_names = [Path(path).stem for path in dataset.video_paths]
    video_info = (
        dataset.df.set_index("video_name")
        .loc[video_names, ["action"]].reset_index()
    )
    video_info["group"] = video_info["video_name"].str.extract(
        r"_(g\d+)_", expand=False,
    )

    rows = []
    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device)
            b, c, t, h, w = videos.shape
            frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            probabilities = (
                model(frames).softmax(dim=1).reshape(b, t, -1).mean(dim=1)
            )
            rows.extend(zip(
                labels.tolist(), probabilities.argmax(dim=1).cpu().tolist(),
            ))

    predictions = pd.concat([
        video_info,
        pd.DataFrame(rows, columns=["true_label", "rgb_pred"]),
    ], axis=1)
    predictions["seed"] = seed
    predictions.to_csv(experiment_dir / "predictions.csv", index=False)

    result = predictions.assign(
        correct=predictions["rgb_pred"].eq(predictions["true_label"])
    )
    seed_result = {"seed": seed, "rgb": result["correct"].mean()}
    seed_results.append(seed_result)
    pd.DataFrame([seed_result]).to_csv(
        experiment_dir / "summary.csv", index=False,
    )

    for report, columns in [
        ("per_class", ["true_label", "action"]),
        ("per_group", ["action", "group"]),
    ]:
        result.groupby(columns)["correct"].agg(
            total="size", accuracy="mean",
        ).to_csv(experiment_dir / f"rgb_{report}.csv")

    num_classes = config["num_classes"]
    cm = pd.crosstab(result["true_label"], result["rgb_pred"]).reindex(
        index=range(num_classes), columns=range(num_classes), fill_value=0,
    )
    cm.to_csv(experiment_dir / "rgb_confusion_matrix.csv")
    save_cm(
        cm, experiment_dir / "rgb_confusion_matrix.png", f"RGB - seed {seed}",
    )

per_seed = pd.DataFrame(seed_results).set_index("seed")
per_seed.to_csv(model_dir / "test_per_seed.csv")
stats = per_seed.agg(["mean", "std"]).T
stats.to_csv(model_dir / "test_mean_std.csv")
print(
    f"RGB: {100 * stats.loc['rgb', 'mean']:.2f}% "
    f"± {100 * stats.loc['rgb', 'std']:.2f}%"
)

for report, columns in [
    ("per_class", ["true_label", "action"]),
    ("per_group", ["action", "group"]),
]:
    tables = [
        pd.read_csv(runs[seed][0] / f"rgb_{report}.csv") for seed in seeds
    ]
    summary = pd.concat(tables).groupby(columns).agg(
        total=("total", "first"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    summary.to_csv(model_dir / f"rgb_{report}_mean_std.csv")

matrices = [
    pd.read_csv(runs[seed][0] / "rgb_confusion_matrix.csv", index_col=0)
    for seed in seeds
]
matrices = pd.concat([cm.div(cm.sum(axis=1), axis=0) for cm in matrices])
cm_mean = matrices.groupby(level=0).mean()
cm_std = matrices.groupby(level=0).std()
cm_mean.to_csv(model_dir / "rgb_confusion_matrix_mean.csv")
cm_std.to_csv(model_dir / "rgb_confusion_matrix_std.csv")
save_cm(
    cm_mean, model_dir / "rgb_confusion_matrix_mean.png",
    "RGB - mean over seeds", percent=True,
)