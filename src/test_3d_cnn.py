import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import functional as F

from datasets import FrameVideoDataset


dataset_variant = os.getenv("DATASET_VARIANT", "original")
model_dir = (
    Path("experiments") / "ResNet3D" / dataset_variant / "final"
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
        cmap="Blues",
        vmin=0,
        vmax=100 if percent else None,
    )
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(cm)),
        yticks=range(len(cm)),
    )
    fig.colorbar(
        heatmap,
        ax=ax,
        label="Videos (%)" if percent else "Videos",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_video_transform(weights):
    """
    Transform a single PIL image into a normalised tensor.

    FrameVideoDataset applies this transform separately
    to every frame and stacks them into [C, T, H, W].
    """

    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    def transform(frame):
        # PIL Image -> [C, H, W]
        frame = F.to_tensor(frame)

        # Match the preprocessing used during training
        frame = F.resize(frame, [128, 171])
        frame = F.center_crop(frame, [112, 112])
        frame = F.normalize(frame, mean=mean, std=std)

        return frame

    return transform


def build_model(num_classes):
    model = r3d_18(weights=None)
    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes,
    )
    return model


runs = {}

for config_path in model_dir.glob("*/config.json"):
    experiment_dir = config_path.parent
    checkpoint_path = experiment_dir / "best_model.pt"

    if not checkpoint_path.is_file():
        print(f"Skipping incomplete run: {experiment_dir.name}")
        continue

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    seed = int(config["seed"])

    if (
        seed not in runs
        or checkpoint_path.stat().st_mtime
        > (runs[seed][0] / "best_model.pt").stat().st_mtime
    ):
        runs[seed] = (experiment_dir, config)

missing_seeds = [seed for seed in seeds if seed not in runs]

if missing_seeds:
    raise FileNotFoundError(
        f"No trained checkpoint found for seeds: {missing_seeds}"
    )

for seed in seeds:
    print(f"Seed {seed}: {runs[seed][0].name}")


seed_results = []

for seed in seeds:
    experiment_dir, config = runs[seed]

    checkpoint = torch.load(
        experiment_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )

    model = build_model(
        num_classes=config["num_classes"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    weights_name = config.get("weights", "DEFAULT")
    weights = R3D_18_Weights[weights_name]

    transform = make_video_transform(weights)

    dataset = FrameVideoDataset(
        root_dir=os.getenv(
            "DATA_DIR", config["data_dir"]
        ),
        split="test",
        transform=transform,
        stack_frames=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    video_names = [
        Path(path).stem
        for path in dataset.video_paths
    ]

    video_info = (
        dataset.df.set_index("video_name")
        .loc[video_names, ["action"]]
        .reset_index()
    )

    video_info["group"] = (
        video_info["video_name"]
        .str.extract(r"_(g\d+)_", expand=False)
    )

    rows = []

    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device)

            # Expected: [B, 3, T, H, W]
            logits = model(videos)

            predictions = logits.argmax(dim=1)

            rows.extend(
                zip(
                    labels.tolist(),
                    predictions.cpu().tolist(),
                )
            )

    predictions = pd.concat(
        [
            video_info,
            pd.DataFrame(
                rows,
                columns=["true_label", "rgb_pred"],
            ),
        ],
        axis=1,
    )

    predictions["seed"] = seed

    predictions.to_csv(
        experiment_dir / "predictions.csv",
        index=False,
    )

    result = predictions.assign(
        correct=predictions["rgb_pred"].eq(
            predictions["true_label"]
        )
    )

    seed_result = {
        "seed": seed,
        "rgb": result["correct"].mean(),
    }

    seed_results.append(seed_result)

    pd.DataFrame([seed_result]).to_csv(
        experiment_dir / "summary.csv",
        index=False,
    )

    for report, columns in [
        ("per_class", ["true_label", "action"]),
        ("per_group", ["action", "group"]),
    ]:
        result.groupby(columns)["correct"].agg(
            total="size",
            accuracy="mean",
        ).to_csv(
            experiment_dir / f"rgb_{report}.csv"
        )

    num_classes = config["num_classes"]

    cm = pd.crosstab(
        result["true_label"],
        result["rgb_pred"],
    ).reindex(
        index=range(num_classes),
        columns=range(num_classes),
        fill_value=0,
    )

    cm.to_csv(
        experiment_dir / "rgb_confusion_matrix.csv"
    )

    save_cm(
        cm,
        experiment_dir / "rgb_confusion_matrix.png",
        f"3D ResNet - seed {seed}",
    )


per_seed = pd.DataFrame(
    seed_results
).set_index("seed")

per_seed.to_csv(
    model_dir / "test_per_seed.csv"
)

stats = per_seed.agg(["mean", "std"]).T

stats.to_csv(
    model_dir / "test_mean_std.csv"
)

print(
    f"3D ResNet: "
    f"{100 * stats.loc['rgb', 'mean']:.2f}% "
    f"± {100 * stats.loc['rgb', 'std']:.2f}%"
)


for report, columns in [
    ("per_class", ["true_label", "action"]),
    ("per_group", ["action", "group"]),
]:
    tables = [
        pd.read_csv(
            runs[seed][0] / f"rgb_{report}.csv"
        )
        for seed in seeds
    ]

    summary = (
        pd.concat(tables)
        .groupby(columns)
        .agg(
            total=("total", "first"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
    )

    summary.to_csv(
        model_dir / f"rgb_{report}_mean_std.csv"
    )


matrices = [
    pd.read_csv(
        runs[seed][0] / "rgb_confusion_matrix.csv",
        index_col=0,
    )
    for seed in seeds
]

matrices = pd.concat(
    [
        cm.div(
            cm.sum(axis=1).replace(0, float("nan")),
            axis=0,
        )
        for cm in matrices
    ]
)

cm_mean = matrices.groupby(level=0).mean()
cm_std = matrices.groupby(level=0).std()

cm_mean.to_csv(
    model_dir / "rgb_confusion_matrix_mean.csv"
)

cm_std.to_csv(
    model_dir / "rgb_confusion_matrix_std.csv"
)

save_cm(
    cm_mean,
    model_dir / "rgb_confusion_matrix_mean.png",
    "3D ResNet - mean over seeds",
    percent=True,
)
