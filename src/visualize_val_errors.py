import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset, FlowVideoDataset
from models import SingleFrameCNN, ResNet50Stream, DualStreamResNet50

# Choose which experiment directories to visualize.
final_dirs = [
    Path("experiments/SingleFrameResNet50/original/final"),
    Path("experiments/SingleFrameResNet50/noleakage/final"),
    Path("experiments/DualStreamResNet50/final"),
    Path("experiments/SingleFrameCNN/final"),
]
data_overrides = {}  # Optional: {"saved/dataset/path": "current/dataset/path"}
max_errors = 6
frame_indices = [0, 4, 9]
batch_size = 4

model_builders = {
    "SingleFrameCNN": lambda c: SingleFrameCNN(
        num_classes=c["num_classes"],
    ),
    "SingleFrameResNet50": lambda c: ResNet50Stream(
        num_classes=c["num_classes"], weights=None,
    ),
    "DualStreamResNet50": lambda c: DualStreamResNet50(
        num_classes=c["num_classes"], flow_channels=c["flow_channels"],
        rgb_weight=c["rgb_weight"], weights=None,
    ),
}


def predict_probabilities(model, videos, flows, config):
    if config["model_name"] == "DualStreamResNet50":
        flows = flows.to(videos.device)
        if config.get("subtract_flow_mean", False):
            flows = flows - flows.mean(dim=(-2, -1), keepdim=True)
        bound = config["flow_bound"]
        flows = flows.clamp(-bound, bound) / bound

        # DualStreamResNet50 already returns fused probabilities.
        return model(videos, flows)

    b, c, t, h, w = videos.shape
    frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return model(frames).softmax(dim=1).reshape(b, t, -1).mean(dim=1)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

config_paths = [
    config_path
    for final_dir in final_dirs
    for config_path in sorted(final_dir.glob("*/config.json"))
]

for config_path in config_paths:
    run_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config["model_name"]
    is_dual = model_name == "DualStreamResNet50"
    data_dir = data_overrides.get(config["data_dir"], config["data_dir"])
    dataset_name = config.get("dataset_variant", Path(data_dir).name)

    checkpoint = torch.load(
        run_dir / "best_model.pt", map_location=device, weights_only=True,
    )
    model = model_builders[model_name](config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if model_name == "SingleFrameCNN":
        transform = T.Compose([T.Resize(config["resize"]), T.ToTensor()])
        mean = torch.zeros(3, 1, 1)
        std = torch.ones(3, 1, 1)
    else:
        transform = ResNet50_Weights[config["weights"]].transforms()
        mean = torch.tensor(transform.mean).view(3, 1, 1)
        std = torch.tensor(transform.std).view(3, 1, 1)

    dataset = FrameVideoDataset(
        root_dir=data_dir, split="val",
        transform=transform, stack_frames=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if is_dual:
        flow_dataset = FlowVideoDataset(root_dir=data_dir, split="val")
        flow_batches = iter(DataLoader(
            flow_dataset, batch_size=batch_size, shuffle=False,
        ))

    class_names = (
        dataset.df.drop_duplicates("label").set_index("label")["action"].to_dict()
    )
    rows = []
    with torch.inference_mode():
        for videos, labels in loader:
            videos = videos.to(device)
            flows = next(flow_batches)[0] if is_dual else None
            probabilities = predict_probabilities(model, videos, flows, config)
            confidence, predicted = probabilities.max(dim=1)
            rows.extend(zip(
                labels.tolist(),
                predicted.cpu().tolist(),
                confidence.cpu().tolist(),
            ))

    predictions = pd.DataFrame(
        rows, columns=["true_label", "predicted_label", "confidence"],
    )
    predictions["video_index"] = predictions.index
    predictions["video_name"] = [Path(p).stem for p in dataset.video_paths]
    predictions["true_action"] = predictions["true_label"].map(class_names)
    predictions["predicted_action"] = predictions["predicted_label"].map(class_names)
    predictions["model_name"] = model_name
    predictions["dataset"] = dataset_name
    predictions["seed"] = config["seed"]
    predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    errors = predictions[
        predictions["true_label"] != predictions["predicted_label"]
    ].sort_values("confidence", ascending=False)
    errors.to_csv(run_dir / "val_errors.csv", index=False)
    print(
        f"{model_name} | {dataset_name} | {run_dir.name}: "
        f"{len(errors)}/{len(predictions)} validation errors"
    )

    shown = errors.head(max_errors)
    if shown.empty:
        continue

    fig, axes = plt.subplots(
        len(shown), 1, figsize=(12, 3 * len(shown)), squeeze=False,
    )
    for ax, row in zip(axes[:, 0], shown.itertuples(index=False)):
        video, _ = dataset[row.video_index]
        strip = torch.cat([video[:, i] for i in frame_indices], dim=2)
        image = (strip * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(
            f"{row.video_name}\n"
            f"True: {row.true_action} | Predicted: {row.predicted_action} | "
            f"Probability: {row.confidence:.1%}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        f"{model_name} - {dataset_name} - seed {config['seed']}\n"
        f"frames {', '.join(str(i + 1) for i in frame_indices)}"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "val_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)