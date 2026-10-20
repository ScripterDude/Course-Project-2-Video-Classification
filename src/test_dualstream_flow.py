from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import json
from torchvision.models import ResNet50_Weights
from datasets import FlowVideoDataset, FrameVideoDataset, root_dir
from models import DualStreamResNet50

#find the best val video acc
model_name = "DualStreamResNet50"
model_dir = Path("experiments") / model_name
# root_dir = Path("Data/ucf101_noleakage")
best_val_accuracy = -1.0
experiment_dir = None

for stats_path in sorted(model_dir.glob("*/stats.csv")):
    folder = stats_path.parent

    stats = pd.read_csv(stats_path)
    val_accuracy = stats["val_video_accuracy"].max()

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        experiment_dir = folder

print("Selected experiment:", experiment_dir)
print(f"Best validation video accuracy: {100 * best_val_accuracy:.1f}%")

#load config
config_path = experiment_dir / "config.json"
with open(config_path, "r") as f:
    config = json.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint = torch.load(experiment_dir / "best_model.pt",map_location=device,weights_only=True)
model = DualStreamResNet50(num_classes=config["num_classes"], flow_channels=config["flow_channels"], rgb_weight=config["rgb_weight"], weights=None)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# load up test
rgb_transform = ResNet50_Weights[config["weights"]].transforms()

rgb_test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=rgb_transform, stack_frames=True)
flow_test_dataset = FlowVideoDataset(root_dir=root_dir, split="test")

rgb_test_loader = DataLoader(rgb_test_dataset, batch_size=config["batch_size"], shuffle=False)
flow_test_loader = DataLoader(flow_test_dataset, batch_size=config["batch_size"], shuffle=False)

rgb_correct = 0
flow_correct = 0
correct = 0
total = 0

with torch.inference_mode():
    for (videos, labels), (flows, flow_labels) in zip(rgb_test_loader, flow_test_loader):
        if not torch.equal(labels, flow_labels):
            raise ValueError("RGB and flow test labels do not match.")

        videos = videos.to(device)
        flows = flows.to(device).clamp(-config["flow_bound"], config["flow_bound"]) / config["flow_bound"]
        labels = labels.to(device)

        # [B, 3, T, H, W] -> [B*T, 3, H, W]
        b, c, t, h, w = videos.shape
        frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # mean RGB probabilities across frames then fuse with flow
        rgb_probs = model.rgb_stream(frames).softmax(dim=1).reshape(b, t, -1).mean(dim=1)
        flow_probs = model.flow_stream(flows).softmax(dim=1)
        probabilities = model.rgb_weight * rgb_probs + (1.0 - model.rgb_weight) * flow_probs

        rgb_correct += (rgb_probs.argmax(dim=1) == labels).sum().item()
        flow_correct += (flow_probs.argmax(dim=1) == labels).sum().item()
        correct += (probabilities.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

rgb_accuracy = rgb_correct / total
flow_accuracy = flow_correct / total
test_accuracy = correct / total

print(f"RGB test video accuracy: {100 * rgb_accuracy:.1f}% ({rgb_correct}/{total})")
print(f"Flow test video accuracy: {100 * flow_accuracy:.1f}% ({flow_correct}/{total})")
print(f"Fused test video accuracy: {100 * test_accuracy:.1f}% ({correct}/{total})")

results = {
    "epoch": checkpoint["epoch"],
    "rgb_test_video_accuracy": rgb_accuracy,
    "flow_test_video_accuracy": flow_accuracy,
    "test_video_accuracy": test_accuracy,
    "rgb_correct": rgb_correct,
    "flow_correct": flow_correct,
    "correct": correct,
    "total": total,
}
pd.DataFrame([results]).to_csv(experiment_dir / "test_stats.csv", index=False)