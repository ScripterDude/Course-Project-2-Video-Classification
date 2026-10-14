from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import json
from datasets import FrameVideoDataset, root_dir
from models import SingleFrameCNN

#find the best val video acc
model_name = "SingleFrameCNN"
model_dir = Path("experiments") / model_name

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

model = SingleFrameCNN(num_classes=config["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# load up test
transform = T.Compose([T.Resize(config["resize"]),T.ToTensor()])

test_dataset = FrameVideoDataset(root_dir=root_dir,split="test",transform=transform,stack_frames=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


# test accuracy
correct = 0
total = 0

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        # [B, 3, T, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
        batch_size, channels, num_frames, h, w = videos.shape

        frames = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, h, w)

        logits = model(frames)
        frame_probs = torch.softmax(logits, dim=1).reshape(batch_size, num_frames, 10)

        video_probs = frame_probs.mean(dim=1)
        predictions = video_probs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

test_accuracy = correct / total

print(f"Loaded epoch: {checkpoint['epoch']}")
print(f"Test video accuracy: {100 * test_accuracy:.1f}% ({correct}/{total})")

pd.DataFrame([{"test_video_accuracy": test_accuracy,"correct": correct,"total": total}]).to_csv(experiment_dir / "test_stats.csv", index=False)