from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
from datasets import FrameImageDataset, FrameVideoDataset, root_dir
from models import SingleFrameCNN
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

# set hyperparams
seed = 0
torch.manual_seed(seed)
epochs = 30
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0001
resize = (64, 64)
patience = 5
best_val_accuracy = -1.0
epochs_without_improvement = 0

#log experiments
model_name = "SingleFrameCNN"
experiment_name = (f"seed{seed}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_{resize[0]}x{resize[1]}_pat{patience}_time{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

experiment_dir = Path("experiments") / model_name / experiment_name
run_dir = experiment_dir
run_dir.mkdir(parents=True, exist_ok=True)

#save conf:
model_config = {
    "model_name": model_name,
    "num_classes": 10,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "resize": resize,
    "patience": patience,
    "data_dir": root_dir,
}

(run_dir / "config.json").write_text(json.dumps(model_config))
history = []
print("Saving run to:", run_dir)

#prep data
transform = T.Compose([T.Resize(resize), T.ToTensor()])
train_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
val_dataset = FrameVideoDataset( root_dir=root_dir, split="val",transform=transform, stack_frames=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False )

#check hw
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#setup model, loss function, and optimizer
model = SingleFrameCNN(num_classes=10).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


#lets train
for epoch in range(epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        # predict and compute loss
        optimizer.zero_grad()
        logits = model(frames)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # track loss and accuracy
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / total:.4f} | Frame accuracy: {100 * correct / total:.1f}%" )
    # eval
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            # [B, 3, T, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
            batch_size, channels, num_frames, h, w = videos.shape

            frames = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, h, w)


            logits = model(frames)
            frame_probs = torch.softmax(logits, dim=1)

            # gather preds [B, T, 10]
            frame_probs = frame_probs.reshape(batch_size, num_frames, 10)

            # mean over frames to get video-level predictions
            video_probs = frame_probs.mean(dim=1)
            predictions = video_probs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += batch_size

    val_accuracy = val_correct / val_total
    print(f"Validation video accuracy: {100 * val_accuracy:.1f}%, best: {100 * best_val_accuracy:.1f}%")

    history.append({
        "epoch": epoch + 1,
        "loss": total_loss / total,
        "train_frame_accuracy": correct / total,
        "val_video_accuracy": val_accuracy,
    })

    pd.DataFrame(history).to_csv(run_dir / "stats.csv", index=False)

    # early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save({ "model_state_dict": model.state_dict(),"resize": resize,"epoch": epoch + 1,"val_accuracy": val_accuracy,"batch_size": batch_size},run_dir / "best_model.pt" )

        print(f"Saved best model: {100 * val_accuracy:.1f}%")

    else:
        epochs_without_improvement += 1
        print(f"No improvement: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

#save plots
stats = pd.DataFrame(history)
stats.plot(x="epoch", subplots=True, figsize=(7, 8))

plt.tight_layout()
plt.savefig(run_dir / "plots.png")
plt.close()