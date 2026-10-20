from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch
from datasets import FlowVideoDataset, FrameImageDataset, FrameVideoDataset, root_dir
from models import DualStreamResNet50
from torchvision.models import ResNet50_Weights
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# set hyperparams
seed = 0
torch.manual_seed(seed)
epochs = 100
batch_size = 32
learning_rate = 0.0001
rgb_learning_rate = learning_rate * 0.1
flow_learning_rate = learning_rate * 0.9
weight_decay = 0.0001
resize = (224, 224)
patience = 10
best_val_accuracy = -1.0
epochs_without_improvement = 0
flow_channels = 18
flow_bound = 32.0 #default from https://github.com/open-mmlab/denseflow
weights = ResNet50_Weights.DEFAULT
rgb_weight = 0.35

#log experiments
# root_dir = "../Data/ucf101_noleakage"
model_name = "DualStreamResNet50"
experiment_name = (f"seed{seed}_epochs{epochs}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_{resize[0]}x{resize[1]}_pat{patience}_time{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

experiment_dir = Path("experiments") / model_name / experiment_name
run_dir = experiment_dir
run_dir.mkdir(parents=True, exist_ok=True)

#save conf:
model_config = {
    "model_name": model_name,
    "num_classes": 10,
    "flow_channels": flow_channels,
    "flow_bound": flow_bound,
    "normalization": "flows.clamp(-flow_bound, flow_bound) / flow_bound",
    "weights": weights.name,
    "rgb_weight": rgb_weight,
    "seed": seed,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "resize": resize,
    "patience": patience,
    "data_dir": root_dir,
    'rgb_learning_rate': rgb_learning_rate,
    'flow_learning_rate': flow_learning_rate,
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,
    'scheduler_patience': 2,
    'scheduler_min_lr': 1e-6,
}

(run_dir / "config.json").write_text(json.dumps(model_config))
history = []
print("Saving run to:", run_dir)

#prep data
rgb_transform = weights.transforms()

rgb_train_dataset = FrameImageDataset(root_dir=root_dir, split='train', transform=rgb_transform)
# rgb_train_dataset = FrameImageDataset(root_dir=root_dir, split='train')
flow_train_dataset = FlowVideoDataset(root_dir=root_dir, split='train')

rgb_val_dataset = FrameVideoDataset(root_dir=root_dir,split='val',transform=rgb_transform,stack_frames=True)
# rgb_val_dataset = FrameVideoDataset(root_dir=root_dir,split='val',stack_frames=True)
flow_val_dataset = FlowVideoDataset(root_dir=root_dir, split='val')

# if rgb_val_dataset.video_paths != flow_val_dataset.video_paths:
#     raise ValueError('RGB and flow validation video order does not match.')

rgb_train_loader = DataLoader(rgb_train_dataset, batch_size=batch_size, shuffle=True)
flow_train_loader = DataLoader(flow_train_dataset, batch_size=batch_size, shuffle=True)
rgb_val_loader = DataLoader(rgb_val_dataset, batch_size=batch_size, shuffle=False)
flow_val_loader = DataLoader(flow_val_dataset, batch_size=batch_size, shuffle=False)

#check hw
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#setup model, loss function, and optimizer

model = DualStreamResNet50(num_classes=10,flow_channels=flow_channels,rgb_weight=rgb_weight,weights=weights).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
rgb_optimizer = torch.optim.Adam(model.rgb_stream.parameters(), lr=rgb_learning_rate, weight_decay=weight_decay)
flow_optimizer = torch.optim.Adam(model.flow_stream.parameters(), lr=flow_learning_rate, weight_decay=weight_decay)
#use scheduler to reduce overfitting
rgb_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rgb_optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(flow_optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)

#cap the flows
def cap_flows(flows):
    return flows.clamp(-flow_bound, flow_bound) / flow_bound

#lets train
def train_stream(stream, loader, optimizer, is_flow=False):
    stream.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_flow:
            inputs = cap_flows(inputs)

        optimizer.zero_grad()
        logits = stream(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += current_batch_size

    return total_loss / total, correct / total


for epoch in range(epochs):
    model.train()

    # train echa stream
    rgb_loss, rgb_accuracy = train_stream(model.rgb_stream, rgb_train_loader, rgb_optimizer)
    flow_loss, flow_accuracy = train_stream(model.flow_stream,flow_train_loader,flow_optimizer,is_flow=True)

    #eval on fused
    model.eval()
    val_correct = 0
    val_total = 0
    rgb_val_correct = 0
    flow_val_correct = 0

    model.eval()

    with torch.no_grad():
        for rgb_batch, flow_batch in zip(rgb_val_loader, flow_val_loader):
            videos, labels = rgb_batch
            flows, flow_labels = flow_batch

            # if not torch.equal(labels, flow_labels):
            #     raise ValueError('RGB and flow validation labels do not match.')

            videos = videos.to(device)
            flows = cap_flows(flows.to(device))
            labels = labels.to(device)

            b, c, t, h, w = videos.shape
            frames = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

            rgb_probs = model.rgb_stream(frames).softmax(dim=1).reshape(b, t, -1).mean(dim=1)
            flow_probs = model.flow_stream(flows).softmax(dim=1)
            probabilities = model.rgb_weight * rgb_probs + (1.0 - model.rgb_weight) * flow_probs

            rgb_val_correct += (rgb_probs.argmax(dim=1) == labels).sum().item()
            flow_val_correct += (flow_probs.argmax(dim=1) == labels).sum().item()
            val_correct += (probabilities.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    rgb_val_accuracy = rgb_val_correct / val_total
    flow_val_accuracy = flow_val_correct / val_total
    val_accuracy = val_correct / val_total
    print(100 * '-')
    print(f'Epoch {epoch + 1}/{epochs} | RGB loss: {rgb_loss:.4f} | RGB frame accuracy: {100 * rgb_accuracy:.1f}% | Flow loss: {flow_loss:.4f} | Flow video accuracy: {100 * flow_accuracy:.1f}%')
    print(f'RGB validation: {100 * rgb_val_accuracy:.1f}% | Flow validation: {100 * flow_val_accuracy:.1f}% | Fused validation: {100 * val_accuracy:.1f}%')

    rgb_scheduler.step(rgb_val_accuracy)
    flow_scheduler.step(flow_val_accuracy)

    print(f"Next epoch LR: RGB={rgb_optimizer.param_groups[0]['lr']:.2e} | Flow={flow_optimizer.param_groups[0]['lr']:.2e}")

    history.append({
        'epoch': epoch + 1,
        'rgb_loss': rgb_loss,
        'flow_loss': flow_loss,
        'rgb_train_frame_accuracy': rgb_accuracy,
        'flow_train_video_accuracy': flow_accuracy,
        'rgb_learning_rate': rgb_optimizer.param_groups[0]['lr'],
        'flow_learning_rate': flow_optimizer.param_groups[0]['lr'],
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_factor': 0.5,
        'scheduler_patience': 2,
        'scheduler_min_lr': 1e-6,
        'val_video_accuracy': val_accuracy,
        'rgb_val_video_accuracy': rgb_val_accuracy,
        'flow_val_video_accuracy': flow_val_accuracy,
    })
    pd.DataFrame(history).to_csv(run_dir / 'stats.csv', index=False)

    # early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'epoch': epoch + 1,
            'val_accuracy': val_accuracy,
            'resize': resize,
            'batch_size': batch_size,
            'flow_channels': flow_channels,
            'flow_bound': flow_bound,
            'rgb_weight': rgb_weight,
        }, run_dir / 'best_model.pt')

        print(f'Saved best model: {100 * best_val_accuracy:.1f}%')
    else:
        epochs_without_improvement += 1
        print(f'No improvement: {epochs_without_improvement}/{patience}')

        if epochs_without_improvement >= patience:
            print(f'Early stopping. Best validation accuracy: {100 * best_val_accuracy:.1f}%')
            break

#save plots
stats = pd.DataFrame(history)
stats.plot(x='epoch', subplots=True, figsize=(8, 12))
plt.tight_layout()
plt.savefig(run_dir / 'plots.png')
plt.close()
