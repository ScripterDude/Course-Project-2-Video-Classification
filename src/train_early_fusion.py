import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset, root_dir
from models import EarlyFusionResNet50

BATCH_SIZE, EPOCHS, LR = 8, 10, 3e-5
WEIGHTS = ResNet50_Weights.DEFAULT
RESULTS_DIR = 'results/EarlyFusion'


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tf = WEIGHTS.transforms()
    train_ds = FrameVideoDataset(root_dir, 'train', tf, stack_frames=True)
    val_ds = FrameVideoDataset(root_dir, 'val', tf, stack_frames=True)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, BATCH_SIZE, num_workers=4)

    n_classes = train_ds.df['label'].nunique()
    model = EarlyFusionResNet50(train_ds.n_sampled_frames, n_classes, WEIGHTS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best = 0.0
    for ep in range(EPOCHS):
        tl, ta = run_epoch(model, train_dl, device, opt)
        vl, va = run_epoch(model, val_dl, device)
        print(f'epoch {ep+1:2d} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f}')
        if va > best:
            best = va
            torch.save(model.state_dict(), f'{RESULTS_DIR}/best_model.pt')
    print(f'best val acc: {best:.3f}')