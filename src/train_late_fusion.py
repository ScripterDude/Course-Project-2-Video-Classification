import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset, root_dir
from models import LateFusionResNet50

BATCH_SIZE = 8
EPOCHS = 10
LR = 0.0001 ##learning rate. lower than from-scratch since the backbone is pretrained
WEIGHTS = ResNet50_Weights.DEFAULT ##pretrained imagenet weights; .transforms() gives the matching resize+normalize
RESULTS_DIR = 'results/LateFusion'


#Training
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = nn.functional.cross_entropy(logits, labels)
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

    transform = WEIGHTS.transforms()
    train_ds = FrameVideoDataset(root_dir, split='train', transform=transform, stack_frames=True)
    val_ds = FrameVideoDataset(root_dir, split='val', transform=transform, stack_frames=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    n_classes = train_ds.df['label'].nunique()
    model = LateFusionResNet50(num_classes=n_classes, weights=WEIGHTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device)
        print(f'epoch {epoch+1:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} '
              f'| val loss {va_loss:.3f} acc {va_acc:.3f}')
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'model_state_dict': model.state_dict()},
                       os.path.join(RESULTS_DIR, 'best_model.pt'))

    print(f'best val acc: {best_acc:.3f}')