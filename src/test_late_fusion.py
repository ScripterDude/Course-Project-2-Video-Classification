import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from datasets import FrameVideoDataset, root_dir
from models import LateFusionResNet50

BATCH_SIZE = 8
WEIGHTS = ResNet50_Weights.DEFAULT
# pass the checkpoint as an argument, e.g.
# python src/test_late_fusion.py experiments/LateFusionResNet50/original/final/<run>_seed0/best_model.pt
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else 'results/LateFusion/best_model.pt'


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_ds = FrameVideoDataset(root_dir, split='test', transform=WEIGHTS.transforms(), stack_frames=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    n_classes = test_ds.df['label'].nunique()
    model = LateFusionResNet50(num_classes=n_classes, weights=None).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)['model_state_dict'])
    model.eval()

    correct, total = 0, 0
    with torch.inference_mode():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            preds = model(clips).argmax(1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    print(f'test acc: {correct / total:.3f}  ({correct}/{total})')