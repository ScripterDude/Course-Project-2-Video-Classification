import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SingleFrameCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # rgb to 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 to 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64 to 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #avg pool to single feature map
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames):
        features = self.features(frames)
        return self.classifier(features)

class ResNet50Stream(nn.Module):
    def __init__(self, num_classes=10, in_channels=3,weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        self.backbone = resnet50(weights=weights)

        #for the flow model
        if in_channels != 3:
            rgb_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, rgb_conv.out_channels,
                kernel_size=rgb_conv.kernel_size,
                stride=rgb_conv.stride,
                padding=rgb_conv.padding,
                bias=False,
            )

            #init flow from the RGB channel mean
            with torch.no_grad():
                mean_weights = rgb_conv.weight.mean(dim=1, keepdim=True)
                self.backbone.conv1.weight.copy_(mean_weights.repeat(1, in_channels, 1, 1))

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, inputs):
        return self.backbone(inputs)


class DualStreamResNet50(nn.Module):

    def __init__(self, num_classes=10, flow_channels=18, rgb_weight=0.5,weights=ResNet50_Weights.DEFAULT):
        super().__init__()

        self.rgb_weight = rgb_weight
        self.rgb_stream = ResNet50Stream(num_classes=num_classes, in_channels=3, weights=weights)
        self.flow_stream = ResNet50Stream(num_classes=num_classes, in_channels=flow_channels, weights=weights)

    def forward(self, rgb_videos, flows):
        # videos = [B, 3, T, H, W], flows= [B, 18, H, W]
        batch_size, channels, num_frames, height, width = rgb_videos.shape
        frames = rgb_videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        rgb_logits = self.rgb_stream(frames)
        rgb_probs = rgb_logits.softmax(dim=1).reshape(batch_size, num_frames, -1).mean(dim=1)
        flow_probs = self.flow_stream(flows).softmax(dim=1)

        #wieght the two streams
        return (self.rgb_weight * rgb_probs+ (1.0 - self.rgb_weight) * flow_probs)


class LateFusionResNet50(nn.Module):

    def __init__(self, num_classes=10, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        self.backbone = resnet50(weights=weights)
        feature_dim = self.backbone.fc.in_features  #2048
        self.backbone.fc = nn.Identity()              
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, videos):
        #videos = [B, 3, T, H, W]
        batch_size, channels, num_frames, height, width = videos.shape
        frames = videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        features = self.backbone(frames).reshape(batch_size, num_frames, -1)
        #late fusion over frames
        fused = features.mean(dim=1)                    
        return self.classifier(fused)



