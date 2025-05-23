import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MidLevelResNet50_LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MidLevelResNet50_LightCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # ---- Extract necessary layers ----
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # output: (B, 64, 56, 56)
        self.layer1 = resnet.layer1  # output: (B, 256, 56, 56)
        self.layer2 = resnet.layer2  # output: (B, 512, 28, 28)
        self.layer3 = resnet.layer3  # output: (B, 1024, 14, 14) → we'll upsample this

        # ---- Freeze ResNet ----
        for param in self.parameters():
            param.requires_grad = False

        # ---- CNN head (LightCNN style) ----
        self.classifier = nn.Sequential(
            nn.Conv2d(1536, 128, kernel_size=1, padding=1),  # Reduced filters
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # Reduced filters
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)  # Reduced input size
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)                  # (B, 512, 28, 28)
        x3 = self.layer3(x2)                  # (B, 1024, 14, 14)
        x3 = F.interpolate(x3, size=(28, 28), mode='bilinear', align_corners=False)  # Upsample to (B,1024,28,28)

        # Concatenate features: (B, 512+1024 = 1536, 28, 28)
        x_cat = torch.cat([x2, x3], dim=1)    # (B,1536,28,28)

        out = self.classifier(x_cat)
        return out