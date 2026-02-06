"""
ResNet-18 baseline model for single-modal breast ultrasound classification.
Uses ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 with pretrained ImageNet weights for binary classification.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(ResNet18Classifier, self).__init__()

        # Load pretrained ResNet-18
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Replace the final FC layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_resnet18(num_classes=2, pretrained=True, dropout=0.5):
    """Factory function to create ResNet-18 model."""
    return ResNet18Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test the model
    model = create_resnet18(num_classes=2, pretrained=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
