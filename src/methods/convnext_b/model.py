"""
ConvNeXt-B baseline model for single-modal breast ultrasound classification.
Uses ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models


class ConvNeXtBClassifier(nn.Module):
    """
    ConvNeXt-B with pretrained ImageNet weights for binary classification.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(ConvNeXtBClassifier, self).__init__()

        # Load pretrained ConvNeXt-B
        if pretrained:
            self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.convnext_base(weights=None)

        # Get the number of features from the classifier
        num_features = self.backbone.classifier[2].in_features

        # Replace the final classifier with custom classifier
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_convnext_b(num_classes=2, pretrained=True, dropout=0.5):
    """Factory function to create ConvNeXt-B model."""
    return ConvNeXtBClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test the model
    model = create_convnext_b(num_classes=2, pretrained=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
