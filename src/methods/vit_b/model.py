"""
ViT-B baseline model for single-modal breast ultrasound classification.
Uses ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
from torchvision import models


class ViTBClassifier(nn.Module):
    """
    ViT-B/16 with pretrained ImageNet weights for binary classification.
    """

    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(ViTBClassifier, self).__init__()

        # Load pretrained ViT-B/16
        if pretrained:
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.vit_b_16(weights=None)

        # Get the number of features from the last layer
        num_features = self.backbone.heads.head.in_features

        # Replace the final head with custom classifier
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_vit_b(num_classes=2, pretrained=True, dropout=0.5):
    """Factory function to create ViT-B model."""
    return ViTBClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


if __name__ == '__main__':
    # Test the model
    model = create_vit_b(num_classes=2, pretrained=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
