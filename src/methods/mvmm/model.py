"""
MVMM: Multimodal Multi-view CNN for breast cancer diagnosis.

Based on: "Prospective assessment of breast cancer risk from multimodal multiview
ultrasound images via clinically applicable deep learning"
Qian et al., Nature Biomedical Engineering, 2021

Architecture:
- Multi-pathway network with ResNet-18 + SENet backbone
- Each modality processed in separate pathway
- Features concatenated and passed through FC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ModalityPathway(nn.Module):
    """Single modality pathway with ResNet-18 + SENet."""

    def __init__(self, pretrained=True):
        super(ModalityPathway, self).__init__()

        # ResNet-18 backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the final FC layer
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # SE block after ResNet features
        self.se_block = SEBlock(512, reduction=16)

        # Additional conv layers as described in paper
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class MVMM(nn.Module):
    """
    Multimodal Multi-view Model for breast cancer classification.

    Supports 2 or 3 modalities:
    - Bimodal: B-mode + Doppler
    - Multimodal: B-mode + Doppler + Elastography
    """

    def __init__(self, num_classes=2, num_modalities=3, pretrained=True, dropout=0.5):
        super(MVMM, self).__init__()

        self.num_modalities = num_modalities

        # Create pathway for each modality
        self.pathways = nn.ModuleList([
            ModalityPathway(pretrained=pretrained)
            for _ in range(num_modalities)
        ])

        # Feature dimension after each pathway
        pathway_dim = 128
        fused_dim = pathway_dim * num_modalities

        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, num_modalities*C, H, W)
               where each modality has C=3 channels (RGB)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        # Split input into modalities
        # Assuming input is concatenated along channel dimension
        # Each modality has 3 channels
        batch_size = x.size(0)
        channels_per_modality = 3

        # Extract features from each modality pathway
        features = []
        for i, pathway in enumerate(self.pathways):
            start_ch = i * channels_per_modality
            end_ch = (i + 1) * channels_per_modality
            modality_input = x[:, start_ch:end_ch, :, :]
            feat = pathway(modality_input)
            features.append(feat)

        # Concatenate features from all pathways
        fused_features = torch.cat(features, dim=1)

        # Classification
        logits = self.classifier(fused_features)

        return logits


def create_mvmm(num_classes=2, num_modalities=3, pretrained=True):
    """Factory function to create MVMM model."""
    return MVMM(
        num_classes=num_classes,
        num_modalities=num_modalities,
        pretrained=pretrained,
        dropout=0.5
    )


if __name__ == '__main__':
    # Test the model
    model = create_mvmm(num_classes=2, num_modalities=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    x = torch.randn(2, 9, 224, 224)  # 3 modalities * 3 channels = 9
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
