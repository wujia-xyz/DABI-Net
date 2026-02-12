"""
MsGoF: Multi-scale Gradational-Order Fusion Framework for Breast Lesions Classification.

Reference:
Ning, Z., Tu, C., Xiao, Q., Luo, J., & Zhang, Y. (2020).
Multi-scale Gradational-Order Fusion Framework for Breast Lesions Classification Using Ultrasound Images.
MICCAI 2020, pp. 171-180.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GradationalOrderModule(nn.Module):
    """
    Isotropous Gradational-Order Feature Module.
    Learns and combines different-order features to characterize complex textures.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First-order features (standard convolution)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second-order features (using two consecutive convolutions)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Third-order features (using three consecutive convolutions)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)

        # Concatenate different-order features
        fused = torch.cat([f1, f2, f3], dim=1)
        out = self.fusion(fused)

        return out


class FusionBlock(nn.Module):
    """
    Multi-scale Feature Fusion Block with Gradational-Order Module.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.go_module = GradationalOrderModule(in_channels, out_channels)

        # Channel attention for fusion
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.go_module(x)
        att = self.channel_attention(out)
        out = out * att
        return out


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale feature encoder using ResNet backbone.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet18(pretrained=pretrained)

        # Initial layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet blocks
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # Scale 1
        f2 = self.layer2(f1)  # Scale 2
        f3 = self.layer3(f2)  # Scale 3
        f4 = self.layer4(f3)  # Scale 4

        return f1, f2, f3, f4


class MsGoF(nn.Module):
    """
    Multi-scale Gradational-Order Fusion Framework.

    The framework extracts multi-scale features and fuses them using
    gradational-order feature modules to capture morphological characteristics
    of breast lesions.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(pretrained=pretrained)

        # Fusion blocks for each scale
        self.fusion1 = FusionBlock(64, 64)
        self.fusion2 = FusionBlock(128, 128)
        self.fusion3 = FusionBlock(256, 256)
        self.fusion4 = FusionBlock(512, 512)

        # Upsampling layers for multi-scale fusion
        self.up4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract multi-scale features
        f1, f2, f3, f4 = self.encoder(x)

        # Apply fusion blocks with gradational-order modules
        f1 = self.fusion1(f1)
        f2 = self.fusion2(f2)
        f3 = self.fusion3(f3)
        f4 = self.fusion4(f4)

        # Resize all features to the same spatial size (f1's size)
        size = f1.shape[2:]
        f2_up = F.interpolate(f2, size=size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=size, mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        fused = torch.cat([f1, f2_up, f3_up, f4_up], dim=1)

        # Final fusion and pooling
        out = self.final_fusion(fused)
        out = out.view(out.size(0), -1)

        # Classification
        out = self.classifier(out)

        return out


class MsGoFWithROI(nn.Module):
    """
    MsGoF with ROI-guided input.
    Takes both image and ROI mask, crops the lesion region at multiple scales.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        self.msgof = MsGoF(num_classes=num_classes, pretrained=pretrained)

        # Additional branch for mask-guided features
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Fusion of image and mask features
        self.final_fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, mask):
        # Get image features from MsGoF (without final classifier)
        f1, f2, f3, f4 = self.msgof.encoder(x)

        f1 = self.msgof.fusion1(f1)
        f2 = self.msgof.fusion2(f2)
        f3 = self.msgof.fusion3(f3)
        f4 = self.msgof.fusion4(f4)

        size = f1.shape[2:]
        f2_up = F.interpolate(f2, size=size, mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=size, mode='bilinear', align_corners=False)
        f4_up = F.interpolate(f4, size=size, mode='bilinear', align_corners=False)

        fused = torch.cat([f1, f2_up, f3_up, f4_up], dim=1)
        img_feat = self.msgof.final_fusion(fused)
        img_feat = img_feat.view(img_feat.size(0), -1)

        # Get mask features
        mask_feat = self.mask_encoder(mask)
        mask_feat = mask_feat.view(mask_feat.size(0), -1)

        # Combine features
        combined = torch.cat([img_feat, mask_feat], dim=1)
        out = self.final_fc(combined)

        return out


def create_msgof(num_classes=2, pretrained=True, use_roi=False):
    """Create MsGoF model."""
    if use_roi:
        return MsGoFWithROI(num_classes=num_classes, pretrained=pretrained)
    else:
        return MsGoF(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # Test the model
    model = MsGoF(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test with ROI
    model_roi = MsGoFWithROI(num_classes=2)
    mask = torch.randn(2, 1, 224, 224)
    out_roi = model_roi(x, mask)
    print(f"Output with ROI shape: {out_roi.shape}")
