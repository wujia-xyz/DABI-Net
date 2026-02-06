import torch
import torchvision
import torch.nn.functional as F

from torch import nn


class SharedBackBone(nn.Module):
    def __init__(self, pretrain=True, depth=-3):
        super().__init__()

        # Establishment of a backbone network with pretrained weights
        if pretrain:
            self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            print('Pre-training weight loading complete')
        else:
            self.backbone = torchvision.models.resnet50(weights=None)

        # Remove backbone's layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:depth])

    def forward(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)

        return x1, x2, x3


class SharedBackBoneFPN(nn.Module):
    def __init__(self, pretrain=True, depth=None):
        super().__init__()

        # Establishment of a backbone network with pretrained weights
        if pretrain:
            self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            print('Pre-training weight loading complete')
        else:
            self.backbone = torchvision.models.resnet50(weights=None)

        # Load pre-training weights
        if depth is None or not isinstance(depth, (list, tuple)):
            depth = [-5, -4, -3]

        # Remove backbone's layers
        self.backbone_1 = nn.Sequential(*list(self.backbone.children())[:depth[0]])
        self.backbone_2 = nn.Sequential(*list(self.backbone.children())[:depth[1]])
        self.backbone_3 = nn.Sequential(*list(self.backbone.children())[:depth[2]])

        # Use convolutional layers to halve the same size of the feature map
        self.conv_d = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv_s1 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv_s2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv_s3 = nn.Conv2d(1024, 512, kernel_size=(1, 1))

    def forward(self, x1, x2, x3):
        x1_1 = self.backbone_1(x1)
        x1_2 = self.backbone_1(x2)
        x1_3 = self.backbone_1(x3)

        x1_1 = self.conv_d(x1_1)
        x1_2 = self.conv_d(x1_2)
        x1_3 = self.conv_d(x1_3)

        x2_1 = self.backbone_2(x1)
        x2_2 = self.backbone_2(x2)
        x2_3 = self.backbone_2(x3)

        x3_1 = self.backbone_3(x1)
        x3_2 = self.backbone_3(x2)
        x3_3 = self.backbone_3(x3)

        x3_1 = F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=False)
        x3_2 = F.interpolate(x3_2, scale_factor=2, mode='bilinear', align_corners=False)
        x3_3 = F.interpolate(x3_3, scale_factor=2, mode='bilinear', align_corners=False)

        x1 = torch.cat((x1_1, x2_1), dim=1) + x3_1
        x2 = torch.cat((x1_2, x2_2), dim=1) + x3_2
        x3 = torch.cat((x1_3, x2_3), dim=1) + x3_3

        x1 = self.conv_s1(x1)
        x2 = self.conv_s2(x2)
        x3 = self.conv_s3(x3)

        return x1, x2, x3


class SharedTransformer(nn.Module):
    def __init__(self, pretrain=True, out_channels=512, out_size=28):
        super().__init__()

        # Establishment of a backbone network with pretrained weights
        if pretrain:
            self.vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
            print('ViT Pre-training weight loading complete')
        else:
            self.vit = torchvision.models.vit_b_16(weights=None)

        self.hidden_dim = 768
        self.patch_size = 14  # 224 / 16 = 14

        # Adapt to match ResNet FPN output format: [B, 512, 28, 28]
        self.adapt = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1),  # 768 -> 512
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False)  # 14x14 -> 28x28
        )

    def _extract_features(self, x):
        """Extract patch features from ViT, output [B, 768, 14, 14]"""
        B = x.shape[0]

        # Patch embedding
        x = self.vit.conv_proj(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]

        # Add class token
        batch_class_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)  # [B, 197, 768]

        # Encoder (includes positional embedding)
        x = self.vit.encoder(x)  # [B, 197, 768]

        # Remove class token and reshape to 2D
        patch_tokens = x[:, 1:, :]  # [B, 196, 768]
        feat_2d = patch_tokens.transpose(1, 2).reshape(B, self.hidden_dim, self.patch_size, self.patch_size)

        return feat_2d  # [B, 768, 14, 14]

    def forward(self, x1, x2, x3):
        x1 = self.adapt(self._extract_features(x1))
        x2 = self.adapt(self._extract_features(x2))
        x3 = self.adapt(self._extract_features(x3))
        return x1, x2, x3


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A1 = torch.rand([32, 3, 224, 224])
    A2 = torch.rand([32, 3, 224, 224])
    A3 = torch.rand([32, 3, 224, 224])

    shared = SharedBackBone(pretrain=True, depth=-4)
    C1, C2, C3 = shared(A1, A2, A3)

    print(C1.shape)
    print(C2.shape)
    print(C3.shape)
