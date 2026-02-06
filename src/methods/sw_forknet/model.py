"""
SW-ForkNet: Swin Transformer-based Fork Network for Breast Tumor Classification

Reference:
Üzen, H., Firat, H., Atila, O., & Şengür, A. (2024).
Swin transformer-based fork architecture for automated breast tumor classification.
Expert Systems with Applications, 256, 125009.

Architecture:
- Backbone: DenseNet121
- Three feature branches:
  1. Semantic features: DenseNet121 final layer → GAP → FC(256)
  2. Spatial features: DenseNet121 L3 → sSE block → GAP → 256-dim
  3. Long-context features: DenseNet121 L3 → Swin Transformer → GAP → 256-dim
- Feature fusion: Element-wise Add
- Classifier: FC + Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class sSEBlock(nn.Module):
    """Spatial Squeeze-and-Excitation Block"""
    def __init__(self, in_channels):
        super().__init__()
        # Pointwise convolution to get spatial attention weights
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        # Get spatial attention weights
        w = self.conv1(x)  # (B, 128, H, W)
        w = F.relu(w)
        w = self.conv2(w)  # (B, 1, H, W)
        w = torch.sigmoid(w)  # Spatial attention weights

        # Apply spatial attention
        out = x * w  # (B, C, H, W)
        return out


class PatchEmbed(nn.Module):
    """Patch Embedding for Swin Transformer"""
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Create relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """Partition into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class SwinTransformerModule(nn.Module):
    """Simplified Swin Transformer for feature extraction"""
    def __init__(self, in_channels, embed_dim=256, num_heads=8, window_size=4, depth=2):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size=4)

        self.layers = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.layers.append(
                SwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size
                )
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x, H, W = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, H, W)

        x = self.norm(x)
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        return x


class SWForkNet(nn.Module):
    """
    SW-ForkNet: Swin Transformer-based Fork Network

    Three branches:
    1. Semantic features from DenseNet121 final layer
    2. Spatial features from L3 with sSE block
    3. Long-context features from L3 with Swin Transformer
    """
    def __init__(self, num_classes=2, feature_dim=256, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim

        # Load pretrained DenseNet121
        densenet = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)

        # Extract DenseNet121 components
        # Initial layers: conv0, norm0, relu0, pool0
        self.initial = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0
        )

        # Dense blocks and transitions
        self.denseblock1 = densenet.features.denseblock1
        self.transition1 = densenet.features.transition1
        self.denseblock2 = densenet.features.denseblock2
        self.transition2 = densenet.features.transition2

        # L3 output channels (after denseblock2, before transition2): 512
        # Note: DenseNet121 structure:
        # - After denseblock2: (B, 512, 28, 28)
        # - After transition2: (B, 256, 14, 14)
        # We use features after denseblock2 (before transition2) as L3
        self.l3_channels = 512

        # Continue to final layers for semantic features
        self.denseblock3 = densenet.features.denseblock3
        self.transition3 = densenet.features.transition3
        self.denseblock4 = densenet.features.denseblock4
        self.final_norm = densenet.features.norm5

        # Final channels: 1024
        self.final_channels = 1024

        # Branch 1: Semantic features
        self.semantic_fc = nn.Linear(self.final_channels, feature_dim)

        # Branch 2: Spatial features with sSE
        self.sse_block = sSEBlock(self.l3_channels)
        self.spatial_fc = nn.Linear(self.l3_channels, feature_dim)

        # Branch 3: Long-context features with Swin Transformer
        self.swin_transformer = SwinTransformerModule(
            in_channels=self.l3_channels,
            embed_dim=feature_dim,
            num_heads=8,
            window_size=4,
            depth=2
        )

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Initial layers
        x = self.initial(x)

        # Dense blocks 1-2
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)

        # L3 features (after denseblock2, before transition2)
        l3_features = x  # (B, 512, 28, 28) for 224x224 input

        x = self.transition2(x)

        # Continue to final layers
        x = self.denseblock3(x)
        x = self.transition3(x)
        x = self.denseblock4(x)
        x = self.final_norm(x)
        x = F.relu(x)

        # Branch 1: Semantic features
        semantic = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # (B, 1024)
        semantic = self.semantic_fc(semantic)  # (B, 256)

        # Branch 2: Spatial features with sSE
        spatial = self.sse_block(l3_features)  # (B, 512, 28, 28)
        spatial = F.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1)  # (B, 512)
        spatial = self.spatial_fc(spatial)  # (B, 256)

        # Branch 3: Long-context features with Swin Transformer
        long_context = self.swin_transformer(l3_features)  # (B, 256)

        # Feature fusion: Element-wise Add
        fused = semantic + spatial + long_context  # (B, 256)

        # Classification
        out = self.classifier(fused)

        return out


def create_sw_forknet(num_classes=2, pretrained=True):
    """Create SW-ForkNet model"""
    return SWForkNet(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # Test model
    model = create_sw_forknet(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
