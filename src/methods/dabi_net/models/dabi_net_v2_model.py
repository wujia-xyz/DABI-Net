"""
DABI-Net: Depth-Aware Bidirectional Interaction Network

A novel approach for ultrasound image classification that combines:
1. Simple depth encoding (learnable + sinusoidal, no physics assumptions)
2. Depth-aware attention mechanism (attention weights modulated by depth)
3. True bidirectional interaction (Top-Down and Bottom-Up with cross-attention)

Key Innovation: True Bidirectional Processing
- Top-Down (TD): processes depth 0→15 (shallow to deep)
- Bottom-Up (BU): processes depth 15→0 (deep to shallow) via sequence flip
- Cross-attention enables information exchange between TD and BU
- Fusion aligns BU back to original order before concatenation

Architecture:
    Image → DINOv2 → [16, 16, 768] → Row Pooling → [16, 768]
          → Depth Encoding → True Bidirectional Interaction
          → Fusion (with alignment) → Classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import math


class SimpleDepthEncoding(nn.Module):
    """
    Simple depth encoding without physics assumptions.
    Uses learnable embeddings + sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int = 64, num_depths: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_depths = num_depths

        # Learnable depth embeddings
        self.depth_embed = nn.Embedding(num_depths, embed_dim // 2)

        # Sinusoidal encoding frequencies (fixed)
        self.register_buffer(
            'freq_bands',
            torch.linspace(1.0, num_depths / 2, embed_dim // 4)
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim // 2 + embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, depth_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_indices: [N] integer depth indices (0 to num_depths-1)
        Returns:
            depth_encoding: [N, embed_dim]
        """
        # Normalized depth (0 to 1)
        depth_normalized = depth_indices.float() / (self.num_depths - 1)

        # Learnable embedding
        learned = self.depth_embed(depth_indices)  # [N, embed_dim//2]

        # Sinusoidal encoding
        depth_expanded = depth_normalized.unsqueeze(-1)  # [N, 1]
        sin_features = torch.sin(depth_expanded * self.freq_bands * math.pi)
        cos_features = torch.cos(depth_expanded * self.freq_bands * math.pi)
        sinusoidal = torch.cat([sin_features, cos_features], dim=-1)

        # Combine
        combined = torch.cat([learned, sinusoidal], dim=-1)
        encoding = self.out_proj(combined)

        return encoding


class AttentionRowPooling(nn.Module):
    """
    Attention-based pooling for aggregating patch features within a row.
    """

    def __init__(self, feature_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim

        # Query for attention (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Key and Value projections
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = feature_dim ** -0.5

    def forward(self, row_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            row_features: [B, num_rows, patches_per_row, feature_dim]
        Returns:
            pooled: [B, num_rows, feature_dim]
        """
        B, num_rows, patches_per_row, D = row_features.shape

        # Reshape for attention: [B * num_rows, patches_per_row, D]
        x = row_features.reshape(B * num_rows, patches_per_row, D)

        # Expand query for batch
        query = self.query.expand(B * num_rows, -1, -1)

        # Compute key and value
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Attention scores
        attn = torch.bmm(query, key.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.bmm(attn, value)
        out = self.out_proj(out.squeeze(1))

        return out.reshape(B, num_rows, D)


class DepthAwareAttention(nn.Module):
    """
    Depth-Aware Attention mechanism.

    Attention weights = feature similarity + depth bias
    Depth bias is learned by MLP, not physics formula.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_depths: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Depth bias network: learns relationship between depth pairs
        # Input: [depth_i, depth_j, |depth_i - depth_j|] normalized
        self.depth_bias_net = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads)
        )

        # Pre-compute depth pair features for efficiency
        self.num_depths = num_depths
        self._precompute_depth_pairs()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _precompute_depth_pairs(self):
        """Pre-compute depth pair features [i, j, |i-j|] for all pairs."""
        num_depths = self.num_depths
        # Create all pairs
        i_idx = torch.arange(num_depths).unsqueeze(1).expand(-1, num_depths)
        j_idx = torch.arange(num_depths).unsqueeze(0).expand(num_depths, -1)

        # Normalize to [0, 1]
        i_norm = i_idx.float() / (num_depths - 1)
        j_norm = j_idx.float() / (num_depths - 1)
        diff_norm = (i_idx - j_idx).abs().float() / (num_depths - 1)

        # Stack: [num_depths, num_depths, 3]
        depth_pairs = torch.stack([i_norm, j_norm, diff_norm], dim=-1)
        self.register_buffer('depth_pairs', depth_pairs)

    def forward(self, x: torch.Tensor, depth_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N = num_depths
            depth_indices: not used (for API compatibility)
        Returns:
            output: [B, N, D]
        """
        B, N, D = x.shape

        # Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores from features
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add depth bias
        depth_bias = self.depth_bias_net(self.depth_pairs)  # [N, N, H]
        depth_bias = depth_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, N, N]
        attn = attn + depth_bias

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, H, N, D_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(x + out)

        return out


class BidirectionalInteractionLayer(nn.Module):
    """
    Bidirectional Interaction Layer.

    Top-Down and Bottom-Up paths with cross-attention interaction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_depths: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        # Depth-aware attention for each direction
        self.td_attn = DepthAwareAttention(dim, num_heads, num_depths, dropout)
        self.bu_attn = DepthAwareAttention(dim, num_heads, num_depths, dropout)

        # Cross-attention for interaction
        self.cross_attn_td = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_bu = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Layer norms for cross-attention
        self.norm_td = nn.LayerNorm(dim)
        self.norm_bu = nn.LayerNorm(dim)

        # FFN for each direction
        self.ffn_td = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_bu = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn_td = nn.LayerNorm(dim)
        self.norm_ffn_bu = nn.LayerNorm(dim)

    def forward(self, h_td: torch.Tensor, h_bu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_td: Top-Down features [B, N, D]
            h_bu: Bottom-Up features [B, N, D]
        Returns:
            h_td_out: Updated Top-Down features [B, N, D]
            h_bu_out: Updated Bottom-Up features [B, N, D]
        """
        # Depth-aware self-attention
        h_td = self.td_attn(h_td)
        h_bu = self.bu_attn(h_bu)

        # Cross-attention interaction
        # TD attends to BU
        h_td_cross, _ = self.cross_attn_td(h_td, h_bu, h_bu)
        h_td = self.norm_td(h_td + h_td_cross)

        # BU attends to TD
        h_bu_cross, _ = self.cross_attn_bu(h_bu, h_td, h_td)
        h_bu = self.norm_bu(h_bu + h_bu_cross)

        # FFN
        h_td = self.norm_ffn_td(h_td + self.ffn_td(h_td))
        h_bu = self.norm_ffn_bu(h_bu + self.ffn_bu(h_bu))

        return h_td, h_bu


class DINOv2DABINetV2Model(nn.Module):
    """
    DABI-Net v2: Depth-Aware Bidirectional Interaction Network (True Bidirectional).

    Key Fix from v1:
    - v1 Problem: TD and BU both initialized with same x, no true directionality
    - v2 Solution: TD processes depth 0→15, BU processes depth 15→0 (flipped)

    Architecture:
        Image → DINOv2 → [16, 16, 768] → Row Pooling → [16, 768]
              → Depth Encoding → TRUE Bidirectional Interaction
              → Fusion (with alignment) → Classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        dinov2_model: str = 'dinov2_vitb14',
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        load_method: str = 'hf',
        pooling: str = 'mean',
        row_pooling: str = 'attention',
        depth_embed_dim: int = 64,
        simple_classifier: bool = True,
        bidirectional_mode: str = 'true_bidirectional'  # 'single', 'same_direction', 'true_bidirectional'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        self.grid_size = 16
        self.row_pooling_type = row_pooling
        self.bidirectional_mode = bidirectional_mode

        # Load DINOv2
        if load_method == 'hf':
            from transformers import AutoModel
            model_map = {
                'dinov2_vitb14': 'facebook/dinov2-base',
                'dinov2_vits14': 'facebook/dinov2-small',
                'dinov2_vitl14': 'facebook/dinov2-large',
            }
            self.dinov2 = AutoModel.from_pretrained(
                model_map.get(dinov2_model, dinov2_model)
            )
            self.dino_dim = self.dinov2.config.hidden_size
        else:
            import os
            local_dir = os.path.expanduser(
                '~/.cache/torch/hub/facebookresearch_dinov2_main'
            )
            self.dinov2 = torch.hub.load(
                local_dir, dinov2_model, source='local', pretrained=True
            )
            self.dino_dim = self.dinov2.embed_dim

        # Freeze backbone
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # Row pooling module
        if row_pooling == 'attention':
            self.row_pooling = AttentionRowPooling(
                feature_dim=self.dino_dim,
                num_heads=num_heads
            )
        else:
            self.row_pooling = None

        # Depth encoding
        self.depth_encoding = SimpleDepthEncoding(
            embed_dim=depth_embed_dim,
            num_depths=16
        )

        # Input projection (dino_dim + depth_embed_dim -> hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.dino_dim + depth_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Bidirectional Interaction Layers
        self.interaction_layers = nn.ModuleList([
            BidirectionalInteractionLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                num_depths=16,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Fusion: concat TD and BU, then project
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Classifier
        if simple_classifier:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

        # Global pooling type
        self.pooling = pooling

        print(f"DINOv2DABINetV2Model initialized:")
        print(f"  - DINOv2: {dinov2_model} (dim={self.dino_dim})")
        print(f"  - Nodes: 16 rows (depth layers)")
        print(f"  - Row pooling: {row_pooling}")
        print(f"  - Depth encoding: learnable + sinusoidal (dim={depth_embed_dim})")
        print(f"  - Bidirectional Interaction: {num_layers} layers, hidden={hidden_dim}")
        print(f"  - Bidirectional mode: {bidirectional_mode}")
        print(f"  - Classifier: {'simple' if simple_classifier else 'MLP'}")

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features."""
        if self.freeze_backbone:
            with torch.no_grad():
                if hasattr(self.dinov2, 'forward_features'):
                    features = self.dinov2.forward_features(images)
                    patch_features = features['x_norm_patchtokens']
                else:
                    outputs = self.dinov2(images, return_dict=True)
                    patch_features = outputs.last_hidden_state[:, 1:, :]
        else:
            if hasattr(self.dinov2, 'forward_features'):
                features = self.dinov2.forward_features(images)
                patch_features = features['x_norm_patchtokens']
            else:
                outputs = self.dinov2(images, return_dict=True)
                patch_features = outputs.last_hidden_state[:, 1:, :]

        return patch_features

    def aggregate_rows(self, patch_features: torch.Tensor) -> torch.Tensor:
        """Aggregate patch features by row."""
        B, N, D = patch_features.shape
        patch_grid = patch_features.view(B, self.grid_size, self.grid_size, D)

        if self.row_pooling_type == 'attention' and self.row_pooling is not None:
            row_features = self.row_pooling(patch_grid)
        elif self.row_pooling_type == 'max':
            row_features = patch_grid.max(dim=2)[0]
        else:
            row_features = patch_grid.mean(dim=2)

        return row_features

    def forward(
        self,
        images: torch.Tensor,
        images_np: np.ndarray = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with TRUE bidirectional processing.

        Key Fix from v1:
        - TD processes depth 0→15 (shallow to deep)
        - BU processes depth 15→0 (deep to shallow) via flip
        - Before fusion, BU is flipped back to align with TD

        Args:
            images: [B, 3, H, W] normalized images
            images_np: Not used (for API compatibility)

        Returns:
            logits: [B, num_classes]
        """
        B = images.shape[0]
        device = images.device

        # Stage 1: Extract patch features [B, 256, dino_dim]
        patch_features = self.extract_patch_features(images)

        # Stage 2: Aggregate by row [B, 16, dino_dim]
        row_features = self.aggregate_rows(patch_features)

        # Stage 3: Add depth encoding
        depth_indices = torch.arange(self.grid_size, device=device)
        depth_enc = self.depth_encoding(depth_indices)  # [16, depth_embed_dim]
        depth_enc = depth_enc.unsqueeze(0).expand(B, -1, -1)  # [B, 16, depth_embed_dim]

        # Concatenate row features with depth encoding
        x = torch.cat([row_features, depth_enc], dim=-1)  # [B, 16, dino_dim + depth_embed_dim]

        # Project to hidden dim
        x = self.input_proj(x)  # [B, 16, hidden_dim]

        # Stage 4: Bidirectional Interaction (mode-dependent)
        if self.bidirectional_mode == 'single':
            # Single branch: only TD, no cross-attention
            h_td = x
            for layer in self.interaction_layers:
                # Only use TD's self-attention and FFN, skip cross-attention
                h_td = layer.td_attn(h_td)
                h_td = layer.norm_ffn_td(h_td + layer.ffn_td(h_td))
            fused = h_td  # No fusion needed for single branch
        elif self.bidirectional_mode == 'same_direction':
            # Same-direction dual branch: TD + TD (no flip)
            h_td = x
            h_bu = x  # Same direction, no flip
            for layer in self.interaction_layers:
                h_td, h_bu = layer(h_td, h_bu)
            # No need to flip BU back since it was never flipped
            fused = torch.cat([h_td, h_bu], dim=-1)  # [B, 16, hidden_dim * 2]
            fused = self.fusion(fused)  # [B, 16, hidden_dim]
        else:  # 'true_bidirectional' (default)
            # True bidirectional: TD (0→15) + BU (15→0)
            h_td = x                    # [B, 16, hidden_dim] - original order
            h_bu = x.flip(dims=[1])     # [B, 16, hidden_dim] - reversed order
            for layer in self.interaction_layers:
                h_td, h_bu = layer(h_td, h_bu)
            # Stage 5: Fusion - Align BU back to original order before fusion
            h_bu_aligned = h_bu.flip(dims=[1])  # Flip back to match TD's order
            fused = torch.cat([h_td, h_bu_aligned], dim=-1)  # [B, 16, hidden_dim * 2]
            fused = self.fusion(fused)  # [B, 16, hidden_dim]

        # Stage 6: Global pooling
        if self.pooling == 'sum':
            graph_features = fused.sum(dim=1)
        else:
            graph_features = fused.mean(dim=1)

        # Classify
        logits = self.classifier(graph_features)

        return logits


if __name__ == '__main__':
    print("Testing DINOv2DABINetV2Model (TRUE Bidirectional)...")

    model = DINOv2DABINetV2Model(
        num_classes=2,
        dinov2_model='dinov2_vitb14',
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        load_method='hf',
        row_pooling='attention'
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward
    model = model.cuda()
    dummy_input = torch.randn(2, 3, 224, 224).cuda()

    with torch.no_grad():
        logits = model(dummy_input)
        print(f"Output shape: {logits.shape}")  # Should be [2, 2]
