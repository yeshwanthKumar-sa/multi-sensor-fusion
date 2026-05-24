"""
Multi-Sensor Fusion Model for Autonomous Vehicle Perception
Author: S. Yeshwanth (22211A66B2), BVRIT
"""

import torch
import torch.nn as nn
import torchvision

CLASSES = ["Car", "Pedestrian", "Cyclist", "Truck", "Barrier"]


class CameraCNN(nn.Module):
    """ResNet-18 backbone for extracting visual/semantic features from camera images."""

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = nn.Identity()
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        self.backbone = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 512)


class LiDARNet(nn.Module):
    """Point-cloud MLP encoder for extracting geometric/spatial features from LiDAR."""

    def __init__(self, in_dim: int = 3, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3)
        B, N, C = x.shape
        x_flat = x.view(B * N, C)
        feats = self.mlp(x_flat).view(B, N, -1)
        return feats.mean(dim=1)  # (B, 128)


class RadarNet(nn.Module):
    """MLP encoder for extracting motion/velocity features from radar signals."""

    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, 64)


class MultiSensorFusionModel(nn.Module):
    """
    Deep Sensor Fusion Network combining Camera, LiDAR, and Radar inputs.

    Architecture:
        - Per-sensor encoders extract modality-specific features.
        - Linear projections map all features to a common 256-d latent space.
        - Transformer encoder captures cross-modal interactions.
        - Adaptive attention aggregates the fused token sequence.
        - Task heads produce class logits, 3D position, and confidence.
    """

    def __init__(self, num_classes: int = len(CLASSES), d_model: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()

        # Sensor encoders
        self.camera = CameraCNN(freeze_backbone=True)
        self.lidar = LiDARNet(in_dim=3, out_dim=128)
        self.radar = RadarNet(in_dim=3, out_dim=64)

        # Project all modalities to common latent dimension
        self.proj_cam = nn.Linear(512, d_model)
        self.proj_lidar = nn.Linear(128, d_model)
        self.proj_radar = nn.Linear(64, d_model)

        # Transformer-based cross-modal alignment
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Adaptive attention for weighted sensor aggregation
        self.attn = nn.Linear(d_model, 1)

        # Task heads
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, cam: torch.Tensor, lidar: torch.Tensor, radar: torch.Tensor) -> dict:
        # Extract per-sensor features
        cam_f = self.proj_cam(self.camera(cam))        # (B, 256)
        lidar_f = self.proj_lidar(self.lidar(lidar))  # (B, 256)
        radar_f = self.proj_radar(self.radar(radar))  # (B, 256)

        # Stack into token sequence: (B, 3, 256)
        x = torch.stack([cam_f, lidar_f, radar_f], dim=1)

        # Cross-modal transformer fusion
        x = self.transformer(x)  # (B, 3, 256)

        # Adaptive weighted aggregation
        weights = torch.softmax(self.attn(x), dim=1)  # (B, 3, 1)
        fused = (x * weights).sum(dim=1)              # (B, 256)

        return {
            "class_logits": self.cls_head(fused),      # (B, num_classes)
            "position": self.pos_head(fused),           # (B, 3)  — tanh in [-1,1], scale x50 for meters
            "confidence": self.conf_head(fused),        # (B, 1)
        }
