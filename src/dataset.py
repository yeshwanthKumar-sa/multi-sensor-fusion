"""
Dataset utilities for Multi-Sensor Fusion (nuScenes).
Handles loading, preprocessing, and normalization of camera, LiDAR, and radar data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    raise ImportError("Install nuscenes-devkit:  pip install nuscenes-devkit")

# ── constants ──────────────────────────────────────────────────────────────────
CLASSES = ["Car", "Pedestrian", "Cyclist", "Truck", "Barrier"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

NUSCENES_CLASS_MAP = {
    "vehicle.car": "Car",
    "human.pedestrian.adult": "Pedestrian",
    "human.pedestrian.child": "Pedestrian",
    "vehicle.bicycle": "Cyclist",
    "vehicle.truck": "Truck",
    "movable_object.barrier": "Barrier",
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

MAX_LIDAR_POINTS = 1024


# ── helpers ────────────────────────────────────────────────────────────────────

def load_camera(nusc: "NuScenes", sample: dict, dataroot: str) -> torch.Tensor:
    """Load and preprocess front-camera image → (1, 3, 224, 224)."""
    token = sample["data"]["CAM_FRONT"]
    data = nusc.get("sample_data", token)
    path = os.path.join(dataroot, data["filename"])
    img = Image.open(path).convert("RGB")
    return IMG_TRANSFORM(img).unsqueeze(0)


def load_lidar(nusc: "NuScenes", sample: dict, dataroot: str,
               max_pts: int = MAX_LIDAR_POINTS) -> torch.Tensor:
    """Load and preprocess LiDAR point cloud → (1, N, 3)."""
    token = sample["data"]["LIDAR_TOP"]
    data = nusc.get("sample_data", token)
    path = os.path.join(dataroot, data["filename"])

    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    xyz = points[:, :3]

    # Random sub-sample to fixed size
    if xyz.shape[0] > max_pts:
        idx = np.random.choice(xyz.shape[0], max_pts, replace=False)
        xyz = xyz[idx]

    # Normalize each point to unit sphere
    norms = np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-6
    xyz = xyz / norms

    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)


def load_radar(nusc: "NuScenes", sample: dict, dataroot: str) -> torch.Tensor:
    """Load and preprocess front-radar signal → (1, 3)."""
    token = sample["data"]["RADAR_FRONT"]
    data = nusc.get("sample_data", token)
    path = os.path.join(dataroot, data["filename"])

    raw = np.fromfile(path, dtype=np.float32)
    feat = np.array([raw.mean(), raw.std(), raw.max()], dtype=np.float32)

    norm = np.linalg.norm(feat)
    if not np.isfinite(norm) or norm < 1e-6:
        feat = np.zeros(3, dtype=np.float32)
    else:
        feat /= norm

    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, 3)


def get_label(nusc: "NuScenes", sample: dict) -> int:
    """Return the class index of the first annotated object in the sample."""
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        category = ann["category_name"]
        for key, mapped in NUSCENES_CLASS_MAP.items():
            if category.startswith(key):
                return CLASS_TO_IDX[mapped]
    return 0  # default → "Car"


# ── Dataset ────────────────────────────────────────────────────────────────────

class NuScenesMultiSensorDataset(Dataset):
    """
    PyTorch Dataset wrapping nuScenes mini.

    Args:
        dataroot:  Path to the nuScenes dataset root.
        version:   Dataset version, e.g. 'v1.0-mini'.
        verbose:   Print nuScenes loading info.
    """

    def __init__(self, dataroot: str, version: str = "v1.0-mini", verbose: bool = False):
        self.dataroot = dataroot
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.samples = self.nusc.sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        cam = load_camera(self.nusc, sample, self.dataroot).squeeze(0)      # (3, 224, 224)
        lidar = load_lidar(self.nusc, sample, self.dataroot).squeeze(0)     # (N, 3)
        radar = load_radar(self.nusc, sample, self.dataroot).squeeze(0)     # (3,)
        label = get_label(self.nusc, sample)

        return {
            "cam": cam,
            "lidar": lidar,
            "radar": radar,
            "label": torch.tensor(label, dtype=torch.long),
        }
