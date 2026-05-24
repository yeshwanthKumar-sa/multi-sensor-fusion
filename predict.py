"""
Inference script — run the trained fusion model on a single nuScenes sample.

Usage:
    python predict.py --dataroot ./data --checkpoint checkpoints/best_model.pth --sample_idx 0
"""

import argparse
import torch

from src.dataset import NuScenesMultiSensorDataset, CLASSES, load_camera, load_lidar, load_radar
from src.model import MultiSensorFusionModel


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with Multi-Sensor Fusion Model")
    parser.add_argument("--dataroot",   type=str, required=True)
    parser.add_argument("--version",    type=str, default="v1.0-mini")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the nuScenes sample to run")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model = MultiSensorFusionModel(num_classes=len(CLASSES))
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load one sample
    dataset = NuScenesMultiSensorDataset(dataroot=args.dataroot, version=args.version)
    sample  = dataset.nusc.sample[args.sample_idx]

    cam   = load_camera(dataset.nusc, sample, args.dataroot).to(device)   # (1,3,224,224)
    lidar = load_lidar(dataset.nusc, sample, args.dataroot).to(device)    # (1,N,3)
    radar = load_radar(dataset.nusc, sample, args.dataroot).to(device)    # (1,3)

    with torch.no_grad():
        out = model(cam, lidar, radar)

    cls_id     = out["class_logits"].argmax(dim=1).item()
    pos        = out["position"][0] * 50          # scale tanh output to ±50 m
    distance   = torch.norm(pos).item()
    speed      = torch.norm(radar).item() * 3.6   # rough km/h proxy
    confidence = out["confidence"].item() * 100

    print("\n── Prediction ─────────────────────────────")
    print(f"  Object class : {CLASSES[cls_id]}")
    print(f"  3D Position  : ({pos[0]:.1f} m, {pos[1]:.1f} m, {pos[2]:.1f} m)")
    print(f"  Distance     : {distance:.1f} m")
    print(f"  Speed (est.) : {speed:.1f} km/h")
    print(f"  Confidence   : {confidence:.0f}%")


if __name__ == "__main__":
    main()
