"""
Training script for Multi-Sensor Fusion Model.

Usage:
    python train.py --dataroot ./data --epochs 20 --batch_size 4 --lr 1e-3
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import NuScenesMultiSensorDataset, CLASSES
from src.model import MultiSensorFusionModel


# ── argument parsing ───────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Train Multi-Sensor Fusion Model")
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Path to nuScenes dataset root (contains v1.0-mini/)")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="",
                        help="Force device: 'cpu' or 'cuda'. Auto-detects if empty.")
    return parser.parse_args()


# ── loss ───────────────────────────────────────────────────────────────────────

class FusionLoss(nn.Module):
    """
    Combined loss:
        L_total = L_classification + λ_pos * L_position + λ_conf * L_confidence
    """

    def __init__(self, lambda_pos: float = 0.5, lambda_conf: float = 0.3):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.pos_loss = nn.MSELoss()
        self.conf_loss = nn.BCELoss()
        self.lambda_pos = lambda_pos
        self.lambda_conf = lambda_conf

    def forward(self, outputs: dict, labels: torch.Tensor) -> torch.Tensor:
        # Classification
        l_cls = self.cls_loss(outputs["class_logits"], labels)

        # Position regression — use normalised label coords as proxy target
        B = labels.size(0)
        pos_target = torch.zeros(B, 3, device=labels.device)
        l_pos = self.pos_loss(outputs["position"], pos_target)

        # Confidence — treat all samples as "detected"
        conf_target = torch.ones(B, 1, device=labels.device)
        l_conf = self.conf_loss(outputs["confidence"], conf_target)

        return l_cls + self.lambda_pos * l_pos + self.lambda_conf * l_conf


# ── collate (handles variable-length LiDAR) ───────────────────────────────────

def collate_fn(batch):
    cams   = torch.stack([b["cam"]   for b in batch])
    lidars = torch.stack([b["lidar"] for b in batch])
    radars = torch.stack([b["radar"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"cam": cams, "lidar": lidars, "radar": radars, "label": labels}


# ── training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        cam    = batch["cam"].to(device)
        lidar  = batch["lidar"].to(device)
        radar  = batch["radar"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(cam, lidar, radar)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs["class_logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        cam    = batch["cam"].to(device)
        lidar  = batch["lidar"].to(device)
        radar  = batch["radar"].to(device)
        labels = batch["label"].to(device)

        outputs = model(cam, lidar, radar)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs["class_logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    print("Loading nuScenes dataset …")
    dataset = NuScenesMultiSensorDataset(dataroot=args.dataroot, version=args.version)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Train samples: {n_train} | Val samples: {n_val}")

    # Model, optimizer, scheduler, loss
    model     = MultiSensorFusionModel(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = FusionLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    print("\n── Training ──────────────────────────────")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": vl_acc,
            }, ckpt_path)
            print(f"  ✓ Saved best model  (val_acc={vl_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
