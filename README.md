# Multi-Sensor Fusion for Autonomous Vehicle Perception

> **B.Tech Major Project** — S. Yeshwanth (22211A66B2)  
> Department of CSE (AI & ML), B V Raju Institute of Technology  
> Guided by: Mr. Kunal Devidas Gaikwad & Mrs. Indumathi V

---

## Overview

A deep learning-based multi-sensor fusion framework that combines **Camera**, **LiDAR**, and **Radar** data for robust autonomous vehicle perception.

### Key Features
- **ResNet-18** CNN backbone for visual/semantic feature extraction
- **Point-cloud MLP** for LiDAR geometric features
- **Radar MLP** for velocity/distance features
- **Transformer encoder** for cross-modal feature alignment
- **Adaptive attention** for dynamic sensor weighting
- Task heads for **object classification**, **3D position estimation**, and **confidence scoring**

### Model Architecture

```
Camera  ──► CameraCNN (ResNet-18) ──► Linear(512→256) ──┐
LiDAR   ──► LiDARNet  (MLP)       ──► Linear(128→256) ──┤──► Transformer ──► Adaptive Attention ──► Task Heads
Radar   ──► RadarNet  (MLP)       ──► Linear(64→256)  ──┘
```

---

## Project Structure

```
multi-sensor-fusion/
├── src/
│   ├── model.py        # Model architecture (CameraCNN, LiDARNet, RadarNet, FusionModel)
│   ├── dataset.py      # nuScenes dataset loader & preprocessing
│   └── __init__.py
├── train.py            # Training script with CLI args
├── predict.py          # Inference script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yeshwanthKumar-sa/multi-sensor-fusion.git
cd multi-sensor-fusion
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the nuScenes dataset
1. Register at [https://www.nuscenes.org/](https://www.nuscenes.org/)
2. Download **nuScenes mini** (~4 GB)
3. Extract to `./data/` so the structure looks like:
```
data/
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
```

---

## Training

```bash
python train.py \
    --dataroot ./data \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-3 \
    --save_dir checkpoints
```

| Argument | Default | Description |
|---|---|---|
| `--dataroot` | *(required)* | Path to nuScenes dataset root |
| `--version` | `v1.0-mini` | Dataset version |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--val_split` | `0.2` | Fraction of data for validation |
| `--save_dir` | `checkpoints` | Directory for saving `.pth` files |
| `--device` | *(auto)* | Force `cpu` or `cuda` |

The best checkpoint is saved automatically to `checkpoints/best_model.pth`.

---

## Inference

```bash
python predict.py \
    --dataroot ./data \
    --checkpoint checkpoints/best_model.pth \
    --sample_idx 0
```

**Example output:**
```
── Prediction ─────────────────────────────
  Object class : Car
  3D Position  : (12.3 m, -4.1 m, 0.8 m)
  Distance     : 13.1 m
  Speed (est.) : 42.5 km/h
  Confidence   : 87%
```

---

## Results (nuScenes mini)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Camera Only | 81.2% | 79.5% | 80.3% | 79.9% |
| LiDAR Only | 85.6% | 84.2% | 83.9% | 84.0% |
| Radar Only | 73.4% | 71.8% | 72.6% | 72.2% |
| **Multi-Sensor Fusion** | **92.8%** | **91.5%** | **92.1%** | **91.8%** |

---

## Classes Detected

| ID | Class |
|---|---|
| 0 | Car |
| 1 | Pedestrian |
| 2 | Cyclist |
| 3 | Truck |
| 4 | Barrier |

---

## Tech Stack

- Python 3.10+
- PyTorch 2.x
- TorchVision
- nuScenes devkit
- NumPy, Pillow

---

## Paper

**Title:** Deep Learning Based Adaptive Multi-Sensor Fusion for Robust Object Detection in Autonomous Vehicles  
**Conference:** ICETEST-2026 (IEEE Sponsored), Malla Reddy Institute of Technology & Science  
**Paper ID:** ICTACSE10@-002

---

## License

This project is for academic purposes. Dataset usage governed by the [nuScenes terms](https://www.nuscenes.org/terms-of-use).
