# XFeat-SuperPoint Hybrid Model

A hybrid neural network that combines **XFeat's** fast, lightweight keypoint detection head with **SuperPoint's** robust 256-dimensional descriptor manifold via a fully differentiable bicubic spatial sampling interconnect — producing a **LightGlue-compatible** feature extractor.

---

## Architecture Overview

```
Grayscale Image (B, 1, H, W)
         │
         ├──────────────────────────────────────────┐
         │                                          │
   XFeat normalise                        SuperPoint normalise
   (ImageNet-gray stats)                  ([0, 1] range)
         │                                          │
   XFeatCore                              SuperPointCore  ← FROZEN
   (encoder + kp_head)                   (encoder + desc_head)
         │                                          │
   Heatmap K                             Desc Map D
   (B, 65, H/8, W/8)                    (B, 256, H/8, W/8)
   ← TRAINABLE                           ← intercepted before upsample
         │                                          │
   Decode: softmax → pixel-shuffle → NMS            │
         │                                          │
   Keypoints (N, 2) ─────── ×(1/8) ─────────────► grid coords
                                   → normalize [-1,1]
                                   → grid_sample (bicubic)   ◄─ D
                                   → L2-normalize
                                          │
                               Descriptors (N, 256)
         │                                │
         └────────────────────────────────┘
                        │
              LightGlue Payload
         { 'keypoints':   list[(N,2)] ∈ [0,1]²  }
         { 'descriptors': list[(256,N)]           }
```

## Key Design Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| SuperPoint backbone | **Frozen** | Preserve learned geometric descriptor manifold |
| XFeat keypoint head | **Active** | Learn which locations yield best SP descriptors |
| Interpolation | **Bicubic** | Matches SuperPoint's own descriptor decoder |
| Loss | **Hinge** (λ=250, mp=1, mn=0.2) | SuperPoint §3.4 — proven descriptor learning |
| Output | **LightGlue payload** | Drop-in replacement for SP+LightGlue pipeline |

---

## Setup

### 1. Clone this repository
```bash
git clone https://github.com/MalharRane/XFeat-SuperPoint-Hybrid-Model.git
cd XFeat-SuperPoint-Hybrid-Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Link upstream models

**XFeat:**
```bash
git clone https://github.com/verlab/accelerated_features.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/accelerated_features
```

**SuperPoint** (rpautrat implementation):
```bash
git clone https://github.com/rpautrat/SuperPoint.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/SuperPoint
```

### 4. Download pretrained weights

```bash
mkdir -p weights

# SuperPoint weights (MagicLeap)
wget -O weights/superpoint_v1.pth \
  https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth

# XFeat weights (official)
wget -O weights/xfeat.pth \
  https://github.com/verlab/accelerated_features/releases/download/v1.0/xfeat.pth
```

### 5. (Optional) Download MegaDepth dataset

Follow the instructions in the [LoFTR repository](https://github.com/zju3dv/LoFTR) to download:
- MegaDepth images & depth maps
- Scene info `.npz` files (train/val splits)

---

## Training

### Phase 1 — Synthetic Pre-training (recommended)
```bash
python train.py \
  --mode synthetic \
  --data_root /path/to/coco/train2017 \
  --batch_size 8 \
  --max_epochs 30
```

### Phase 2 — MegaDepth Fine-tuning
```bash
python train.py \
  --mode megadepth \
  --data_root /path/to/megadepth \
  --resume checkpoints/best.pth \
  --lr 1e-4 \
  --max_epochs 50
```

### Google Colab
Open `XFeat_SuperPoint_Hybrid_Colab.ipynb` — it handles all setup, data download, and training automatically.

---

## Inference

```python
import torch
from models.hybrid_model import HybridModel

# Load model
model = HybridModel(xfeat_core, superpoint_core)
model.load_state_dict(torch.load('checkpoints/best.pth')['model'])
model.eval()

# Extract features (LightGlue-compatible)
with torch.no_grad():
    image = torch.rand(1, 1, 480, 640)  # grayscale [0,1]
    output = model(image)

# output['keypoints'][0]   → (N, 2) in [0,1]²
# output['descriptors'][0] → (256, N) L2-normalised

# Plug directly into LightGlue:
# matches = lightglue({'image0': output_A, 'image1': output_B})
```

---

## File Structure

```
XFeat-SuperPoint-Hybrid-Model/
├── models/
│   ├── hybrid_model.py      ← HybridModel (main nn.Module)
│   └── sampler.py           ← DifferentiableDescriptorSampler
├── losses/
│   └── hinge_loss.py        ← HomographyHingeLoss
├── data/
│   └── megadepth_dataset.py ← Dataset loaders
├── train.py                 ← Training entry point
├── config.yaml              ← Hyperparameter configuration
├── requirements.txt
└── XFeat_SuperPoint_Hybrid_Colab.ipynb  ← Google Colab notebook
```

---

## References

- **XFeat**: Potje et al., "Accelerated Features for Lightweight Image Matching", CVPR 2024
- **SuperPoint**: DeTone et al., "Self-Supervised Interest Point Detection and Description", CVPRW 2018
- **LightGlue**: Lindenberger et al., "Local Feature Matching at Light Speed", ICCV 2023
- **MegaDepth**: Li & Snavely, "MegaDepth: Learning Single-View Depth Prediction", CVPR 2018
