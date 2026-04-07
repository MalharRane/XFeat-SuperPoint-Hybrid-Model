# XFeat-SuperPoint Hybrid Model

A hybrid neural network that combines **XFeat's** fast, lightweight keypoint detection head with **SuperPoint's** robust 256-dimensional descriptor manifold via a fully differentiable bicubic spatial sampling interconnect — producing a **LightGlue-compatible** feature extractor.

---

## Architecture Overview

```
Grayscale Image (B, 1, H, W)   — H and W must be divisible by 8
         │
         ├──────────────────────────────────────────┐
         │                                          │
   XFeat normalise                        SuperPoint normalise
   (ImageNet-gray stats)                  (clamp to [0, 1])
         │                                          │
   XFeatCore                              SuperPointCore  ← FROZEN
   (encoder + kp_head)                   (encoder + desc_head)
         │                                          │
   Heatmap K                             Desc Map D
   (B, 64|65, H/8, W/8)                 (B, 256, H/8, W/8) — float32
   ← TRAINABLE (kp_head only)            ← intercepted before upsample
         │                                          │
   Decode: sigmoid/softmax               desc_map always cast to float32
       → pixel-shuffle                   (prevents AMP dtype mismatch)
       → NMS + border suppression                   │
       → top-K by score                             │
         │                                          │
   Keypoints px (N, 2) ─── ÷8 → normalize [-1,1] → grid_sample (bicubic) ◄─ D
                                                    │
                                        Descriptors (N, 256) L2-normalised
         │                                          │
         └────────────────────────────────────────-─┘
                        │
              LightGlue Payload
         { 'keypoints':   list[(N, 2)] ∈ [0,1]²   }
         { 'descriptors': list[(N, 256)] L2-normed  }
```

> **Image size constraint:** H and W must both be divisible by 8.  
> The pixel-shuffle decoder reconstructs full-resolution score maps from 8×8 cell grids; non-multiples of 8 will cause a reshape error.

## Key Design Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| SuperPoint backbone | **Frozen** | Preserve learned geometric descriptor manifold |
| XFeat keypoint head only | **Trainable** | Learn which locations yield the most discriminative SP descriptors |
| Interpolation | **Bicubic** | Matches SuperPoint's own descriptor decoder |
| Loss | **Score-weighted hinge** (λ_d=250, mp=1.0, mn=0.2) + **repeatability reward** (λ_rep=0.5) | Hinge pulls matched descriptors together; score weights gate the gradient; repeatability reward pushes scores up at geometrically stable locations |
| Gradient path | Flows through **score values** (not keypoint coordinates) | Detached NMS positions select keypoints; differentiable score values gate the loss weight, teaching the kp_head which positions are repeatable |
| AMP safety | `desc_map` always cast to **float32** before `grid_sample` | Bicubic `F.grid_sample` requires matching dtypes; SuperPoint encoder may emit float16 under autocast |
| Output format | `keypoints: (N, 2)`, `descriptors: (N, 256)` | Standard per-sample tensor layout; transpose to `(256, N)` if your matcher expects that |

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

**XFeat** (required):
```bash
git clone https://github.com/verlab/accelerated_features.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/accelerated_features
```

**SuperPoint** — use either the rpautrat implementation or write a thin wrapper that exposes `get_descriptor_map(x) → (B, 256, H/8, W/8)`:
```bash
git clone https://github.com/rpautrat/SuperPoint.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/SuperPoint
```

If neither is available the model falls back to a `SuperPointStub` for shape-compatibility testing (not trained).

### 4. Download pretrained weights

```bash
mkdir -p weights

# XFeat weights (official release — note .pt extension)
wget -O weights/xfeat.pt \
  https://github.com/verlab/accelerated_features/releases/download/v1.0/xfeat.pt

# SuperPoint weights (MagicLeap)
wget -O weights/superpoint_v1.pth \
  https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
```

### 5. (Optional) Download MegaDepth-1500

The Colab notebook expects:
```
MyDrive/hybrid_feature_matching/data/megadepth_test_1500/
    ├── megadepth_test_1500_scene_info/   ← .npz pair files
    └── megadepth_test_1500.tar           ← images + depth maps
```

Full MegaDepth can be downloaded following the [LoFTR instructions](https://github.com/zju3dv/LoFTR).

---

## Training

### Loss function

Training uses a **score-weighted hinge loss** with an optional **repeatability reward**:

```
L = L_hinge + λ_rep · L_rep

L_hinge  = Σᵢⱼ W_ij [ λ_d · s_ij · relu(mp - sim_ij)       ← positive term
                      + (1 - s_ij) · relu(sim_ij - mn) ]    ← negative term
           / (N · M)

W_ij     = score1_i · score2_j / mean(W)     ← gradient flows to kp_head
L_rep    = -(mean score at geometrically matched positions)   ← repeatability reward
```

Default hyperparameters (see `config.yaml`): λ_d=250, mp=1.0, mn=0.2, λ_rep=0.5, τ=6px.

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

### One-command 2-stage training (synthetic → MegaDepth)
```bash
python train.py \
  --config config.yaml \
  --two_stage \
  --synthetic_data_root /path/to/coco/train2017 \
  --megadepth_data_root /path/to/megadepth \
  --stage1_epochs 30 \
  --stage2_epochs 50
```

### Accuracy-focused knobs
- `max_pairs_per_scene`: increase MegaDepth supervision density per scene.
- `lr_patience`, `lr_factor`, `lr_threshold`: control faster/stronger LR decay on plateaus.
- `lambda_d`, `lambda_rep`, `correspondence_threshold`: loss sweep knobs for better positive/negative separation.
- `min_keypoint_score`: discard weak detections before top-K (reduces low-texture sky points).
- `unfreeze_at_epoch` + `unfreeze_keywords`: staged unfreezing of additional XFeat modules.
- `model_selection_metric`: choose checkpointing signal (`loss`, `sim_gap`, or `repeatability`).
- Validation now logs:
  - `sim_gap = pos_sim_mean - neg_sim_mean`
  - `repeatability_{1,2,mean}` (proxy from geometric-match coverage)

### Google Colab
Open **`XFeat_SuperPoint_Fixed_Training.ipynb`** — it handles all setup, data download, and training automatically on MegaDepth-1500.

### Fixed A/B benchmark + LightGlue match inspection

Use the same held-out pairs to compare two checkpoints and inspect matching quality:

```bash
python evaluate_ab_lightglue.py \
  --config config.yaml \
  --mode megadepth \
  --data_root /path/to/megadepth \
  --scene_info_dir /path/to/scene_info \
  --old_ckpt /path/to/old_best.pth \
  --new_ckpt /path/to/new_best.pth \
  --num_pairs 100 \
  --mma_thresholds 1,3,5 \
  --precision_threshold 3.0 \
  --save_vis_count 10
```

Outputs:
- `summary.yaml` with A/B metrics and `delta_new_minus_old`
- Mean metrics: `inlier_ratio`, `mma@1/3/5px`, `precision`, `n_matches`
- Sanity diagnostics: `sim_gap`, `repeatability_mean`
- Match visualizations (inlier/outlier colored lines) in `benchmarks/ab_lightglue/{old_vis,new_vis}/`

> Requires LightGlue in your environment (`pip install lightglue`).

---

## Inference

```python
import torch
from models.hybrid_model import HybridModel

# Instantiate (provide your own xfeat_core and superpoint_core)
model = HybridModel(
    xfeat_core=xfeat_core,
    superpoint_core=superpoint_core,
    num_keypoints=512,      # max keypoints per image
    nms_radius=8,           # NMS window half-radius (pixels)
    descriptor_dim=256,
    border_margin=16,       # suppress detections near image borders
)
model.load_state_dict(torch.load('checkpoints/best.pth')['model'])
model.eval()

# Extract features — image H and W must be divisible by 8
with torch.no_grad():
    image = torch.rand(1, 1, 480, 640)   # grayscale float32 in [0, 1]
    output = model(image)

# output['keypoints'][0]    → (N, 2)   normalised to [0, 1]²
# output['descriptors'][0]  → (N, 256) L2-normalised

# Plug into LightGlue (expects (256, N) — transpose if needed):
# matches = lightglue({'image0': output_A, 'image1': output_B})
```

### `forward_train` — training mode with intermediate outputs

```python
out = model.forward_train(image)
# out['keypoints']    → list[(N, 2)]   normalised keypoint coords
# out['descriptors']  → list[(N, 256)] L2-normalised descriptors
# out['keypoints_px'] → list[(N, 2)]   pixel-space keypoints
# out['scores']       → list[(N,)]     differentiable detection scores
# out['heatmap']      → (B, C, H/8, W/8) raw XFeat logits
```

---

## File Structure

```
XFeat-SuperPoint-Hybrid-Model/
├── models/
│   ├── __init__.py
│   ├── hybrid_model.py      ← HybridModel (main nn.Module)
│   └── sampler.py           ← DifferentiableDescriptorSampler
├── losses/
│   ├── __init__.py
│   └── hinge_loss.py        ← HomographyHingeLoss (score-weighted + repeatability reward)
├── data/
│   ├── __init__.py
│   └── megadepth_dataset.py ← MegaDepth & synthetic dataset loaders
├── utils/
│   └── __init__.py
├── DOCS/
│   ├── XFeat.pdf
│   ├── Superpoint.pdf
│   └── LightGlue.pdf
├── train.py                 ← Training entry point (CLI + config.yaml)
├── config.yaml              ← All hyperparameters
├── requirements.txt
├── XFeat_SuperPoint_Fixed_Training.ipynb          ← Colab: MegaDepth-1500 training
└── XFeat_SuperPoint_MegaDepth1500_Colab_fixed2.ipynb  ← Colab: alternate MegaDepth notebook
```

---

## Known Issues & Fixes

| Issue | Status | Fix |
|-------|--------|-----|
| `RuntimeError` at first training batch under AMP | ✅ Fixed | `desc_map` is cast to float32 before `F.grid_sample`; bicubic mode requires matching dtypes and is unreliable in float16 |
| In-place border zeroing broke autograd graph | ✅ Fixed | Out-of-place masking in `_decode_xfeat_heatmap` |
| Hard NMS disconnected score values from loss | ✅ Fixed | NMS uses detached positions for index selection; score *values* at those positions remain differentiable |
| Descriptors were `(256, N)` but loss expected `(N, 256)` | ✅ Fixed | Model now outputs `(N, 256)` directly |
| Auxiliary L2 loss pushed all weights toward zero | ✅ Fixed | Replaced with score-weighted hinge loss + repeatability reward |
| SuperPoint params unfrozen but received zero gradient | ✅ Fixed | Removed incorrect `requires_grad_(True)` on frozen SP params |
| Approximate homography inaccurate for 3-D scenes with parallax | ✅ Fixed | Optional depth-based dense warp field (when depth maps are available) |
| Resized MegaDepth images used unscaled intrinsics | ✅ Fixed | Scale K₁/K₂ to resized image size before homography and depth reprojection |
| Mixed batches dropped depth warp if any sample was missing it | ✅ Fixed | Preserve per-sample optional warp fields and fallback to homography only for missing samples |
| No validation loop | ✅ Fixed | Proper val loop with early stopping |
| `MultiStepLR` with hard-coded milestones | ✅ Fixed | `ReduceLROnPlateau` adapts to actual val loss |
| Image H/W not divisible by 8 | ⚠️ User error | Pixel-shuffle decoder assumes 8×8 cells; resize inputs to multiples of 8 |

---

## References

- **XFeat**: Potje et al., "Accelerated Features for Lightweight Image Matching", CVPR 2024  
- **SuperPoint**: DeTone et al., "Self-Supervised Interest Point Detection and Description", CVPRW 2018  
- **LightGlue**: Lindenberger et al., "Local Feature Matching at Light Speed", ICCV 2023  
- **MegaDepth**: Li & Snavely, "MegaDepth: Learning Single-View Depth Prediction", CVPR 2018
