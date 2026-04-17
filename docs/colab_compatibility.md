# Colab Compatibility Matrix

## Tested combinations

| Notebook date | Python | torch / torchvision | accelerated_features | SuperPoint | Hybrid repo |
|---|---|---|---|---|---|
| 2026-04-17 | 3.10.x (Colab) | 2.3.1 / 0.18.1 | `e92685f57f8318b18725c5c8c0bd28c7fe188d9a` | `1411bbd68c50163555d39c1b26e9e046ebd48f27` | pinned by `HYBRID_REPO_REF` (default: current clone) |

## Known bad combinations

- `accelerated_features` on moving `main` without pinning can break HybridModel integration.
- Any setup where XFeat does not expose a callable `forward` and no compatible wrapper API (`detectAndCompute`/`detect_and_compute`/`extract`/`detect`) causes forward preflight failure.
- Image sizes not divisible by 8 (`image_height`, `image_width`) are unsupported by the decoder path.
