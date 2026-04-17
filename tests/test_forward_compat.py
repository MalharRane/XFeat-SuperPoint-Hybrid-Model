import unittest

import torch
import torch.nn as nn

from models.hybrid_model import HybridModel


class _SuperPointStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.desc_head = nn.Conv2d(1, 256, kernel_size=1)

    def get_descriptor_map(self, x):
        return self.desc_head(torch.nn.functional.avg_pool2d(x, kernel_size=8, stride=8))


class _XFeatForwardStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.kp_head = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        logits = self.kp_head(torch.nn.functional.avg_pool2d(x, kernel_size=8, stride=8))
        return {'keypoints': logits}


class _XFeatAdapterStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.kp_head = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        raise NotImplementedError("forward intentionally unsupported in wrapper stub")

    def detectAndCompute(self, x):
        b, _, h, w = x.shape
        kps = []
        scores = []
        for _ in range(b):
            kps.append(torch.tensor([[w * 0.25, h * 0.25], [w * 0.5, h * 0.5]], dtype=torch.float32))
            scores.append(torch.tensor([1.0, 0.8], dtype=torch.float32))
        return {'keypoints': kps, 'scores': scores}


class HybridForwardCompatTest(unittest.TestCase):
    def test_forward_train_smoke_forward_api(self):
        model = HybridModel(
            xfeat_core=_XFeatForwardStub(),
            superpoint_core=_SuperPointStub(),
            num_keypoints=32,
        )
        x = torch.rand(2, 1, 64, 64)
        out = model.forward_train(x)
        self.assertIn('keypoints', out)
        self.assertIn('descriptors', out)
        self.assertEqual(len(out['keypoints']), 2)
        self.assertEqual(out['descriptors'][0].shape[1], 256)

    def test_forward_train_smoke_adapter_api(self):
        model = HybridModel(
            xfeat_core=_XFeatAdapterStub(),
            superpoint_core=_SuperPointStub(),
            num_keypoints=16,
        )
        x = torch.rand(1, 1, 64, 64)
        out = model.forward_train(x)
        self.assertEqual(out.get('xfeat_adapter_path'), 'detectAndCompute')
        self.assertEqual(len(out['keypoints']), 1)
        self.assertEqual(out['descriptors'][0].shape[1], 256)


if __name__ == '__main__':
    unittest.main()
