import torch
import torch.nn as nn
from utils.gradcam import GradCAM
from config import config

class SparsePrototypeAlignment(nn.Module):
    def __init__(self, k=config.K_REGIONS, K=config.K_SHOTS):
        super(SparsePrototypeAlignment, self).__init__()
        self.k = k
        self.K = K
        self.gradcam = GradCAM()
        self.prototypes = nn.Parameter(
            torch.randn(config.NUM_CLASSES, 2, config.OUTPUT_DIM * k), requires_grad=True
        )
        self.counts = torch.zeros(config.NUM_CLASSES, 2).to(config.DEVICE)

    def select_regions(self, cam):
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        _, indices = cam_flat.topk(self.k, dim=1)
        return indices

    def sparse_features(self, feature_map, regions):
        B, C, H, W = feature_map.shape
        sparse_feats = []
        for b in range(B):
            idx = regions[b]
            feat = feature_map[b, :, idx // W, idx % W]  # [C, k]
            sparse_feats.append(feat.flatten())  # [C*k]
        return torch.stack(sparse_feats)

    def compute_prototypes(self, features, labels, modality=0):
        """features: [B, C*k], labels: [B], modality: 0 (vis) or 1 (ir)"""
        C = config.NUM_CLASSES
        batch_prototypes = torch.zeros(C, features.shape[1]).to(config.DEVICE)
        for c in range(C):
            mask = (labels == c)
            if mask.sum() > 0:
                samples = features[mask][:self.K]
                batch_prototypes[c] = samples.mean(dim=0)
                self.counts[c, modality] = min(mask.sum(), self.K)
                if mask.sum() < self.K:
                    print(f"Warning: Class {c} has {mask.sum()} samples, using {mask.sum()} for prototype")
            elif self.counts[c, modality] == 0:
                batch_prototypes[c] = torch.randn(features.shape[1]).to(config.DEVICE) * 0.01
            else:
                batch_prototypes[c] = self.prototypes[c, modality].detach()
        # Update with EMA and normalize
        self.prototypes.data[:, modality] = 0.9 * self.prototypes.data[:, modality] + 0.1 * batch_prototypes
        self.prototypes.data[:, modality] /= (self.prototypes.data[:, modality].norm(dim=1, keepdim=True) + 1e-8)
        return self.prototypes[:, modality]