import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self):
        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def compute(self, feature_map, output, labels):
        feature_map.register_hook(self.save_gradients)
        loss = output[torch.arange(len(labels)), labels].sum()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * feature_map).sum(dim=1))
        cam = cam / (cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-8)
        return cam
