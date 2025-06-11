import torch
import torch.nn as nn
import torchvision.models as models
from config import config
from torchvision.models import MobileNet_V3_Small_Weights

class DualModalExtractor(nn.Module):
    def __init__(self, output_dim=576):
        super(DualModalExtractor, self).__init__()
        self.vis_branch = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.ir_branch = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.vis_norm = nn.InstanceNorm2d(576)
        self.ir_norm = nn.InstanceNorm2d(576)

    def adaptive_downsample(self, x, modality):
        mu = x.mean(dim=[2, 3])
        sigma2 = ((x - mu.unsqueeze(2).unsqueeze(3)) ** 2).mean(dim=[2, 3])
        stride = 1 if sigma2.mean() > 0.1 else 2
        x = nn.functional.avg_pool2d(x, kernel_size=3, stride=stride, padding=1)
        return x

    def forward(self, x_vis, x_ir):
        vis_out, vis_feat = None, None
        if x_vis is not None:
            vis_feat = self.vis_branch.features(x_vis)
            vis_feat = self.vis_norm(vis_feat)
            vis_out = self.adaptive_downsample(vis_feat, "vis")
        ir_out, ir_feat = None, None
        if x_ir is not None:
            ir_feat = self.ir_branch.features(x_ir)
            ir_feat = self.ir_norm(ir_feat)
            ir_out = self.adaptive_downsample(ir_feat, "ir")
        return vis_out, ir_out, vis_feat, ir_feat
# class DualModalExtractor(nn.Module):
#     def __init__(self, output_dim=config.OUTPUT_DIM):
#         super(DualModalExtractor, self).__init__()
#         self.vis_branch = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
#         self.ir_branch = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
#         print("Vis branch pretrained weights loaded:", self.vis_branch.features[0][0].weight[0][0][0][:5])
#         # 移除 classifier 修改，保留原始特征提取
#         self.vis_norm = nn.InstanceNorm2d(576)
#         self.ir_norm = nn.InstanceNorm2d(576)
#
#     def adaptive_downsample(self, x, modality):
#         mu = x.mean(dim=[2, 3])
#         sigma2 = ((x - mu.unsqueeze(2).unsqueeze(3)) ** 2).mean(dim=[2, 3])
#         stride = 1 if sigma2.mean() > 0.1 else 2
#         print(f"{modality} stride: {stride}, input shape: {x.shape}")
#         x = nn.functional.avg_pool2d(x, kernel_size=3, stride=stride, padding=1)
#         print(f"{modality} output shape: {x.shape}")
#         return x
#
#     def forward(self, x_vis, x_ir):
#         # 提取特征，保持 4D 张量
#         vis_feat = self.vis_branch.features(x_vis)  # [16, 576, 8, 4]
#         vis_feat = self.vis_norm(vis_feat)
#         vis_out = self.adaptive_downsample(vis_feat, "vis")  # [16, 576, H, W]
#
#         ir_feat = self.ir_branch.features(x_ir)  # [16, 576, 8, 4]
#         ir_feat = self.ir_norm(ir_feat)
#         ir_out = self.adaptive_downsample(ir_feat, "ir")  # [16, 576, H, W]
#
#         return vis_out, ir_out, vis_feat, ir_feat
