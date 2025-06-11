import torch
import torch.nn as nn
from models.extractor import DualModalExtractor
from models.alignment import SparsePrototypeAlignment
from models.fusion import CrossModalFusion
from config import config

class ReIDModel(nn.Module):
    def __init__(self, output_dim=config.OUTPUT_DIM, num_classes=config.NUM_CLASSES, k=config.K_REGIONS, g=config.G_GROUPS):
        super(ReIDModel, self).__init__()
        self.extractor = DualModalExtractor()
        self.alignment = SparsePrototypeAlignment(k=k)
        self.fusion = CrossModalFusion(input_dim=output_dim, g=g)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x_vis, x_ir, labels=None, training=True):
        vis_out, ir_out, vis_feat, ir_feat = self.extractor(x_vis, x_ir)
        if vis_out is not None:
            vis_out = self.pool(vis_out).view(vis_out.size(0), -1)
            assert vis_out.shape[1] == config.OUTPUT_DIM, f"vis_out dim {vis_out.shape[1]} != {config.OUTPUT_DIM}"
        if ir_out is not None:
            ir_out = self.pool(ir_out).view(ir_out.size(0), -1)
            assert ir_out.shape[1] == config.OUTPUT_DIM, f"ir_out dim {ir_out.shape[1]} != {config.OUTPUT_DIM}"

        if training and labels is not None:
            if vis_out is not None and ir_out is not None:
                vis_cam = self.alignment.gradcam.compute(vis_feat, vis_out, labels)
                ir_cam = self.alignment.gradcam.compute(ir_feat, ir_out, labels)
                vis_regions = self.alignment.select_regions(vis_cam)
                ir_regions = self.alignment.select_regions(ir_cam)
                vis_sparse = self.alignment.sparse_features(vis_feat, vis_regions)
                ir_sparse = self.alignment.sparse_features(ir_feat, ir_regions)
                vis_prototypes = self.alignment.compute_prototypes(vis_sparse, labels, modality=0)
                ir_prototypes = self.alignment.compute_prototypes(ir_sparse, labels, modality=1)
                fused = self.fusion(vis_out, ir_out)
                logits = self.classifier(fused)
                return logits, fused  # 返回 logits 和融合特征
            else:
                raise ValueError("Training requires both vis and ir inputs")
        else:
            if vis_out is not None and ir_out is not None:
                fused = self.fusion(vis_out, ir_out)
            elif vis_out is not None:
                fused = vis_out
            elif ir_out is not None:
                fused = ir_out
            else:
                raise ValueError("At least one of vis or ir must be provided")
            return self.classifier(fused) if training else fused