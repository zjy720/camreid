import torch
import torch.nn as nn
from config import config


class CrossModalFusion(nn.Module):
    def __init__(self, input_dim=config.OUTPUT_DIM, g=config.G_GROUPS):
        super(CrossModalFusion, self).__init__()
        self.g = g
        self.proj_vis = nn.Linear(input_dim, input_dim // 2)
        self.proj_ir = nn.Linear(input_dim, input_dim // 2)
        embed_dim = input_dim // g
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2)
        print(f"Fusion embed_dim: {embed_dim}")

    def forward(self, vis_feat, ir_feat):
        if vis_feat.shape[-1] != config.OUTPUT_DIM or ir_feat.shape[-1] != config.OUTPUT_DIM:
            raise ValueError(
                f"Expected feature dim {config.OUTPUT_DIM}, got vis: {vis_feat.shape[-1]}, ir: {ir_feat.shape[-1]}")

        vis_proj = self.proj_vis(vis_feat)
        ir_proj = self.proj_ir(ir_feat)
        cat_feat = torch.cat([vis_proj, ir_proj], dim=1)
        B, C = cat_feat.shape
        cat_feat = cat_feat.view(B, self.g, C // self.g).transpose(0, 1)

        # Optimize attention computation
        attn_output, _ = self.attn(cat_feat, cat_feat, cat_feat)
        fused = attn_output.transpose(0, 1).reshape(B, C)
        return fused