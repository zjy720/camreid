# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from models.reid_model import ReIDModel
# from data.dataloader import SYSUDataset, custom_collate_fn
# from config import config
# from tqdm import tqdm
# import torch.distributions as dist
# import torch.nn.functional as F
#
#
# def compute_mmd_loss(source_feats, target_feats):
#     source_dist = dist.Normal(source_feats.mean(dim=0), source_feats.std(dim=0))
#     target_dist = dist.Normal(target_feats.mean(dim=0), target_feats.std(dim=0))
#     mmd = 0
#     for i in range(source_feats.size(1)):
#         mmd += (source_dist.loc[i] - target_dist.loc[i]) ** 2 + (source_dist.scale[i] - target_dist.scale[i]) ** 2
#     return mmd
#
#
# class CombinedLoss(nn.Module):
#     def __init__(self, lambda_1=0.5, lambda_2=0.3, lambda_3=0.2):
#         super().__init__()
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2
#         self.lambda_3 = lambda_3
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
#
#     def forward(self, logits, feats, labels):
#         ce_loss = self.ce_loss(logits, labels)
#         anchor, positive, negative = feats[::3], feats[1::3], feats[2::3]
#         triplet_loss = self.triplet_loss(anchor, positive, negative)
#         mmd_loss = compute_mmd_loss(feats[:len(feats) // 2], feats[len(feats) // 2:])
#         return self.lambda_1 * ce_loss + self.lambda_2 * triplet_loss + self.lambda_3 * mmd_loss
#
#
# def train():
#     model = ReIDModel().to(config.DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     loss_fn = CombinedLoss(lambda_1=config.LAMBDA_1, lambda_2=config.LAMBDA_2, lambda_3=config.LAMBDA_3)
#
#     dataset = SYSUDataset(root=config.DATA_ROOT, mode="train")
#     dataloader = DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#
#         collate_fn=custom_collate_fn
#     )
#
#     model.train()
#     for epoch in range(config.NUM_EPOCHS):
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
#         for batch_idx, (x_vis, x_ir, labels) in enumerate(progress_bar):
#             if x_vis is None or x_ir is None:
#                 print(f"Warning: Skipping invalid batch {batch_idx}")
#                 continue
#             x_vis = x_vis.to(config.DEVICE)
#             x_ir = x_ir.to(config.DEVICE)
#             labels = labels.to(config.DEVICE)
#
#             optimizer.zero_grad()
#             logits, feats = model(x_vis, x_ir, training=True)
#             loss = loss_fn(logits, feats, labels)
#             loss.backward()
#             optimizer.step()
#
#             progress_bar.set_postfix({"loss": loss.item()})
#
#         scheduler.step()
#
#     torch.save(model.state_dict(), config.MODEL_PATH)
#     return model
#
#
# if __name__ == "__main__":
#     train()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.reid_model import ReIDModel
from data.dataloader import SYSUDataset, custom_collate_fn
from config import config
from tqdm import tqdm
import torch.distributions as dist
import torch.nn.functional as F

def compute_mmd_loss(source_feats, target_feats):
    source_dist = dist.Normal(source_feats.mean(dim=0), source_feats.std(dim=0))
    target_dist = dist.Normal(target_feats.mean(dim=0), target_feats.std(dim=0))
    mmd = 0
    for i in range(source_feats.size(1)):
        mmd += (source_dist.loc[i] - target_dist.loc[i]) ** 2 + (source_dist.scale[i] - target_dist.scale[i]) ** 2
    return mmd

class CombinedLoss(nn.Module):
    def __init__(self, lambda_1=0.5, lambda_2=0.3, lambda_3=0.2):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, logits, feats, labels):
        ce_loss = self.ce_loss(logits, labels)
        # 三元组采样：确保样本数一致
        batch_size = feats.size(0) // 2  # feats = [RGB, IR]
        rgb_feats = feats[:batch_size]  # 前半部分是 RGB
        ir_feats = feats[batch_size:]  # 后半部分是 IR

        # 计算可用的三元组数量
        num_triplets = min(batch_size, len(labels)) // 3  # 每组三元组需要3个样本
        if num_triplets > 0:
            # 随机选择三元组索引
            indices = torch.randperm(batch_size)[:num_triplets * 3]
            anchor_indices = indices[:num_triplets]
            positive_indices = indices[:num_triplets]  # 同一身份的 IR
            negative_indices = indices[num_triplets:2 * num_triplets]  # 不同身份的 RGB

            # 确保样本数一致
            anchor = rgb_feats[anchor_indices]
            positive = ir_feats[positive_indices]
            negative = rgb_feats[negative_indices]

            print(f"Triplet shapes: anchor {anchor.shape}, positive {positive.shape}, negative {negative.shape}")
            triplet_loss = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_loss = torch.tensor(0.0, device=feats.device)  # 跳过小批次

        mmd_loss = compute_mmd_loss(rgb_feats, ir_feats)
        return self.lambda_1 * ce_loss + self.lambda_2 * triplet_loss + self.lambda_3 * mmd_loss


def train():
    model = ReIDModel().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = CombinedLoss(lambda_1=config.LAMBDA_1, lambda_2=config.LAMBDA_2, lambda_3=config.LAMBDA_3)

    dataset = SYSUDataset(root=config.DATA_ROOT, mode="train")
    print(f"Dataset size: {len(dataset)} samples")
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last = True  # 丢弃最后一个不完整批次
    )

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        for batch_idx, (x_vis, x_ir, rgb_labels, ir_labels) in enumerate(progress_bar):
            # 验证 RGB 和 IR 标签一致
            if not torch.equal(rgb_labels, ir_labels):
                raise ValueError(f"RGB and IR labels mismatch at batch {batch_idx}")
            labels = rgb_labels.to(config.DEVICE)
            x_vis = x_vis.to(config.DEVICE)
            x_ir = x_ir.to(config.DEVICE)

            optimizer.zero_grad()
            logits, fused = model(x_vis, x_ir, labels=labels, training=True)
            # 拼接 vis_out 和 ir_out 作为 feats
            vis_out, ir_out, _, _ = model.extractor(x_vis, x_ir)
            vis_out = model.pool(vis_out).view(vis_out.size(0), -1)
            ir_out = model.pool(ir_out).view(ir_out.size(0), -1)
            feats = torch.cat([vis_out, ir_out], dim=0)  # [batch_size * 2, output_dim]
            print(f"Batch {batch_idx}: feats shape {feats.shape}, labels shape {labels.shape}")
            loss = loss_fn(logits, feats, labels)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item()})

        scheduler.step()

    torch.save(model.state_dict(), config.MODEL_PATH)
    return model

if __name__ == "__main__":
    train()

