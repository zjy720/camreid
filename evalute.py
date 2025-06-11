import torch
from models.reid_model import ReIDModel
from data.dataloader import SYSUDataset, custom_collate_fn
from torch.utils.data import DataLoader
import numpy as np
from config import config
from ptflops import get_model_complexity_info
import pandas as pd

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP = [], []
    num_valid_q = 0
    skipped_pids = []
    for q_idx in range(num_q):
        q_camid, q_pid = q_camids[q_idx], q_pids[q_idx]
        order = indices[q_idx]
        keep = q_camid != g_camids[order]
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            print(f"Warning: Query PID {q_pid} has no matches in gallery")
            skipped_pids.append(q_pid)
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = orig_cmc.sum()
        tmp_cmc = dir = os.path.join(config.DATA_ROOT, cam)
    if os.path.isdir(cam_dir):
        ids = set([d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir, d)) and d.isdigit()])
        print(f"{cam}: {len(ids)} IDs, max ID: {max(ids, default='None')}")
        valid_ids = ids if valid_ids is None else valid_ids.intersection(ids)
    else:
        print(f"Warning: Camera directory {cam_dir} does not exist")
        valid_ids = set() if valid_ids is None else valid_ids

# 检查交集结果
if not valid_ids:
    raise ValueError("No valid IDs found in gallery cameras")
valid_ids = sorted(valid_ids)

# 打印结果
print(f"Valid gallery IDs: {len(valid_ids)} IDs, max ID: {valid_ids[-1] if valid_ids else 'None'}")
print(f"First few IDs: {valid_ids[:5] if valid_ids else []}")

# 写入新的 test_id1.txt 文件
output_file = os.path.join(root, 'exp/test_id1.txt')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    f.write(','.join(str(int(id)) for id in valid_ids))
print(f"New file created: {output_file}")