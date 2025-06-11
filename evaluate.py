# # import torch
# # from models.reid_model import ReIDModel
# # from data.dataloader import SYSUDataset, custom_collate_fn
# # from torch.utils.data import DataLoader
# # import numpy as np
# # from config import config
# # from ptflops import get_model_complexity_info
# # import pandas as pd
# #
# # def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
# #     num_q, num_g = distmat.shape
# #     if num_g < max_rank:
# #         max_rank = num_g
# #     indices = np.argsort(distmat, axis=1)
# #     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
# #     all_cmc, all_AP = [], []
# #     num_valid_q = 0
# #     skipped_pids = []
# #     for q_idx in range(num_q):
# #         q_camid, q_pid = q_camids[q_idx], q_pids[q_idx]
# #         order = indices[q_idx]
# #         keep = q_camid != g_camids[order]
# #         orig_cmc = matches[q_idx][keep]
# #         if not np.any(orig_cmc):
# #             print(f"Warning: Query PID {q_pid} has no matches in gallery")
# #             skipped_pids.append(q_pid)
# #             continue
# #         cmc = orig_cmc.cumsum()
# #         cmc[cmc > 1] = 1
# #         all_cmc.append(cmc[:max_rank])
# #         num_rel = orig_cmc.sum()
# #         tmp_cmc = dir = os.path.join(config.DATA_ROOT, cam)
# #     if os.path.isdir(cam_dir):
# #         ids = set([d for d in os.listdir(cam_dir) if os.path.isdir(os.path.join(cam_dir, d)) and d.isdigit()])
# #         print(f"{cam}: {len(ids)} IDs, max ID: {max(ids, default='None')}")
# #         valid_ids = ids if valid_ids is None else valid_ids.intersection(ids)
# #     else:
# #         print(f"Warning: Camera directory {cam_dir} does not exist")
# #         valid_ids = set() if valid_ids is None else valid_ids
# #
# # # 检查交集结果
# # if not valid_ids:
# #     raise ValueError("No valid IDs found in gallery cameras")
# # valid_ids = sorted(valid_ids)
# #
# # # 打印结果
# # print(f"Valid gallery IDs: {len(valid_ids)} IDs, max ID: {valid_ids[-1] if valid_ids else 'None'}")
# # print(f"First few IDs: {valid_ids[:5] if valid_ids else []}")
# #
# # # 写入新的 test_id1.txt 文件
# # output_file = os.path.join(root, 'exp/test_id1.txt')
# # os.makedirs(os.path.dirname(output_file), exist_ok=True)
# # with open(output_file, 'w') as f:
# #     f.write(','.join(str(int(id)) for id in valid_ids))
# # print(f"New file created: {output_file}")
#
# import torch
# from models.reid_model import ReIDModel
# from data.dataloader import SYSUDataset, custom_collate_fn
# from torch.utils.data import DataLoader
# import numpy as np
# from config import config
# from ptflops import get_model_complexity_info
# import pandas as pd
#
#
# def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
#     """Evaluate ReID performance on SYSU-MM01."""
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#     indices = np.argsort(distmat, axis=1)
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     all_cmc, all_AP = [], []
#     num_valid_q = 0
#     skipped_pids = []
#     for q_idx in range(num_q):
#         q_camid, q_pid = q_camids[q_idx], q_pids[q_idx]
#         order = indices[q_idx]
#         keep = q_camid != g_camids[order]
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             print(f"Warning: Query PID {q_pid} has no matches in gallery")
#             skipped_pids.append(q_pid)
#             continue
#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1
#         all_cmc.append(cmc[:max_rank])
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#         num_valid_q += 1
#     if num_valid_q == 0:
#         raise ValueError("No valid queries found")
#     all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     if skipped_pids:
#         print(f"Skipped {len(skipped_pids)} query PIDs: {sorted(set(skipped_pids))}")
#     return all_cmc, mAP
#
#
# def evaluate(model, k=3, g=4, K=5):
#     """Evaluate model with specified k, g, K."""
#     model.eval()
#     model.alignment.k = k
#     model.fusion.g = g
#     model.alignment.K = K
#
#     query_dataset = SYSUDataset(root=config.DATA_ROOT, mode="query")
#     query_loader = DataLoader(
#         query_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
#     )
#     gallery_dataset = SYSUDataset(root=config.DATA_ROOT, mode="gallery")
#     gallery_loader = DataLoader(
#         gallery_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
#     )
#
#     query_feats, query_labels, query_camids = [], [], []
#     gallery_feats, gallery_labels, gallery_camids = [], [], []
#     with torch.no_grad():
#         for i, (x_vis, x_ir, labels, camids) in enumerate(query_loader):
#             x_vis = x_vis.to(config.DEVICE) if x_vis is not None else None
#             x_ir = x_ir.to(config.DEVICE) if x_ir is not None else None
#             if x_vis is None and x_ir is None:
#                 print(f"Warning: Empty query batch {i}, skipping")
#                 continue
#             feats = model(x_vis, x_ir, training=False)
#             query_feats.append(feats.cpu())
#             query_labels.append(labels)
#             query_camids.append(camids)
#         for i, (x_vis, x_ir, labels, camids) in enumerate(gallery_loader):
#             x_vis = x_vis.to(config.DEVICE) if x_vis is not None else None
#             x_ir = x_ir.to(config.DEVICE) if x_ir is not None else None
#             if x_vis is None and x_ir is None:
#                 print(f"Warning: Empty gallery batch {i}, skipping")
#                 continue
#             feats = model(x_vis, x_ir, training=False)
#             gallery_feats.append(feats.cpu())
#             gallery_labels.append(labels)
#             gallery_camids.append(camids)
#
#     if not query_feats or not gallery_feats:
#         raise ValueError("No valid features extracted from query or gallery")
#
#     query_feats = torch.cat(query_feats)
#     query_labels = torch.cat(query_labels)
#     query_camids = torch.cat(query_camids)
#     gallery_feats = torch.cat(gallery_feats)
#     gallery_labels = torch.cat(gallery_labels)
#     gallery_camids = torch.cat(gallery_camids)
#
#     distmat = torch.cdist(query_feats, gallery_feats)
#     cmc, mAP = eval_sysu(
#         distmat.cpu().numpy(),
#         query_labels.cpu().numpy(),
#         gallery_labels.cpu().numpy(),
#         query_camids.cpu().numpy(),
#         gallery_camids.cpu().numpy()
#     )
#
#     # Compute FLOPs
#     flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
#     print(f"k={k}, g={g}, K={K} | Rank-1: {cmc[0]:.4f}, mAP: {mAP:.4f}, FLOPs: {flops / 1e9:.2f}G")
#     return cmc[0], mAP, flops / 1e9
#
#
# def ablation_study():
#     """Run ablation study for k, g, K."""
#     model = ReIDModel().to(config.DEVICE)
#     try:
#         model.load_state_dict(torch.load(config.MODEL_PATH))
#     except FileNotFoundError:
#         print(f"Error: Model checkpoint {config.MODEL_PATH} not found")
#         return []
#
#     # Baseline configuration
#     baseline_rank1, baseline_mAP, _ = evaluate(model, k=3, g=4, K=5)
#     print(f"Baseline: Rank-1: {baseline_rank1:.4f}, mAP: {baseline_mAP:.4f}")
#
#     configs = [
#         {'k': 3, 'g': 4, 'K': 5},  # Full model
#         {'k': 14 * 14, 'g': 4, 'K': 5},  # No sparse selection
#         {'k': 3, 'g': 1, 'K': 5},  # No grouped attention
#         {'k': 3, 'g': 4, 'K': 1},  # Fewer shots
#     ]
#     results = []
#     for cfg in configs:
#         rank1, mAP, flops = evaluate(model, **cfg)
#         mAP_gain = mAP - baseline_mAP
#         results.append({
#             'k': cfg['k'],
#             'g': cfg['g'],
#             'K': cfg['K'],
#             'Rank-1': rank1,
#             'mAP': mAP,
#             'mAP Gain': mAP_gain,
#             'FLOPs': flops
#         })
#
#     # Save results to CSV
#     df = pd.DataFrame(results)
#     df.to_csv("ablation_results.csv", index=False)
#     print("Ablation Study Results:")
#     print(df)
#     return results
#
#
# if __name__ == "__main__":
#     model = ReIDModel().to(config.DEVICE)
#     try:
#         model.load_state_dict(torch.load(config.MODEL_PATH))
#         evaluate(model)
#         ablation_study()
#     except FileNotFoundError:
#         print(f"Error: Model checkpoint {config.MODEL_PATH} not found. Please train the model first.")
#
import torch
from models.reid_model import ReIDModel
from data.dataloader import SYSUDataset, custom_collate_fn
from torch.utils.data import DataLoader
import numpy as np
from config import config
from ptflops import get_model_complexity_info
import pandas as pd

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluate ReID performance on SYSU-MM01."""
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
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1
    if num_valid_q == 0:
        raise ValueError("No valid queries found")
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    if skipped_pids:
        print(f"Skipped {len(skipped_pids)} query PIDs: {sorted(set(skipped_pids))}")
    return all_cmc, mAP

def evaluate(model, k=3, g=4, K=5, test_mode="all", trial=0):
    """Evaluate model with specified k, g, K."""
    model.eval()
    model.alignment.k = k
    model.fusion.g = g
    model.alignment.K = K

    query_dataset = SYSUDataset(root=config.DATA_ROOT, mode="query", test_mode=test_mode, trial=trial)
    query_loader = DataLoader(
        query_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
    )
    gallery_dataset = SYSUDataset(root=config.DATA_ROOT, mode="gallery", test_mode=test_mode, trial=trial)
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
    )

    query_feats, query_labels, query_camids = [], [], []
    gallery_feats, gallery_labels, gallery_camids = [], [], []
    with torch.no_grad():
        for i, (x_vis, x_ir, labels, camids) in enumerate(query_loader):
            x_vis = x_vis.to(config.DEVICE) if x_vis is not None else None
            x_ir = x_ir.to(config.DEVICE) if x_ir is not None else None
            if x_vis is None and x_ir is None:
                print(f"Warning: Empty query batch {i}, skipping")
                continue
            feats = model(x_vis, x_ir, training=False)
            query_feats.append(feats.cpu())
            query_labels.append(labels)
            query_camids.append(camids)
        for i, (x_vis, x_ir, labels, camids) in enumerate(gallery_loader):
            x_vis = x_vis.to(config.DEVICE) if x_vis is not None else None
            x_ir = x_ir.to(config.DEVICE) if x_ir is not None else None
            if x_vis is None and x_ir is None:
                print(f"Warning: Empty gallery batch {i}, skipping")
                continue
            feats = model(x_vis, x_ir, training=False)
            gallery_feats.append(feats.cpu())
            gallery_labels.append(labels)
            gallery_camids.append(camids)

    if not query_feats or not gallery_feats:
        raise ValueError("No valid features extracted from query or gallery")

    query_feats = torch.cat(query_feats)
    query_labels = torch.cat(query_labels)
    query_camids = torch.cat(query_camids)
    gallery_feats = torch.cat(gallery_feats)
    gallery_labels = torch.cat(gallery_labels)
    gallery_camids = torch.cat(gallery_camids)

    distmat = torch.cdist(query_feats, gallery_feats)
    cmc, mAP = eval_sysu(
        distmat.cpu().numpy(),
        query_labels.cpu().numpy(),
        gallery_labels.cpu().numpy(),
        query_camids.cpu().numpy(),
        gallery_camids.cpu().numpy()
    )

    # Compute FLOPs
    flops, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
    print(f"k={k}, g={g}, K={K} | Rank-1: {cmc[0]:.4f}, mAP: {mAP:.4f}, FLOPs: {flops / 1e9:.2f}G")
    return cmc[0], mAP, flops / 1e9

def ablation_study():
    """Run ablation study for k, g, K."""
    model = ReIDModel().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Model checkpoint {config.MODEL_PATH} not found")
        return []

    # Baseline configuration
    baseline_rank1, baseline_mAP, _ = evaluate(model, k=3, g=4, K=5)
    print(f"Baseline: Rank-1: {baseline_rank1:.4f}, mAP: {baseline_mAP:.4f}")

    configs = [
        {'k': 3, 'g': 4, 'K': 5},  # Full model
        {'k': 14 * 14, 'g': 4, 'K': 5},  # No sparse selection
        {'k': 3, 'g': 1, 'K': 5},  # No grouped attention
        {'k': 3, 'g': 4, 'K': 1},  # Fewer shots
    ]
    results = []
    for cfg in configs:
        rank1, mAP, flops = evaluate(model, **cfg)
        mAP_gain = mAP - baseline_mAP
        results.append({
            'k': cfg['k'],
            'g': cfg['g'],
            'K': cfg['K'],
            'Rank-1': rank1,
            'mAP': mAP,
            'mAP Gain': mAP_gain,
            'FLOPs': flops
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)
    print("Ablation Study Results:")
    print(df)
    return results

if __name__ == "__main__":
    model = ReIDModel().to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH))
        evaluate(model)
        ablation_study()
    except FileNotFoundError:
        print(f"Error: Model checkpoint {config.MODEL_PATH} not found. Please train the model first.")
