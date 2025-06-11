#
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import random
# class SYSUDataset(Dataset):
#     def __init__(self, root, mode="train", colorIndex=None, thermalIndex=None):
#         self.root = root
#         self.mode = mode
#         self.test_mode = test_mode  # 'all' 或 'indoor'
#         self.trial = trial
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),  # 转换为 [C, H, W]，归一化到 [0, 1]
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准归一化
#         ])
#         self.colorIndex = colorIndex
#         self.thermalIndex = thermalIndex
#         self.rgb_imgs = []
#         self.rgb_labels = []
#         self.ir_imgs = []
#         self.ir_labels = []
#         print(f"Initializing SYSUDataset with root: {self.root}")
#         self._load_data()
#
#     def _load_ids(self, file_path):
#         with open(file_path, 'r') as f:
#             ids = f.read().strip().split(',')
#             return ["%04d" % int(id) for id in ids if id.strip()]
#     def _load_data(self):
#         data_dir = self.root
#         print(f"Checking dataset directory: {data_dir}")
#         if not os.path.exists(data_dir):
#             raise FileNotFoundError(f"Dataset directory {data_dir} not found")
#
#         if self.mode == "train":
#             # 加载 RGB 数据
#             rgb_imgs_path = os.path.join(data_dir, "SYSU-MM01train_rgb_resized_img.npy")
#             rgb_labels_path = os.path.join(data_dir, "SYSU-MM01train_rgb_resized_label.npy")
#             if not os.path.exists(rgb_imgs_path) or not os.path.exists(rgb_labels_path):
#                 raise FileNotFoundError(f"RGB data files not found in {data_dir}")
#             self.rgb_imgs = np.load(rgb_imgs_path)
#             self.rgb_labels = np.load(rgb_labels_path)
#             print(f"Loaded RGB: {self.rgb_imgs.shape}, labels: {self.rgb_labels.shape}")
#
#             # 加载 IR 数据
#             ir_imgs_path = os.path.join(data_dir, "SYSU-MM01train_ir_resized_img.npy")
#             ir_labels_path = os.path.join(data_dir, "SYSU-MM01train_ir_resized_label.npy")
#             if not os.path.exists(ir_imgs_path) or not os.path.exists(ir_labels_path):
#                 raise FileNotFoundError(f"IR data files not found in {data_dir}")
#             self.ir_imgs = np.load(ir_imgs_path)
#             self.ir_labels = np.load(ir_labels_path)
#             print(f"Loaded IR: {self.ir_imgs.shape}, labels: {self.ir_labels.shape}")
#
#             # 按身份 ID 配对 RGB 和 IR 数据
#             if self.colorIndex is None or self.thermalIndex is None:
#                 # 获取唯一的身份 ID
#                 rgb_ids = np.unique(self.rgb_labels)
#                 ir_ids = np.unique(self.ir_labels)
#                 common_ids = np.intersect1d(rgb_ids, ir_ids)
#                 print(f"Common IDs: {len(common_ids)}")
#
#                 # 为每个身份 ID 选择配对的 RGB 和 IR 样本
#                 self.colorIndex = []
#                 self.thermalIndex = []
#                 for id in common_ids:
#                     rgb_idx = np.where(self.rgb_labels == id)[0]
#                     ir_idx = np.where(self.ir_labels == id)[0]
#                     # 随机选择一个样本（可扩展为选择多个）
#                     if len(rgb_idx) > 0 and len(ir_idx) > 0:
#                         self.colorIndex.append(np.random.choice(rgb_idx))
#                         self.thermalIndex.append(np.random.choice(ir_idx))
#
#                 self.colorIndex = np.array(self.colorIndex)
#                 self.thermalIndex = np.array(self.thermalIndex)
#                 print(f"Paired samples: {len(self.colorIndex)}")
#         elif self.mode == "query":
#             # 加载 query 数据（IR 图像）
#             ir_cameras = ['cam3', 'cam6']  # all 或 indoor 模式相同
#             file_path = os.path.join(data_dir, 'exp/test_id.txt')
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"Test ID file {file_path} not found")
#             ids = self._load_ids(file_path)
#             for id in sorted(ids):
#                 for cam in ir_cameras:
#                     img_dir = os.path.join(data_dir, cam, id)
#                     if os.path.isdir(img_dir):
#                         new_files = sorted(
#                             [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith('.jpg')])
#                         self.img_paths.extend(new_files)
#                         self.labels.extend([int(id)] * len(new_files))
#                         self.camids.extend([int(cam[-1])] * len(new_files))  # cam3 -> 3, cam6 -> 6
#         elif self.mode == "gallery":
#             # 加载 gallery 数据（RGB 图像）
#             random.seed(self.trial)
#             rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5'] if self.test_mode == 'all' else ['cam1', 'cam2']
#             file_path = os.path.join(data_dir, 'exp/test_id.txt')
#             if not os.path.exists(file_path):
#                 raise FileNotFoundError(f"Test ID file {file_path} not found")
#             ids = self._load_ids(file_path)
#             for id in sorted(ids):
#                 for cam in rgb_cameras:
#                     img_dir = os.path.join(data_dir, cam, id)
#                     if os.path.isdir(img_dir):
#                         new_files = sorted(
#                             [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith('.jpg')])
#                         if new_files:
#                             self.img_paths.append(random.choice(new_files))
#                             self.labels.append(int(id))
#                             self.camids.append(int(cam[-1]))  # cam1 -> 1, cam2 -> 2, etc.
#         else:
#             raise NotImplementedError(f"Mode {self.mode} not implemented")
#
#         if self.mode in ["query", "gallery"]:
#             self.labels = np.array(self.labels, dtype=np.int32)
#             self.camids = np.array(self.camids, dtype=np.int32)
#             print(f"{self.mode} dataset: {len(self.labels)} samples")
#
#     def __len__(self):
#         return len(self.colorIndex) if self.mode == "train" else len(self.labels)
#
#     def __getitem__(self, index):
#         if self.mode == "train":
#             # 获取 RGB 数据
#             rgb_img = self.rgb_imgs[self.colorIndex[index]]
#             rgb_label = self.rgb_labels[self.colorIndex[index]]
#             # 获取 IR 数据
#             ir_img = self.ir_imgs[self.thermalIndex[index]]
#             ir_label = self.ir_labels[self.thermalIndex[index]]
#             # 验证标签一致
#             if rgb_label != ir_label:
#                 raise ValueError(f"Label mismatch at index {index}: RGB {rgb_label}, IR {ir_label}")
#             # 应用变换
#             rgb_img = self.transform(rgb_img)
#             ir_img = self.transform(ir_img)
#             # 如果 IR 是单通道，转换为3通道
#             if ir_img.shape[0] == 1:
#                 ir_img = ir_img.repeat(3, 1, 1)
#             return rgb_img, ir_img, rgb_label, ir_label
#         else:  # query 或 gallery
#             img_path = self.img_paths[index]
#             label = self.labels[index]
#             camid = self.camids[index]
#             img = Image.open(img_path).resize((144, 288))  # 与训练一致
#             img = np.array(img) / 255.0
#             if self.mode == "query" and img.ndim == 2:  # IR 单通道
#                 img = img[..., np.newaxis]
#             elif img.ndim == 2:  # RGB 灰度转 3 通道
#                 img = np.stack([img] * 3, axis=-1)
#             img = self.transform(img)
#             if img.shape[0] == 1:  # 确保 IR 单通道转为 3 通道
#                 img = img.repeat(3, 1, 1)
#             # query: 返回 IR 图像，gallery: 返回 RGB 图像
#             if self.mode == "query":
#                 return None, img, label, camid
#             else:  # gallery
#                 return img, None, label, camid
#
# def custom_collate_fn(batch):
#     if batch[0][0] is not None or batch[0][1] is not None:  # train 模式
#         rgb_imgs, ir_imgs, rgb_labels, ir_labels = zip(*batch)
#         rgb_imgs = torch.stack(rgb_imgs)
#         ir_imgs = torch.stack(ir_imgs)
#         rgb_labels = torch.tensor(rgb_labels, dtype=torch.long)
#         ir_labels = torch.tensor(ir_labels, dtype=torch.long)
#         return rgb_imgs, ir_imgs, rgb_labels, ir_labels
#     else:  # query 或 gallery 模式
#         vis_imgs, ir_imgs, labels, camids = zip(*batch)
#         labels = torch.tensor(labels, dtype=torch.long)
#         camids = torch.tensor(camids, dtype=torch.long)
#         if vis_imgs[0] is not None:  # gallery
#             vis_imgs = torch.stack(vis_imgs)
#             return vis_imgs, None, labels, camids
#         else:  # query
#             ir_imgs = torch.stack(ir_imgs)
#             return None, ir_imgs, labels, camids

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random

class SYSUDataset(Dataset):
    def __init__(self, root, mode="train", colorIndex=None, thermalIndex=None, test_mode="all", trial=0):
        self.root = root
        self.mode = mode
        self.test_mode = test_mode  # 'all' 或 'indoor'
        self.trial = trial  # 用于 gallery 随机选择
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 [C, H, W]，归一化到 [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准归一化
        ])
        self.colorIndex = colorIndex
        self.thermalIndex = thermalIndex
        self.rgb_imgs = []
        self.rgb_labels = []
        self.ir_imgs = []
        self.ir_labels = []
        self.img_paths = []
        self.labels = []
        self.camids = []
        print(f"Initializing SYSUDataset with root: {self.root}, mode: {self.mode}, test_mode: {self.test_mode}")
        self._load_data()

    def _load_ids(self, file_path):
        with open(file_path, 'r') as f:
            ids = f.read().strip().split(',')
            return ["%04d" % int(id) for id in ids if id.strip()]

    def _load_data(self):
        data_dir = self.root
        print(f"Checking dataset directory: {data_dir}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Dataset directory {data_dir} not found")

        if self.mode == "train":
            # 加载 RGB 数据
            rgb_imgs_path = os.path.join(data_dir, "SYSU-MM01train_rgb_resized_img.npy")
            rgb_labels_path = os.path.join(data_dir, "SYSU-MM01train_rgb_resized_label.npy")
            if not os.path.exists(rgb_imgs_path) or not os.path.exists(rgb_labels_path):
                raise FileNotFoundError(f"RGB data files not found in {data_dir}")
            self.rgb_imgs = np.load(rgb_imgs_path)
            self.rgb_labels = np.load(rgb_labels_path)
            print(f"Loaded RGB: {self.rgb_imgs.shape}, labels: {self.rgb_labels.shape}")

            # 加载 IR 数据
            ir_imgs_path = os.path.join(data_dir, "SYSU-MM01train_ir_resized_img.npy")
            ir_labels_path = os.path.join(data_dir, "SYSU-MM01train_ir_resized_label.npy")
            if not os.path.exists(ir_imgs_path) or not os.path.exists(ir_labels_path):
                raise FileNotFoundError(f"IR data files not found in {data_dir}")
            self.ir_imgs = np.load(ir_imgs_path)
            self.ir_labels = np.load(ir_labels_path)
            print(f"Loaded IR: {self.ir_imgs.shape}, labels: {self.ir_labels.shape}")

            # 按身份 ID 配对 RGB 和 IR 数据
            if self.colorIndex is None or self.thermalIndex is None:
                rgb_ids = np.unique(self.rgb_labels)
                ir_ids = np.unique(self.ir_labels)
                common_ids = np.intersect1d(rgb_ids, ir_ids)
                print(f"Common IDs: {len(common_ids)}")

                self.colorIndex = []
                self.thermalIndex = []
                for id in common_ids:
                    rgb_idx = np.where(self.rgb_labels == id)[0]
                    ir_idx = np.where(self.ir_labels == id)[0]
                    if len(rgb_idx) > 0 and len(ir_idx) > 0:
                        self.colorIndex.append(np.random.choice(rgb_idx))
                        self.thermalIndex.append(np.random.choice(ir_idx))

                self.colorIndex = np.array(self.colorIndex)
                self.thermalIndex = np.array(self.thermalIndex)
                print(f"Paired samples: {len(self.colorIndex)}")
        elif self.mode == "query":
            # 加载 query 数据（IR 图像）
            ir_cameras = ['cam3', 'cam6']  # all 或 indoor 模式相同
            file_path = os.path.join(data_dir, 'exp/test_id.txt')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test ID file {file_path} not found")
            ids = self._load_ids(file_path)
            for id in sorted(ids):
                for cam in ir_cameras:
                    img_dir = os.path.join(data_dir, cam, id)
                    if os.path.isdir(img_dir):
                        new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith('.jpg')])
                        self.img_paths.extend(new_files)
                        self.labels.extend([int(id)] * len(new_files))
                        self.camids.extend([int(cam[-1])] * len(new_files))  # cam3 -> 3, cam6 -> 6
        elif self.mode == "gallery":
            # 加载 gallery 数据（RGB 图像）
            random.seed(self.trial)
            rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5'] if self.test_mode == 'all' else ['cam1', 'cam2']
            file_path = os.path.join(data_dir, 'exp/test_id.txt')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Test ID file {file_path} not found")
            ids = self._load_ids(file_path)
            for id in sorted(ids):
                for cam in rgb_cameras:
                    img_dir = os.path.join(data_dir, cam, id)
                    if os.path.isdir(img_dir):
                        new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith('.jpg')])
                        if new_files:
                            self.img_paths.append(random.choice(new_files))
                            self.labels.append(int(id))
                            self.camids.append(int(cam[-1]))  # cam1 -> 1, cam2 -> 2, etc.
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        if self.mode in ["query", "gallery"]:
            self.labels = np.array(self.labels, dtype=np.int32)
            self.camids = np.array(self.camids, dtype=np.int32)
            print(f"{self.mode} dataset: {len(self.labels)} samples")

    def __len__(self):
        return len(self.colorIndex) if self.mode == "train" else len(self.labels)

    def __getitem__(self, index):
        if self.mode == "train":
            # 获取 RGB 数据
            rgb_img = self.rgb_imgs[self.colorIndex[index]]
            rgb_label = self.rgb_labels[self.colorIndex[index]]
            # 获取 IR 数据
            ir_img = self.ir_imgs[self.thermalIndex[index]]
            ir_label = self.ir_labels[self.thermalIndex[index]]
            # 验证标签一致
            if rgb_label != ir_label:
                raise ValueError(f"Label mismatch at index {index}: RGB {rgb_label}, IR {ir_label}")
            # 应用变换
            rgb_img = self.transform(rgb_img)
            ir_img = self.transform(ir_img)
            # 如果 IR 是单通道，转换为3通道
            if ir_img.shape[0] == 1:
                ir_img = ir_img.repeat(3, 1, 1)
            return rgb_img, ir_img, rgb_label, ir_label
        else:  # query 或 gallery
            img_path = self.img_paths[index]
            label = self.labels[index]
            camid = self.camids[index]
            img = Image.open(img_path).resize((144, 288))  # 与训练一致
            img = np.array(img) / 255.0
            if self.mode == "query" and img.ndim == 2:  # IR 单通道
                img = img[..., np.newaxis]
            elif img.ndim == 2:  # RGB 灰度转 3 通道
                img = np.stack([img] * 3, axis=-1)
            img = self.transform(img)
            if img.shape[0] == 1:  # 确保 IR 单通道转为 3 通道
                img = img.repeat(3, 1, 1)
            # query: 返回 IR 图像，gallery: 返回 RGB 图像
            if self.mode == "query":
                return None, img, label, camid
            else:  # gallery
                return img, None, label, camid

def custom_collate_fn(batch):
    if batch[0][0] is not None and batch[0][1] is not None:  # train mode
        rgb_imgs, ir_imgs, rgb_labels, ir_labels = zip(*batch)
        rgb_imgs = torch.stack([img for img in rgb_imgs if img is not None])
        ir_imgs = torch.stack([img for img in ir_imgs if img is not None])
        rgb_labels = torch.tensor(rgb_labels, dtype=torch.long)
        ir_labels = torch.tensor(ir_labels, dtype=torch.long)
        return rgb_imgs, ir_imgs, rgb_labels, ir_labels
    else:  # query or gallery mode
        vis_imgs, ir_imgs, labels, camids = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
        camids = torch.tensor(camids, dtype=torch.long)
        if batch[0][0] is not None:  # gallery mode
            vis_imgs = torch.stack([img for img in vis_imgs if img is not None])
            return vis_imgs, None, labels, camids
        else:  # query mode
            ir_imgs = torch.stack([img for img in ir_imgs if img is not None])
            return None, ir_imgs, labels, camids

