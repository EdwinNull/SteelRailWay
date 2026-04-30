# -*- coding: utf-8 -*-
"""
钢轨双模态数据集（Stage A：单视角版本）

数据组织：
训练集：data_20260327/
    Cam{1-8}/
        rgb/    *.jpg (正常样本, ~903×6000)
        depth/  *.tiff

测试集：rail_mvtec_gt_test/
    rail_mvtec/
        cam{1-6}/
            test/
                good/   *.jpg (正常样本, ~1739×6000)
                broken/ *.jpg (异常样本)
            ground_truth/
                broken/ *.png (像素级 GT mask)
    rail_mvtec_depth/
        cam{1-6}/
            test/
                good/   *.tiff
                broken/ *.tiff
            ground_truth/
                broken/ *.png

Patch 策略：
    垂直方向按 stride 滑动，水平方向居中裁剪 patch_size 宽，
    保证无论原图宽度如何，送入模型的都是正方形 patch。
"""

import os
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RailDualModalDataset(Dataset):
    """
    单视角钢轨双模态数据集（RGB + Depth）。
    支持 Patch 滑窗切分：垂直滑动 + 水平居中裁剪为正方形 patch。

    Args:
        train_root:  训练集根目录（data_20260327）
        test_root:   测试集根目录（rail_mvtec_gt_test）
        view_id:     当前视角编号（1..8，注意训练集是 Cam1-8，测试集是 cam1-6）
        split:       'train' | 'val' | 'test'
        img_size:    输入网络的边长（默认 256）
        depth_norm:  深度归一化方式：'zscore' | 'minmax' | 'log'
        use_patch:   是否使用 patch 切分（默认 True）
        patch_size:  patch 边长（默认 900）
        patch_stride: patch 滑动步长（默认 850，重叠 50 像素）
        train_sample_ratio: 训练集采样比例（0-1，默认 1.0 使用全部数据）
        train_sample_num: 训练集采样数量（优先级高于 ratio，None 表示使用 ratio）
        random_seed: 随机采样种子（默认 42）
        preload: 是否预加载所有图像到内存（默认 False）
        preload_workers: 预加载时的并行进程数（默认 16）
        train_val_test_split: 训练/验证/测试集划分比例（默认 [0.8, 0.1, 0.1]）
    """

    def __init__(
        self,
        train_root: str,
        test_root: str,
        view_id: int,
        split: str = "train",
        img_size: int = 256,
        depth_norm: str = "zscore",
        use_patch: bool = True,
        patch_size: int = 900,
        patch_stride: int = 850,
        train_sample_ratio: float = 1.0,
        train_sample_num: Optional[int] = None,
        random_seed: int = 42,
        preload: bool = False,
        preload_workers: int = 16,
        train_val_test_split: List[float] = None,
    ):
        super().__init__()
        self.train_root = train_root
        self.test_root = test_root
        self.view_id = view_id
        self.split = split
        self.img_size = img_size
        self.depth_norm = depth_norm
        self.use_patch = use_patch
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.train_sample_ratio = train_sample_ratio
        self.train_sample_num = train_sample_num
        self.random_seed = random_seed
        self.preload = preload
        self.preload_workers = preload_workers
        self.train_val_test_split = train_val_test_split or [0.8, 0.1, 0.1]

        # 预加载缓存
        self.rgb_cache: Dict[str, np.ndarray] = {}
        self.depth_cache: Dict[str, np.ndarray] = {}
        self.gt_cache: Dict[str, np.ndarray] = {}
        self.depth_stats: Dict[str, tuple] = {}  # path → (mean, std) 预计算

        # 扫描磁盘 → 拿到样本清单
        self.samples: List[Dict] = self._scan_files()

        # 训练集采样（如果需要）
        if self.split in ["train", "val"] and len(self.samples) > 0:
            self._apply_sampling()

        # 预加载数据到内存（如果启用）
        if self.preload:
            self._preload_data()

        # 动态检测图像尺寸（从第一个样本读取）
        if self.use_patch and len(self.samples) > 0:
            self.img_height, self.img_width = self._detect_image_size()
            self.num_patches = (self.img_height - self.patch_size) // self.patch_stride + 1
            print(f"[RailDataset] Detected image: {self.img_height}×{self.img_width}, "
                  f"{self.num_patches} patches per image")
        else:
            self.img_height = None
            self.img_width = None
            self.num_patches = 1

        # RGB 走 ImageNet 标准化（关闭 antialias 以加速大比例下采样）
        rgb_resize = (img_size, img_size)
        self.intensity_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(rgb_resize, antialias=False),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ------------------------------------------------------------------ #
    # 文件扫描                                                            #
    # ------------------------------------------------------------------ #
    def _scan_files(self) -> List[Dict]:
        """扫描训练集或测试集"""
        samples = []

        if self.split in ["train", "val"]:
            # 训练集和验证集：从 data_20260327/Cam{view_id}/ 中划分
            cam_dir = os.path.join(self.train_root, f"Cam{self.view_id}")
            rgb_dir = os.path.join(cam_dir, "rgb")
            depth_dir = os.path.join(cam_dir, "depth")

            if not os.path.exists(rgb_dir):
                raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])

            for rgb_file in rgb_files:
                frame_id = rgb_file.replace(".jpg", "")
                depth_file = frame_id + ".tiff"

                rgb_path = os.path.join(rgb_dir, rgb_file)
                depth_path = os.path.join(depth_dir, depth_file)

                if not os.path.exists(depth_path):
                    continue

                samples.append({
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "label": 0,  # 训练集和验证集全是正常样本
                    "gt_path": None,
                    "view_id": self.view_id,
                    "frame_id": frame_id,
                })

        else:  # test
            # 测试集：rail_mvtec_gt_test/rail_mvtec/cam{view_id}/
            # 注意：测试集只有 cam1-6
            if self.view_id > 6:
                print(f"Warning: Test set only has cam1-6, view_id={self.view_id} has no test data")
                return samples

            cam_dir = os.path.join(self.test_root, "rail_mvtec", f"cam{self.view_id}")
            depth_cam_dir = os.path.join(self.test_root, "rail_mvtec_depth", f"cam{self.view_id}")

            # 扫描 good 样本
            good_rgb_dir = os.path.join(cam_dir, "test", "good")
            good_depth_dir = os.path.join(depth_cam_dir, "test", "good")

            if os.path.exists(good_rgb_dir):
                for rgb_file in os.listdir(good_rgb_dir):
                    if not rgb_file.endswith(".jpg"):
                        continue
                    frame_id = rgb_file.replace(".jpg", "")
                    depth_file = frame_id + ".tiff"

                    rgb_path = os.path.join(good_rgb_dir, rgb_file)
                    depth_path = os.path.join(good_depth_dir, depth_file)

                    if not os.path.exists(depth_path):
                        continue

                    samples.append({
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "label": 0,
                        "gt_path": None,
                        "view_id": self.view_id,
                        "frame_id": frame_id,
                    })

            # 扫描 broken 样本
            broken_rgb_dir = os.path.join(cam_dir, "test", "broken")
            broken_depth_dir = os.path.join(depth_cam_dir, "test", "broken")
            gt_dir = os.path.join(cam_dir, "ground_truth", "broken")

            if os.path.exists(broken_rgb_dir):
                for rgb_file in os.listdir(broken_rgb_dir):
                    if not rgb_file.endswith(".jpg"):
                        continue
                    frame_id = rgb_file.replace(".jpg", "")
                    depth_file = frame_id + ".tiff"
                    gt_file = frame_id + ".png"

                    rgb_path = os.path.join(broken_rgb_dir, rgb_file)
                    depth_path = os.path.join(broken_depth_dir, depth_file)
                    gt_path = os.path.join(gt_dir, gt_file)

                    if not os.path.exists(depth_path):
                        continue

                    samples.append({
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "label": 1,  # 异常样本
                        "gt_path": gt_path if os.path.exists(gt_path) else None,
                        "view_id": self.view_id,
                        "frame_id": frame_id,
                    })

        return samples

    def _apply_sampling(self):
        """对训练集/验证集进行采样和划分"""
        total = len(self.samples)

        # 先进行数据采样（如果需要）
        if self.train_sample_num is not None:
            target_num = min(self.train_sample_num, total)
        else:
            target_num = int(total * self.train_sample_ratio)

        if target_num < total:
            np.random.seed(self.random_seed)
            indices = np.random.choice(total, target_num, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"[RailDataset] Sampled {target_num}/{total} images "
                  f"(ratio={target_num/total:.2%})")
        else:
            print(f"[RailDataset] Using all {total} images")

        # 然后进行 train/val 划分
        total_sampled = len(self.samples)
        train_ratio, val_ratio, test_ratio = self.train_val_test_split

        # 计算划分点
        train_end = int(total_sampled * train_ratio)
        val_end = train_end + int(total_sampled * val_ratio)

        # 打乱数据（使用固定种子保证可复现）
        np.random.seed(self.random_seed)
        indices = np.random.permutation(total_sampled)

        if self.split == "train":
            selected_indices = indices[:train_end]
            self.samples = [self.samples[i] for i in sorted(selected_indices)]
            print(f"[RailDataset] Train split: {len(self.samples)} images")
        elif self.split == "val":
            selected_indices = indices[train_end:val_end]
            self.samples = [self.samples[i] for i in sorted(selected_indices)]
            print(f"[RailDataset] Val split: {len(self.samples)} images")

    def _preload_data(self):
        """预加载所有图像到内存（多进程并行）"""
        import time
        from multiprocessing import Pool, Manager
        from tqdm import tqdm

        print(f"\n[RailDataset] Preloading {len(self.samples)} images to memory...")
        print(f"[RailDataset] Using {self.preload_workers} workers")
        start_time = time.time()

        # 使用 Manager 创建共享字典（用于多进程）
        manager = Manager()
        shared_rgb_cache = manager.dict()
        shared_depth_cache = manager.dict()
        shared_gt_cache = manager.dict()

        # 准备加载任务
        tasks = []
        for sample in self.samples:
            tasks.append({
                'rgb_path': sample['rgb_path'],
                'depth_path': sample['depth_path'],
                'gt_path': sample.get('gt_path'),
            })

        # 多进程并行加载
        with Pool(processes=self.preload_workers) as pool:
            results = list(tqdm(
                pool.imap(self._load_single_image, tasks),
                total=len(tasks),
                desc="Preloading",
                ncols=80
            ))

        # 将结果存入缓存，并预计算 depth 统计量
        for result in results:
            if result is not None:
                rgb_path, depth_path, gt_path, rgb_data, depth_data, gt_data = result
                self.rgb_cache[rgb_path] = rgb_data
                self.depth_cache[depth_path] = depth_data
                if gt_path is not None and gt_data is not None:
                    self.gt_cache[gt_path] = gt_data
                # 预计算全图 depth zscore 统计量（避免每个 patch 重算）
                if self.depth_norm == "zscore":
                    valid = depth_data[depth_data > 0]
                    if valid.size > 0:
                        self.depth_stats[depth_path] = (float(valid.mean()), float(valid.std()))
                    else:
                        self.depth_stats[depth_path] = (0.0, 1.0)

        elapsed = time.time() - start_time
        total_size_mb = sum(img.nbytes for img in self.rgb_cache.values()) / 1024 / 1024
        total_size_mb += sum(img.nbytes for img in self.depth_cache.values()) / 1024 / 1024

        print(f"[RailDataset] Preloading completed in {elapsed:.1f}s")
        print(f"[RailDataset] Total memory usage: {total_size_mb:.1f} MB")
        print(f"[RailDataset] Average speed: {len(self.samples)/elapsed:.1f} images/s\n")

    @staticmethod
    def _load_single_image(task):
        """加载单张图像（静态方法，用于多进程）"""
        try:
            rgb_path = task['rgb_path']
            depth_path = task['depth_path']
            gt_path = task.get('gt_path')

            # 加载 RGB
            rgb_data = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_data is None:
                return None
            rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)

            # 加载 Depth
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_data is None:
                return None

            # 加载 GT（如果有）
            gt_data = None
            if gt_path is not None and os.path.exists(gt_path):
                gt_data = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            return (rgb_path, depth_path, gt_path, rgb_data, depth_data, gt_data)
        except Exception as e:
            print(f"Error loading {task.get('rgb_path', 'unknown')}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # 模态加载                                                            #
    # ------------------------------------------------------------------ #
    def _detect_image_size(self) -> tuple:
        """从第一个样本动态检测图像的实际宽高 (height, width)"""
        s = self.samples[0]
        # 优先从缓存读取
        if s["rgb_path"] in self.rgb_cache:
            h, w = self.rgb_cache[s["rgb_path"]].shape[:2]
        else:
            tmp = cv2.imread(s["rgb_path"], cv2.IMREAD_COLOR)
            if tmp is not None:
                h, w = tmp.shape[:2]
            else:
                raise FileNotFoundError(f"Cannot read first image: {s['rgb_path']}")
        return h, w

    def _extract_patch(self, img: np.ndarray, patch_idx: int) -> np.ndarray:
        """提取第 patch_idx 个 patch（垂直滑动 + 水平居中裁剪为正方形）

        - 垂直方向：从 y_start 滑动 patch_size
        - 水平方向：居中裁剪 patch_size（当图像宽度 > patch_size）
                     或保持原宽（当图像宽度 <= patch_size）
        """
        y_start = patch_idx * self.patch_stride
        y_end = y_start + self.patch_size
        if y_end > img.shape[0]:
            y_end = img.shape[0]
            y_start = y_end - self.patch_size

        # 水平方向：居中裁剪为 patch_size 宽（保持正方形输入）
        w = img.shape[1]
        if w > self.patch_size:
            x_start = (w - self.patch_size) // 2
            x_end = x_start + self.patch_size
        else:
            x_start = 0
            x_end = w

        if img.ndim == 3:
            return img[y_start:y_end, x_start:x_end, :]
        else:
            return img[y_start:y_end, x_start:x_end]

    def _load_rgb(self, path: str, patch_idx: int = 0) -> torch.Tensor:
        """加载 RGB 图像"""
        # 从缓存加载（如果已预加载）
        if path in self.rgb_cache:
            img = self.rgb_cache[path].copy()
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 如果使用 patch，提取对应的 patch
        if self.use_patch:
            img = self._extract_patch(img, patch_idx)

        return self.intensity_tf(img)

    def _load_depth(self, path: str, patch_idx: int = 0) -> torch.Tensor:
        """加载深度图"""
        # 从缓存加载（如果已预加载）
        if path in self.depth_cache:
            depth = self.depth_cache[path].copy().astype(np.float32)
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth is None:
                raise FileNotFoundError(path)

        # 如果使用 patch，提取对应的 patch
        if self.use_patch:
            depth = self._extract_patch(depth, patch_idx)

        # 深度归一化（预加载时已计算全图统计量，直接应用）
        if self.depth_norm == "zscore":
            if path in self.depth_stats:
                d_mean, d_std = self.depth_stats[path]
                depth = (depth - d_mean) / (d_std + 1e-6)
            else:
                valid = depth[depth > 0]
                if valid.size > 0:
                    d_mean, d_std = float(valid.mean()), float(valid.std())
                    self.depth_stats[path] = (d_mean, d_std)
                    depth = (depth - d_mean) / (d_std + 1e-6)
        elif self.depth_norm == "minmax":
            d_min, d_max = float(depth.min()), float(depth.max())
            depth = (depth - d_min) / (d_max - d_min + 1e-6)
        elif self.depth_norm == "log":
            depth = np.log1p(np.clip(depth, 0, None))
        else:
            raise ValueError(f"unknown depth_norm: {self.depth_norm}")

        depth = cv2.resize(depth, (self.img_size, self.img_size))
        # 复制成 3 通道
        depth = np.stack([depth, depth, depth], axis=0)
        return torch.from_numpy(depth).float()

    # ------------------------------------------------------------------ #
    # Dataset 接口                                                        #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples) * self.num_patches

    def __getitem__(self, idx: int) -> Dict:
        # 计算是哪张图的哪个 patch
        img_idx = idx // self.num_patches
        patch_idx = idx % self.num_patches

        s = self.samples[img_idx]
        item = {
            "intensity": self._load_rgb(s["rgb_path"], patch_idx),
            "depth": self._load_depth(s["depth_path"], patch_idx),
            "label": int(s["label"]),
            "view_id": int(s.get("view_id", self.view_id)),
            "frame_id": s.get("frame_id", ""),
            "patch_idx": patch_idx,
        }

        gt_path: Optional[str] = s.get("gt_path")
        if gt_path is not None and os.path.exists(gt_path):
            # 从缓存加载（如果已预加载）
            if gt_path in self.gt_cache:
                gt = self.gt_cache[gt_path].copy()
            else:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if self.use_patch:
                gt = self._extract_patch(gt, patch_idx)
            gt = cv2.resize(
                gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
            )
            item["gt"] = torch.from_numpy((gt > 0).astype(np.float32))
        else:
            item["gt"] = torch.zeros(self.img_size, self.img_size)

        return item
