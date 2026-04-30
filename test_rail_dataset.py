# -*- coding: utf-8 -*-
"""
测试钢轨数据集加载是否正确
"""

import os
import sys
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
from datasets.rail_dataset import RailDualModalDataset
import matplotlib.pyplot as plt
import numpy as np


def test_dataset_loading():
    """测试数据集加载"""
    print("="*60)
    print("Testing Rail Dataset Loading")
    print("="*60)

    train_root = "G:/SteelRailWay/data_20260327"
    test_root = "G:/SteelRailWay/rail_mvtec_gt_test"
    view_id = 1

    # 测试训练集
    print("\n[1] Testing train dataset...")
    train_dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="train",
        img_size=256,
        depth_norm="zscore",
        use_patch=True,
        patch_size=900,
        patch_stride=850,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Number of images: {len(train_dataset.samples)}")
    print(f"Patches per image: {train_dataset.num_patches}")

    # 测试测试集
    print("\n[2] Testing test dataset...")
    test_dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="test",
        img_size=256,
        depth_norm="zscore",
        use_patch=True,
        patch_size=900,
        patch_stride=850,
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of images: {len(test_dataset.samples)}")
    print(f"Patches per image: {test_dataset.num_patches}")

    # 测试加载一个样本
    print("\n[3] Testing sample loading...")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Intensity shape: {sample['intensity'].shape}")
        print(f"Depth shape: {sample['depth'].shape}")
        print(f"Label: {sample['label']}")
        print(f"View ID: {sample['view_id']}")
        print(f"Patch idx: {sample['patch_idx']}")
        print(f"Frame ID: {sample['frame_id']}")

        # 检查数值范围
        print(f"\nIntensity range: [{sample['intensity'].min():.3f}, {sample['intensity'].max():.3f}]")
        print(f"Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]")

        # 检查是否有 NaN 或 Inf
        assert not torch.isnan(sample['intensity']).any(), "Intensity contains NaN!"
        assert not torch.isinf(sample['intensity']).any(), "Intensity contains Inf!"
        assert not torch.isnan(sample['depth']).any(), "Depth contains NaN!"
        assert not torch.isinf(sample['depth']).any(), "Depth contains Inf!"
        print("✓ No NaN or Inf detected")

    # 测试 DataLoader
    print("\n[4] Testing DataLoader...")
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Windows 下先用 0 测试
    )

    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Intensity: {batch['intensity'].shape}")
        print(f"  Depth: {batch['depth'].shape}")
        print(f"  Labels: {batch['label']}")
        if batch_idx >= 2:  # 只测试前 3 个 batch
            break

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


def visualize_samples():
    """可视化一些样本"""
    print("\n[5] Visualizing samples...")

    train_root = "G:/SteelRailWay/data_20260327"
    test_root = "G:/SteelRailWay/rail_mvtec_gt_test"
    view_id = 1

    dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="train",
        img_size=256,
        depth_norm="zscore",
        use_patch=True,
        patch_size=900,
        patch_stride=850,
    )

    # 可视化前 4 个 patch
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(min(4, len(dataset))):
        sample = dataset[i]

        # Intensity (需要反归一化)
        intensity = sample['intensity'].numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        intensity = intensity * std + mean
        intensity = np.clip(intensity, 0, 1)
        intensity = np.transpose(intensity, (1, 2, 0))

        # Depth (取第一个通道)
        depth = sample['depth'][0].numpy()

        axes[0, i].imshow(intensity)
        axes[0, i].set_title(f"Patch {i} - RGB")
        axes[0, i].axis('off')

        axes[1, i].imshow(depth, cmap='jet')
        axes[1, i].set_title(f"Patch {i} - Depth")
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = "rail_dataset_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    try:
        test_dataset_loading()
        visualize_samples()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
