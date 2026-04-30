#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据划分功能

验证 train/val/test 的 8:1:1 划分是否正确
"""

import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from datasets.rail_dataset import RailDualModalDataset


def test_data_split():
    """测试数据划分"""
    print("="*60)
    print("Testing Train/Val/Test Split (8:1:1)")
    print("="*60)

    # 配置参数
    train_root = "/data1/Leaddo_data/20260327"  # 服务器路径
    test_root = "/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test"
    view_id = 1
    train_sample_ratio = 0.1  # 使用 10% 数据
    train_val_test_split = [0.8, 0.1, 0.1]

    print(f"\nConfiguration:")
    print(f"  - View ID: {view_id}")
    print(f"  - Sample Ratio: {train_sample_ratio}")
    print(f"  - Split Ratio: {train_val_test_split}")

    # 创建训练集
    print("\n" + "-"*60)
    print("Loading Train Dataset...")
    print("-"*60)
    train_dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="train",
        train_sample_ratio=train_sample_ratio,
        train_val_test_split=train_val_test_split,
        preload=False,  # 不预加载，只测试划分
    )

    # 创建验证集
    print("\n" + "-"*60)
    print("Loading Val Dataset...")
    print("-"*60)
    val_dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="val",
        train_sample_ratio=train_sample_ratio,
        train_val_test_split=train_val_test_split,
        preload=False,
    )

    # 创建测试集
    print("\n" + "-"*60)
    print("Loading Test Dataset...")
    print("-"*60)
    test_dataset = RailDualModalDataset(
        train_root=train_root,
        test_root=test_root,
        view_id=view_id,
        split="test",
        preload=False,
    )

    # 统计信息
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    # 考虑 patch 模式
    if train_dataset.use_patch:
        num_patches = train_dataset.num_patches
        train_images = train_size // num_patches
        val_images = val_size // num_patches
        test_images = test_size // num_patches

        print(f"\nPatch Mode: {num_patches} patches per image")
        print(f"\nImages:")
        print(f"  Train: {train_images} images ({train_size} patches)")
        print(f"  Val:   {val_images} images ({val_size} patches)")
        print(f"  Test:  {test_images} images ({test_size} patches)")

        total_images = train_images + val_images
        train_ratio = train_images / total_images if total_images > 0 else 0
        val_ratio = val_images / total_images if total_images > 0 else 0

        print(f"\nActual Split Ratio (Train+Val):")
        print(f"  Train: {train_ratio:.2%}")
        print(f"  Val:   {val_ratio:.2%}")
    else:
        print(f"\nNo Patch Mode")
        print(f"  Train: {train_size} samples")
        print(f"  Val:   {val_size} samples")
        print(f"  Test:  {test_size} samples")

        total_samples = train_size + val_size
        train_ratio = train_size / total_samples if total_samples > 0 else 0
        val_ratio = val_size / total_samples if total_samples > 0 else 0

        print(f"\nActual Split Ratio (Train+Val):")
        print(f"  Train: {train_ratio:.2%}")
        print(f"  Val:   {val_ratio:.2%}")

    # 检查标签分布
    print("\n" + "="*60)
    print("Label Distribution")
    print("="*60)

    train_labels = [train_dataset.samples[i]["label"] for i in range(len(train_dataset.samples))]
    val_labels = [val_dataset.samples[i]["label"] for i in range(len(val_dataset.samples))]
    test_labels = [test_dataset.samples[i]["label"] for i in range(len(test_dataset.samples))]

    print(f"\nTrain: Normal={train_labels.count(0)}, Abnormal={train_labels.count(1)}")
    print(f"Val:   Normal={val_labels.count(0)}, Abnormal={val_labels.count(1)}")
    print(f"Test:  Normal={test_labels.count(0)}, Abnormal={test_labels.count(1)}")

    # 验证结果
    print("\n" + "="*60)
    print("Validation")
    print("="*60)

    checks = []

    # 检查 1: Train 和 Val 只有正常样本
    check1 = train_labels.count(1) == 0 and val_labels.count(1) == 0
    checks.append(("Train/Val only have normal samples", check1))

    # 检查 2: Test 有异常样本
    check2 = test_labels.count(1) > 0
    checks.append(("Test has abnormal samples", check2))

    # 检查 3: 划分比例接近 8:1:1
    expected_train_ratio = train_val_test_split[0]
    expected_val_ratio = train_val_test_split[1]
    ratio_tolerance = 0.05  # 5% 容差

    check3 = abs(train_ratio - expected_train_ratio) < ratio_tolerance
    check4 = abs(val_ratio - expected_val_ratio) < ratio_tolerance
    checks.append((f"Train ratio ~{expected_train_ratio:.0%}", check3))
    checks.append((f"Val ratio ~{expected_val_ratio:.0%}", check4))

    # 打印检查结果
    print()
    all_passed = True
    for desc, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed!")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    try:
        success = test_data_split()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
