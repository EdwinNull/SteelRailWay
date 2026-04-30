#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立评估脚本：从已有 best ckpt 加载并在测试集上跑 AUROC。

用途：
    - 训练崩在最后评估阶段时，可用本脚本接力评估，避免重新训练
    - 修改测试集后只跑评估
    - 比较不同 epoch 的 ckpt

用法：
    python scripts/eval_from_ckpt.py \
        --ckpt outputs/rail/20260501_xxxxx_cam1_xxx/best_cam1.pth \
        --train_root /data1/Leaddo_data/20260327-resize512 \
        --test_root  ./rail_mvtec_gt_test \
        --view_id 1
"""

# >>> path-bootstrap >>>
import os, sys
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
# <<< path-bootstrap <<<

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets.rail_dataset import RailDualModalDataset
from models.trd.encoder import ResNet50Encoder
from models.trd.decoder import ResNet50DualModalDecoder
from eval.eval_utils import cal_anomaly_map
from sklearn.metrics import roc_auc_score


def safe_load_ckpt(path, device):
    """兼容 PyTorch 2.6+ 的 weights_only 默认 True：显式关掉。
    本工程的 ckpt 由自己脚本生成，来源可信。
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_prefix(sd):
    """去掉 torch.compile 包装后的 _orig_mod. 前缀"""
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


@torch.no_grad()
def evaluate(teacher_rgb, teacher_depth, student_rgb, student_depth,
             test_loader, device):
    """与 train_trd_rail.py 中的 evaluate 一致：patch 分数 max 聚合到原图。"""
    student_rgb.eval()
    student_depth.eval()

    img_scores = {}
    img_labels = {}

    for data in test_loader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        feat_t_rgb = teacher_rgb(rgb)
        feat_t_depth = teacher_depth(depth)
        proj_d, proj_d_amply, feat_s_rgb, feat_s_rgb_am = student_rgb(feat_t_rgb, feat_t_depth)
        proj_r, proj_r_amply, feat_s_depth, feat_s_depth_am = student_depth(feat_t_depth, feat_t_rgb)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode='mul')
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode='mul')
        amap = amap_rgb + amap_depth

        if amap.ndim == 3:
            scores = amap.reshape(amap.shape[0], -1).max(axis=1)
        else:
            scores = np.array([amap.max()])

        for score, label, fid in zip(scores, labels, frame_ids):
            if fid not in img_scores:
                img_scores[fid] = []
                img_labels[fid] = int(label)
            img_scores[fid].append(float(score))

    image_scores = []
    image_labels = []
    for fid in sorted(img_scores.keys()):
        image_scores.append(max(img_scores[fid]))
        image_labels.append(img_labels[fid])

    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)

    if len(np.unique(image_labels)) < 2:
        print("Warning: only one class in test set, AUROC = N/A")
        return None, image_scores, image_labels

    auroc = roc_auc_score(image_labels, image_scores)
    return auroc, image_scores, image_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="best_camN.pth 路径")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--view_id", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore")
    # 测试 patch 配置（与训练脚本默认一致）
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading ckpt: {args.ckpt}")

    # ---- 1. 构建测试集 ----
    test_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="test",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ---- 2. 构建模型并加载权重 ----
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in teacher_rgb.parameters(): p.requires_grad = False
    for p in teacher_depth.parameters(): p.requires_grad = False

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device).eval()

    ckpt = safe_load_ckpt(args.ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt['student_rgb']))
    student_depth.load_state_dict(strip_prefix(ckpt['student_depth']))
    print(f"Loaded epoch={ckpt.get('epoch', '?')}, "
          f"best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f}")

    # ---- 3. 评估 ----
    auroc, scores, labels = evaluate(
        teacher_rgb, teacher_depth, student_rgb, student_depth,
        test_loader, device,
    )

    print("\n" + "=" * 60)
    print(f"  Cam{args.view_id} Test Result")
    print("=" * 60)
    if auroc is None:
        print(f"  AUROC: N/A (single-class test set)")
    else:
        print(f"  AUROC: {auroc:.4f}")
    print(f"  #images: {len(scores)}")
    print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  #abnormal: {(labels == 1).sum()}, #normal: {(labels == 0).sum()}")

    # 把分数落盘，便于后续画分布图
    out_csv = args.ckpt.replace(".pth", "_test_scores.csv")
    with open(out_csv, "w") as f:
        f.write("rank,score,label\n")
        order = np.argsort(-scores)  # 从高到低
        for rank, i in enumerate(order):
            f.write(f"{rank},{scores[i]:.6f},{labels[i]}\n")
    print(f"  scores saved to: {out_csv}")


if __name__ == "__main__":
    main()
