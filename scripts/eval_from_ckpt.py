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
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from datasets.rail_dataset import RailDualModalDataset
from models.trd.encoder import ResNet50Encoder
from models.trd.decoder import ResNet50DualModalDecoder
from eval.eval_utils import cal_anomaly_map
from rail_peft import DepthAffinePEFT, DepthEncoderWithPEFT
from sklearn.metrics import roc_auc_score


try:
    from torch.amp import autocast as _autocast_new

    def autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        return _autocast_new(device_type, enabled=enabled, dtype=dtype)

except ImportError:
    try:
        from torch.cuda.amp import autocast as _autocast_old

        def autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            return _autocast_old(enabled=enabled, dtype=dtype)

    except ImportError:
        autocast = None


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


def load_depth_peft(path, device):
    """Load a DepthAffinePEFT checkpoint saved either as a raw state_dict or payload."""
    payload = safe_load_ckpt(path, device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    else:
        state_dict = payload
        metadata = {}

    if not isinstance(state_dict, dict):
        raise TypeError(f"Invalid depth PEFT checkpoint: {path}")

    state_dict = {
        k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", ""): v
        for k, v in state_dict.items()
        if k.replace("_orig_mod.", "").replace("module.", "").replace("peft.", "") in {"gain", "bias"}
    }
    peft = DepthAffinePEFT().to(device)
    peft.load_state_dict(state_dict, strict=True)
    peft.eval()
    for param in peft.parameters():
        param.requires_grad_(False)
    return peft, metadata


def resolve_amp_dtype(precision, device):
    """评估阶段的 AMP 配置，与训练脚本保持一致。"""
    if device.type != "cuda" or autocast is None:
        return None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision in {"bf16", "fp16"}:
        return torch.float16
    return None


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def write_scores_csv(path, scores, labels):
    with open(path, "w") as f:
        f.write("rank,score,label\n")
        order = np.argsort(-scores)
        for rank, i in enumerate(order):
            f.write(f"{rank},{scores[i]:.6f},{labels[i]}\n")


def append_eval_log(log_file, result):
    """把补评估结果追加到训练 run 的 training.log。"""
    if not log_file:
        return

    status = result["status"]
    reason = result.get("reason", "")
    auroc = format_metric(result.get("auroc"))
    best_val_loss = result.get("best_val_loss")
    best_val_loss_text = (
        f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else "N/A"
    )

    msg = "\n"
    msg += "=" * 50 + "\n"
    msg += "Post-hoc evaluation on test set (best checkpoint)\n"
    msg += "=" * 50 + "\n"
    msg += f"Evaluation time: {result['evaluated_at']}\n"
    msg += f"Script: scripts/eval_from_ckpt.py\n"
    msg += f"Checkpoint: {result['ckpt']}\n"
    if result.get("depth_peft_ckpt"):
        msg += f"Depth PEFT: {result['depth_peft_ckpt']}\n"
    msg += f"Status: {status}"
    if reason:
        msg += f" ({reason})"
    msg += "\n"
    msg += f"Final Result - Cam {result['view_id']}\n"
    msg += f"  Best epoch: {result.get('best_epoch', 'N/A')}\n"
    msg += f"  Best Val Loss: {best_val_loss_text}\n"
    msg += f"  Test AUROC: {auroc}\n"
    msg += f"  #patches: {result['num_patches']}\n"
    msg += f"  #images: {result['num_images']}\n"
    msg += f"  #abnormal: {result['num_abnormal']}, #normal: {result['num_normal']}\n"
    if result.get("score_min") is not None:
        msg += f"  score range: [{result['score_min']:.4f}, {result['score_max']:.4f}]\n"
    if result.get("scores_csv"):
        msg += f"  Scores CSV: {result['scores_csv']}\n"
    msg += "=" * 50 + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg)


@torch.no_grad()
def evaluate(teacher_rgb, teacher_depth, student_rgb, student_depth,
             test_loader, device, amp_dtype=None, channels_last=False):
    """与 train_trd_rail.py 中的 evaluate 一致：patch 分数 max 聚合到原图。"""
    student_rgb.eval()
    student_depth.eval()

    img_scores = {}
    img_labels = {}

    for data in test_loader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)
        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        amp_ctx = (
            autocast(device.type, enabled=True, dtype=amp_dtype)
            if amp_dtype is not None and autocast is not None
            else nullcontext()
        )
        with amp_ctx:
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

    if len(image_scores) == 0:
        print("Warning: no test samples, AUROC = N/A")
        return None, image_scores, image_labels

    if len(np.unique(image_labels)) < 2:
        print("Warning: only one class in test set, AUROC = N/A")
        return None, image_scores, image_labels

    auroc = roc_auc_score(image_labels, image_scores)
    return auroc, image_scores, image_labels


def build_parser():
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
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--output_log", type=str, default=None,
                        help="可选：把最终评估结果追加到指定 training.log")
    parser.add_argument("--append_log", action="store_true",
                        help="配合 --output_log 使用，确认写回日志")
    parser.add_argument("--scores_csv", type=str, default=None,
                        help="可选：指定分数 CSV 输出路径，默认写到 ckpt 同目录")
    parser.add_argument("--result_json", type=str, default=None,
                        help="可选：保存结构化评估结果 JSON，便于批量汇总")
    parser.add_argument("--depth_peft_ckpt", type=str, default=None,
                        help="可选：DepthAffinePEFT checkpoint；提供时仅替换 depth teacher 调用链")
    return parser


def evaluate_from_args(args):
    evaluated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    print(f"Device: {device}")
    if amp_dtype is None:
        print("Precision: fp32")
    else:
        print(f"Precision: {args.precision}")
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

    result = {
        "evaluated_at": evaluated_at,
        "view_id": int(args.view_id),
        "ckpt": str(args.ckpt),
        "depth_peft_ckpt": str(getattr(args, "depth_peft_ckpt", "") or ""),
        "train_root": str(args.train_root),
        "test_root": str(args.test_root),
        "status": "ok",
        "reason": "",
        "best_epoch": "N/A",
        "best_val_loss": None,
        "auroc": None,
        "num_patches": int(len(test_dataset)),
        "num_images": 0,
        "num_abnormal": 0,
        "num_normal": 0,
        "score_min": None,
        "score_max": None,
        "scores_csv": "",
        "output_log": str(args.output_log) if args.output_log else "",
    }

    if len(test_dataset) == 0:
        result["status"] = "skipped"
        result["reason"] = "no test samples for this view"
        print("\n" + "=" * 60)
        print(f"  Cam{args.view_id} Test Result")
        print("=" * 60)
        print("  AUROC: N/A (no test samples)")
        if args.output_log and args.append_log:
            append_eval_log(args.output_log, result)
            print(f"  appended to log: {args.output_log}")
        if args.result_json:
            with open(args.result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

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

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)
        print("Memory format: channels_last")

    ckpt = safe_load_ckpt(args.ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt['student_rgb']))
    student_depth.load_state_dict(strip_prefix(ckpt['student_depth']))
    result["best_epoch"] = ckpt.get("epoch", "N/A")
    result["best_val_loss"] = ckpt.get("best_val_loss")
    best_val_loss_text = (
        f"{result['best_val_loss']:.4f}"
        if isinstance(result["best_val_loss"], (int, float))
        else "N/A"
    )
    print(f"Loaded epoch={result['best_epoch']}, best_val_loss={best_val_loss_text}")

    depth_peft_ckpt = getattr(args, "depth_peft_ckpt", None)
    if depth_peft_ckpt:
        peft, peft_meta = load_depth_peft(depth_peft_ckpt, device)
        teacher_depth = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        result["depth_peft_ckpt"] = str(depth_peft_ckpt)
        result["depth_peft_gain"] = float(peft.gain.detach().cpu())
        result["depth_peft_bias"] = float(peft.bias.detach().cpu())
        result["depth_peft_meta"] = peft_meta
        print(
            "Loaded Depth PEFT: "
            f"{depth_peft_ckpt} (gain={peft.gain.item():.6f}, bias={peft.bias.item():.6f})"
        )

    # ---- 3. 评估 ----
    auroc, scores, labels = evaluate(
        teacher_rgb, teacher_depth, student_rgb, student_depth,
        test_loader, device,
        amp_dtype=amp_dtype, channels_last=args.channels_last,
    )

    result["auroc"] = float(auroc) if auroc is not None else None
    result["num_images"] = int(len(scores))
    result["num_abnormal"] = int((labels == 1).sum()) if len(labels) else 0
    result["num_normal"] = int((labels == 0).sum()) if len(labels) else 0
    if len(scores):
        result["score_min"] = float(scores.min())
        result["score_max"] = float(scores.max())

    print("\n" + "=" * 60)
    print(f"  Cam{args.view_id} Test Result")
    print("=" * 60)
    if auroc is None:
        print(f"  AUROC: N/A (single-class test set)")
    else:
        print(f"  AUROC: {auroc:.4f}")
    print(f"  #images: {len(scores)}")
    if len(scores):
        print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  #abnormal: {(labels == 1).sum() if len(labels) else 0}, "
          f"#normal: {(labels == 0).sum() if len(labels) else 0}")

    # 把分数落盘，便于后续画分布图
    if len(scores):
        out_csv = args.scores_csv or args.ckpt.replace(".pth", "_test_scores.csv")
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        write_scores_csv(out_csv, scores, labels)
        result["scores_csv"] = str(out_csv)
        print(f"  scores saved to: {out_csv}")

    if args.output_log and args.append_log:
        append_eval_log(args.output_log, result)
        print(f"  appended to log: {args.output_log}")

    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluate_from_args(args)


if __name__ == "__main__":
    main()
