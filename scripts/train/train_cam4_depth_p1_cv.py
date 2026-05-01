#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train/evaluate Cam4 DepthAffinePEFT with 4-fold normal-only adaptation."""

# >>> path-bootstrap >>>
import os
import sys

from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
# <<< path-bootstrap <<<

import argparse
import csv
import json
import random
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from datasets.rail_dataset import RailDualModalDataset
from eval.eval_utils import cal_anomaly_map
from models.trd.decoder import ResNet50DualModalDecoder
from models.trd.encoder import ResNet50Encoder
from rail_peft import (
    DepthAffinePEFT,
    DepthEncoderWithPEFT,
    compute_depth_feature_stats,
    fdm_loss,
)
from scripts.eval.eval_from_ckpt import autocast, resolve_amp_dtype, safe_load_ckpt, strip_prefix


def build_run_layout(output_dir: Path, view_id: int) -> dict[str, Path]:
    layout = {
        "output_dir": output_dir,
        "stats_dir": output_dir / "stats",
        "eval_dir": output_dir / "eval",
        "cv_dir": output_dir / "cv",
        "final_dir": output_dir / "final",
        "diagnostics_dir": output_dir / "diagnostics",
        "summary_txt": output_dir / "summary.txt",
        "summary_csv": output_dir / "summary.csv",
        "depth_peft_map": output_dir / "depth_peft_map.json",
        "reference_stats": output_dir / "stats" / "reference_stats.pt",
        "baseline_scores_csv": output_dir / "eval" / "baseline_scores.csv",
        "final_ckpt": output_dir / "final" / f"final_peft_cam{view_id}.pth",
        "final_history_csv": output_dir / "final" / "final_history.csv",
        "final_scores_csv": output_dir / "final" / "final_peft_scores.csv",
    }
    for key in ["stats_dir", "eval_dir", "cv_dir", "final_dir", "diagnostics_dir"]:
        layout[key].mkdir(parents=True, exist_ok=True)
    return layout


def fold_paths(layout: dict[str, Path], fold_idx: int, view_id: int) -> dict[str, Path]:
    fold_dir = layout["cv_dir"] / f"fold{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": fold_dir,
        "peft_ckpt": fold_dir / f"peft_cam{view_id}.pth",
        "history_csv": fold_dir / "history.csv",
        "scores_csv": fold_dir / "scores.csv",
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Cam4 P1: train a 2-parameter depth affine PEFT with 4-fold CV."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Cam4 best_cam4.pth")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--test_root", type=str, required=True)
    parser.add_argument("--view_id", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--depth_norm", type=str, default="zscore", choices=["zscore", "minmax", "log"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16, help="PEFT train batch size")
    parser.add_argument("--stats_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=4)

    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)

    parser.add_argument("--train_sample_ratio", type=float, default=1.0)
    parser.add_argument("--train_sample_num", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="uniform_time", choices=["uniform_time", "random"])
    parser.add_argument("--output_root", type=str, default="outputs/rail_peft")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context_factory(device: torch.device, amp_dtype):
    if amp_dtype is None or autocast is None:
        return nullcontext
    return lambda: autocast(device.type, enabled=True, dtype=amp_dtype)


def loader_kwargs(num_workers: int, device: torch.device):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def indices_for_frames(dataset: RailDualModalDataset, frame_ids: set[str]) -> list[int]:
    indices = []
    patches = dataset.num_patches
    for img_idx, sample in enumerate(dataset.samples):
        if sample["frame_id"] in frame_ids:
            start = img_idx * patches
            indices.extend(range(start, start + patches))
    return indices


def make_loader(dataset, indices, batch_size, num_workers, device, shuffle=False):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs(num_workers, device),
    )


def build_models(args, device: torch.device):
    teacher_rgb = ResNet50Encoder(pretrained=True).to(device).eval()
    teacher_depth = ResNet50Encoder(pretrained=True).to(device).eval()
    for param in teacher_rgb.parameters():
        param.requires_grad_(False)
    for param in teacher_depth.parameters():
        param.requires_grad_(False)

    student_rgb = ResNet50DualModalDecoder(pretrained=False).to(device).eval()
    student_depth = ResNet50DualModalDecoder(pretrained=False).to(device).eval()

    if args.channels_last and device.type == "cuda":
        teacher_rgb = teacher_rgb.to(memory_format=torch.channels_last)
        teacher_depth = teacher_depth.to(memory_format=torch.channels_last)
        student_rgb = student_rgb.to(memory_format=torch.channels_last)
        student_depth = student_depth.to(memory_format=torch.channels_last)

    ckpt = safe_load_ckpt(args.ckpt, device)
    student_rgb.load_state_dict(strip_prefix(ckpt["student_rgb"]))
    student_depth.load_state_dict(strip_prefix(ckpt["student_depth"]))
    return teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt


@torch.no_grad()
def evaluate_detailed(
    teacher_rgb,
    teacher_depth,
    student_rgb,
    student_depth,
    dataloader,
    device: torch.device,
    amp_ctx_factory,
    channels_last: bool,
):
    teacher_rgb.eval()
    teacher_depth.eval()
    student_rgb.eval()
    student_depth.eval()

    img_scores: dict[str, list[float]] = {}
    img_labels: dict[str, int] = {}

    for data in dataloader:
        rgb = data["intensity"].to(device, non_blocking=True)
        depth = data["depth"].to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            rgb = rgb.contiguous(memory_format=torch.channels_last)
            depth = depth.contiguous(memory_format=torch.channels_last)

        labels = data["label"].cpu().numpy()
        frame_ids = data["frame_id"]

        with amp_ctx_factory():
            feat_t_rgb = teacher_rgb(rgb)
            feat_t_depth = teacher_depth(depth)
            _, _, feat_s_rgb, _ = student_rgb(feat_t_rgb, feat_t_depth)
            _, _, feat_s_depth, _ = student_depth(feat_t_depth, feat_t_rgb)

        amap_rgb, _ = cal_anomaly_map(feat_s_rgb, feat_t_rgb, out_size=(256, 256), amap_mode="mul")
        amap_depth, _ = cal_anomaly_map(feat_s_depth, feat_t_depth, out_size=(256, 256), amap_mode="mul")
        amap = amap_rgb + amap_depth

        if amap.ndim == 3:
            batch_scores = amap.reshape(amap.shape[0], -1).max(axis=1)
        else:
            batch_scores = np.array([amap.max()])

        for score, label, frame_id in zip(batch_scores, labels, frame_ids):
            img_scores.setdefault(frame_id, []).append(float(score))
            img_labels.setdefault(frame_id, int(label))

    frame_ids = sorted(img_scores.keys())
    scores = np.array([max(img_scores[frame_id]) for frame_id in frame_ids], dtype=np.float64)
    labels = np.array([img_labels[frame_id] for frame_id in frame_ids], dtype=np.int64)
    auroc = None if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))
    return auroc, frame_ids, scores, labels


def write_scores_csv(path: Path, frame_ids, scores, labels) -> None:
    order = np.argsort(-scores)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "frame_id", "score", "label"])
        writer.writeheader()
        for rank, idx in enumerate(order):
            writer.writerow({
                "rank": rank,
                "frame_id": frame_ids[idx],
                "score": f"{float(scores[idx]):.8f}",
                "label": int(labels[idx]),
            })


def write_history_csv(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "gain", "bias"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_peft(path: Path, peft: DepthAffinePEFT, args, train_good, test_good=None) -> None:
    payload = {
        "kind": "DepthAffinePEFT",
        "state_dict": peft.state_dict(),
        "view_id": int(args.view_id),
        "gain": float(peft.gain.detach().cpu()),
        "bias": float(peft.bias.detach().cpu()),
        "train_good": list(train_good),
        "test_good": list(test_good or []),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, path)


def train_peft(
    teacher_depth,
    train_loader,
    mu_ref,
    var_ref,
    args,
    device: torch.device,
    amp_ctx_factory,
) -> tuple[DepthAffinePEFT, list[dict]]:
    peft = DepthAffinePEFT().to(device)
    depth_branch = DepthEncoderWithPEFT(teacher_depth, peft).to(device)
    optimizer = torch.optim.Adam(depth_branch.trainable_parameters(), lr=args.lr)

    mu_ref = [mu.to(device) for mu in mu_ref]
    var_ref = [var.to(device) for var in var_ref]
    history = []

    for epoch in range(1, args.epochs + 1):
        depth_branch.encoder.eval()
        depth_branch.peft.train()
        loss_sum = 0.0
        batch_count = 0

        for data in train_loader:
            depth = data["depth"].to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                depth = depth.contiguous(memory_format=torch.channels_last)

            with amp_ctx_factory():
                feats = depth_branch(depth)
                loss = fdm_loss(feats, mu_ref, var_ref)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            batch_count += 1

        avg_loss = loss_sum / max(batch_count, 1)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            row = {
                "epoch": epoch,
                "loss": f"{avg_loss:.8f}",
                "gain": f"{peft.gain.item():.8f}",
                "bias": f"{peft.bias.item():.8f}",
            }
            history.append(row)
            print(
                f"[P1] epoch={epoch:03d} loss={avg_loss:.6f} "
                f"gain={peft.gain.item():.6f} bias={peft.bias.item():.6f}"
            )

    return peft, history


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.precision, device)
    amp_ctx_factory = amp_context_factory(device, amp_dtype)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"cam{args.view_id}_p1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_run_layout(output_dir, args.view_id)

    print(f"Device: {device}")
    print("Precision: fp32" if amp_dtype is None else f"Precision: {args.precision}")
    print(f"Output: {output_dir}")

    print("[P1] loading datasets ...")
    stats_dataset = RailDualModalDataset(
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=args.view_id,
        split="train",
        img_size=args.img_size,
        depth_norm=args.depth_norm,
        use_patch=False,
        train_sample_ratio=args.train_sample_ratio,
        train_sample_num=args.train_sample_num,
        train_val_test_split=[1.0, 0.0, 0.0],
        sampling_mode=args.sampling_mode,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )
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

    good_ids = sorted([s["frame_id"] for s in test_dataset.samples if int(s["label"]) == 0])
    broken_ids = sorted([s["frame_id"] for s in test_dataset.samples if int(s["label"]) == 1])
    if len(good_ids) < args.folds:
        raise RuntimeError(f"Need at least {args.folds} good images, got {len(good_ids)}")
    if len(broken_ids) == 0:
        raise RuntimeError("No broken images found; AUROC cannot be computed.")
    print(f"[P1] Cam{args.view_id} test images: good={len(good_ids)}, broken={len(broken_ids)}")

    print("[P1] loading models ...")
    teacher_rgb, teacher_depth, student_rgb, student_depth, ckpt = build_models(args, device)
    print(f"[P1] loaded checkpoint epoch={ckpt.get('epoch', 'N/A')}, best_val_loss={ckpt.get('best_val_loss')}")

    stats_loader = DataLoader(
        stats_dataset,
        batch_size=args.stats_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs(args.num_workers, device),
    )
    print("[P1] computing Cam4 train-domain depth feature stats ...")
    mu_ref, var_ref = compute_depth_feature_stats(
        teacher_depth, stats_loader, device=device, amp_context_factory=amp_ctx_factory
    )
    torch.save({"mu": [t.cpu() for t in mu_ref], "var": [t.cpu() for t in var_ref]}, layout["reference_stats"])
    print(f"[P1] feature layers: {[int(mu.shape[0]) for mu in mu_ref]}")

    all_test_indices = indices_for_frames(test_dataset, set(good_ids + broken_ids))
    all_test_loader = make_loader(
        test_dataset, all_test_indices, args.eval_batch_size, args.num_workers, device, shuffle=False
    )
    baseline_auroc, baseline_frames, baseline_scores, baseline_labels = evaluate_detailed(
        teacher_rgb, teacher_depth, student_rgb, student_depth,
        all_test_loader, device, amp_ctx_factory, args.channels_last,
    )
    baseline_scores_csv = layout["baseline_scores_csv"]
    write_scores_csv(baseline_scores_csv, baseline_frames, baseline_scores, baseline_labels)
    print(f"[P1] baseline AUROC: {baseline_auroc:.4f}")

    fold_parts = [list(part) for part in np.array_split(np.array(good_ids), args.folds)]
    summary_rows = [{
        "run_type": "baseline",
        "fold": "",
        "auroc": f"{baseline_auroc:.8f}",
        "num_images": len(baseline_scores),
        "num_normal": int((baseline_labels == 0).sum()),
        "num_abnormal": int((baseline_labels == 1).sum()),
        "gain": "",
        "bias": "",
        "peft_ckpt": "",
        "scores_csv": str(baseline_scores_csv),
        "train_good": "",
        "test_good": "",
    }]

    fold_aurocs = []
    for fold_idx, test_good in enumerate(fold_parts, start=1):
        train_good = [frame_id for frame_id in good_ids if frame_id not in set(test_good)]
        print("\n" + "=" * 72)
        print(f"[P1] fold {fold_idx}/{args.folds}: train_good={len(train_good)}, test_good={len(test_good)}")
        print("=" * 72)

        train_indices = indices_for_frames(test_dataset, set(train_good))
        eval_indices = indices_for_frames(test_dataset, set(test_good + broken_ids))
        train_loader = make_loader(
            test_dataset, train_indices, args.batch_size, args.num_workers, device, shuffle=True
        )
        eval_loader = make_loader(
            test_dataset, eval_indices, args.eval_batch_size, args.num_workers, device, shuffle=False
        )

        peft, history = train_peft(teacher_depth, train_loader, mu_ref, var_ref, args, device, amp_ctx_factory)
        fold_path = fold_paths(layout, fold_idx, args.view_id)
        peft_ckpt = fold_path["peft_ckpt"]
        save_peft(peft_ckpt, peft, args, train_good=train_good, test_good=test_good)
        write_history_csv(fold_path["history_csv"], history)

        depth_branch = DepthEncoderWithPEFT(teacher_depth, peft).to(device).eval()
        auroc, frames, scores, labels = evaluate_detailed(
            teacher_rgb, depth_branch, student_rgb, student_depth,
            eval_loader, device, amp_ctx_factory, args.channels_last,
        )
        scores_csv = fold_path["scores_csv"]
        write_scores_csv(scores_csv, frames, scores, labels)
        fold_aurocs.append(float(auroc))
        print(f"[P1] fold {fold_idx} AUROC: {auroc:.4f}")

        summary_rows.append({
            "run_type": "p1_cv",
            "fold": fold_idx,
            "auroc": f"{float(auroc):.8f}",
            "num_images": len(scores),
            "num_normal": int((labels == 0).sum()),
            "num_abnormal": int((labels == 1).sum()),
            "gain": f"{peft.gain.item():.8f}",
            "bias": f"{peft.bias.item():.8f}",
            "peft_ckpt": str(peft_ckpt),
            "scores_csv": str(scores_csv),
            "train_good": " ".join(train_good),
            "test_good": " ".join(test_good),
        })

    print("\n" + "=" * 72)
    print("[P1] training final PEFT on all original good images")
    print("=" * 72)
    final_train_indices = indices_for_frames(test_dataset, set(good_ids))
    final_train_loader = make_loader(
        test_dataset, final_train_indices, args.batch_size, args.num_workers, device, shuffle=True
    )
    final_peft, final_history = train_peft(
        teacher_depth, final_train_loader, mu_ref, var_ref, args, device, amp_ctx_factory
    )
    final_ckpt = layout["final_ckpt"]
    save_peft(final_ckpt, final_peft, args, train_good=good_ids)
    write_history_csv(layout["final_history_csv"], final_history)

    final_branch = DepthEncoderWithPEFT(teacher_depth, final_peft).to(device).eval()
    final_auroc, final_frames, final_scores, final_labels = evaluate_detailed(
        teacher_rgb, final_branch, student_rgb, student_depth,
        all_test_loader, device, amp_ctx_factory, args.channels_last,
    )
    final_scores_csv = layout["final_scores_csv"]
    write_scores_csv(final_scores_csv, final_frames, final_scores, final_labels)
    summary_rows.append({
        "run_type": "p1_final_train_all_good",
        "fold": "",
        "auroc": f"{float(final_auroc):.8f}",
        "num_images": len(final_scores),
        "num_normal": int((final_labels == 0).sum()),
        "num_abnormal": int((final_labels == 1).sum()),
        "gain": f"{final_peft.gain.item():.8f}",
        "bias": f"{final_peft.bias.item():.8f}",
        "peft_ckpt": str(final_ckpt),
        "scores_csv": str(final_scores_csv),
        "train_good": " ".join(good_ids),
        "test_good": "",
    })

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs, ddof=1)) if len(fold_aurocs) > 1 else 0.0
    summary_csv = layout["summary_csv"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_type", "fold", "auroc", "num_images", "num_normal", "num_abnormal",
            "gain", "bias", "peft_ckpt", "scores_csv", "train_good", "test_good",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    summary_txt = layout["summary_txt"]
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Cam{args.view_id} DepthAffinePEFT P1\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"train_root: {args.train_root}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"baseline_auroc: {baseline_auroc:.6f}\n")
        f.write(f"cv_auroc_mean: {mean_auroc:.6f}\n")
        f.write(f"cv_auroc_std: {std_auroc:.6f}\n")
        f.write(f"final_peft_auroc_on_original_test: {final_auroc:.6f}\n")
        f.write(f"final_gain: {final_peft.gain.item():.8f}\n")
        f.write(f"final_bias: {final_peft.bias.item():.8f}\n")
        f.write(f"final_peft_ckpt: {final_ckpt}\n")

    peft_map = {str(args.view_id): str(final_ckpt)}
    with open(layout["depth_peft_map"], "w", encoding="utf-8") as f:
        json.dump(peft_map, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("[P1] finished")
    print("=" * 72)
    print(f"Baseline AUROC: {baseline_auroc:.4f}")
    print(f"CV AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    print(f"Final PEFT AUROC on original test: {final_auroc:.4f}")
    print(f"Final PEFT: {final_ckpt}")
    print(f"Summary: {summary_csv}")
    print(f"PEFT map: {layout['depth_peft_map']}")


if __name__ == "__main__":
    main()
