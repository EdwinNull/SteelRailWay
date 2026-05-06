#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal PaDiM baseline on Cam4.

This script reuses the same rail-patch protocol and split helpers as the
PatchCore baseline so the two representation-based methods stay comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from models.trd.encoder import ResNet50Encoder
from scripts.baselines.run_patchcore_cam4 import (
    MANIFEST_SELECTIONS,
    build_rgb_patch_dataset,
    build_loader,
    compute_safe_auroc,
    dataset_summary,
    default_manifest_path,
    parse_layers,
    parse_view_id,
    write_scores_csv,
)


def collect_layer_features(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
) -> dict[int, np.ndarray]:
    encoder.eval()
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["intensity"].to(device, non_blocking=True)
            feats = encoder(rgb)
            for layer_idx in layers:
                feat = feats[layer_idx]
                if target_map_size > 0 and feat.shape[-1] != target_map_size:
                    feat = F.adaptive_avg_pool2d(feat, (target_map_size, target_map_size))
                collected[layer_idx].append(feat.detach().cpu().numpy().astype(np.float32))
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in collected.items()}


def estimate_gaussians(features: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    params: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    eps = 1e-4
    for layer_idx, feat in features.items():
        n, c, h, w = feat.shape
        patches = feat.transpose(0, 2, 3, 1).reshape(n, h * w, c).transpose(1, 0, 2)
        means = patches.mean(axis=1).astype(np.float32)
        inv_covs = np.zeros((h * w, c, c), dtype=np.float32)
        for patch_idx in range(h * w):
            patch_data = patches[patch_idx]
            cov = np.cov(patch_data, rowvar=False).astype(np.float32)
            cov += eps * np.eye(c, dtype=np.float32)
            inv_covs[patch_idx] = np.linalg.inv(cov).astype(np.float32)
        params[layer_idx] = (means, inv_covs)
    return params


def score_padim(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    params: dict[int, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    encoder.eval()
    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    pixel_score_chunks: list[np.ndarray] = []
    pixel_gt_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["intensity"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])
            feats = encoder(rgb)

            batch_maps = []
            for layer_idx in layers:
                feat = feats[layer_idx]
                if target_map_size > 0 and feat.shape[-1] != target_map_size:
                    feat = F.adaptive_avg_pool2d(feat, (target_map_size, target_map_size))
                feat_np = feat.detach().cpu().numpy().astype(np.float32)
                bsz, channels, h, w = feat_np.shape
                means, inv_covs = params[layer_idx]
                feat_patches = feat_np.transpose(0, 2, 3, 1).reshape(bsz, h * w, channels)
                scores = np.zeros((bsz, h * w), dtype=np.float32)
                for b in range(bsz):
                    for patch_idx in range(h * w):
                        diff = feat_patches[b, patch_idx] - means[patch_idx]
                        scores[b, patch_idx] = float(np.sqrt(diff @ inv_covs[patch_idx] @ diff))
                batch_maps.append(scores.reshape(bsz, h, w))

            combined = np.max(np.stack(batch_maps, axis=0), axis=0)
            combined = torch.from_numpy(combined).unsqueeze(1)
            combined = F.interpolate(
                combined,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).numpy()

            pixel_score_chunks.append(combined.reshape(len(frame_ids), -1))
            pixel_gt_chunks.append(gt.reshape(len(frame_ids), -1))

            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(combined[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    ordered_frame_ids = sorted(frame_scores.keys())
    image_scores = np.array([max(frame_scores[fid]) for fid in ordered_frame_ids], dtype=np.float64)
    image_labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)
    pixel_scores = np.concatenate(pixel_score_chunks, axis=0).reshape(-1)
    pixel_labels = np.concatenate(pixel_gt_chunks, axis=0).reshape(-1)
    return image_scores, image_labels, ordered_frame_ids, pixel_scores, pixel_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM baseline on Cam4")
    parser.add_argument("--eval_root", type=str, default=str(_PROJ_ROOT / "rail_mvtec_gt_test"))
    parser.add_argument("--bank_root", type=str, default=str(_PROJ_ROOT / "rail_mvtec_gt_test_aug_cam4_normal50"))
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target_map_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--bank_selection",
        type=str,
        default="manifest_added_good",
        choices=["all", "good", "manifest_added_good", "manifest_original_good"],
    )
    parser.add_argument(
        "--eval_selection",
        type=str,
        default="all",
        choices=["all", "good", "broken", "manifest_original_eval"],
    )
    parser.add_argument("--bank_manifest", type=str, default="")
    parser.add_argument("--eval_manifest", type=str, default="")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default="")
    args = parser.parse_args()

    view_id = parse_view_id(args.cam)
    layers = parse_layers(args.layers)
    device = torch.device(args.device)

    eval_root = Path(args.eval_root)
    bank_root = Path(args.bank_root)
    bank_manifest = Path(args.bank_manifest) if args.bank_manifest else default_manifest_path(bank_root, view_id)
    eval_manifest = Path(args.eval_manifest) if args.eval_manifest else default_manifest_path(eval_root, view_id)

    bank_dataset = build_rgb_patch_dataset(
        root=bank_root,
        view_id=view_id,
        img_size=args.img_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        selection=args.bank_selection,
        manifest_path=bank_manifest,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )
    eval_dataset = build_rgb_patch_dataset(
        root=eval_root,
        view_id=view_id,
        img_size=args.img_size,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        selection=args.eval_selection,
        manifest_path=eval_manifest if args.eval_selection in MANIFEST_SELECTIONS else None,
        preload=args.preload,
        preload_workers=args.preload_workers,
    )

    bank_loader = build_loader(bank_dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    eval_loader = build_loader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    bank_summary = dataset_summary(bank_dataset)
    eval_summary = dataset_summary(eval_dataset)

    encoder = ResNet50Encoder(pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print(f"Using device: {device}")
    print(f"Memory bank dataset: {bank_summary}")
    print(f"Evaluation dataset: {eval_summary}")
    print("Extracting train-domain features...")
    train_features = collect_layer_features(
        encoder,
        bank_loader,
        device=device,
        layers=layers,
        target_map_size=args.target_map_size,
    )
    print("Estimating Gaussians...")
    params = estimate_gaussians(train_features)
    for layer_idx, (means, inv_covs) in params.items():
        print(f"  Layer {layer_idx}: means {means.shape}, inv_cov {inv_covs.shape}")

    print("Scoring evaluation set...")
    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = score_padim(
        encoder,
        eval_loader,
        params=params,
        device=device,
        layers=layers,
        target_map_size=args.target_map_size,
        img_size=args.img_size,
    )

    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)
    print()
    print(f"PaDiM Cam{view_id} Image AUROC: {image_auroc if image_auroc is not None else 'N/A'}")
    print(f"PaDiM Cam{view_id} Pixel AUROC: {pixel_auroc if pixel_auroc is not None else 'N/A'}")

    out_path = Path(args.output) if args.output else (_PROJ_ROOT / "results" / f"padim_cam{view_id}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_csv_path = Path(args.scores_csv) if args.scores_csv else out_path.with_name(out_path.stem + "_scores.csv")
    write_scores_csv(scores_csv_path, frame_ids, image_labels, image_scores)

    result = {
        "method": "PaDiM",
        "cam": f"cam{view_id}",
        "view_id": int(view_id),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "layers": list(layers),
        "target_map_size": int(args.target_map_size),
        "img_size": int(args.img_size),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "eval_root": str(eval_root),
        "bank_root": str(bank_root),
        "bank_selection": str(args.bank_selection),
        "eval_selection": str(args.eval_selection),
        "bank_manifest": str(bank_manifest) if bank_manifest.exists() else "",
        "eval_manifest": str(eval_manifest) if eval_manifest.exists() else "",
        "bank_summary": bank_summary,
        "eval_summary": eval_summary,
        "scores_csv": str(scores_csv_path),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Scores saved to {scores_csv_path}")


if __name__ == "__main__":
    main()
