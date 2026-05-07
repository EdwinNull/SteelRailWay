#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal PaDiM baseline on Cam4 for horizontal comparison table.

PaDiM: multivariate Gaussian per patch position.
Uses WideResNet50_2 (same backbone as TRD for fair comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from scripts.baselines.run_patchcore_cam4 import (
    RailRGBDataset, extract_features,
)


def estimate_gaussians(features: dict[int, list[np.ndarray]],
                       layers: tuple = (1, 2)) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Estimate per-patch multivariate Gaussian (mean, inv_cov)."""
    params = {}
    for layer_idx in layers:
        feat_list = features[layer_idx]
        # Stack: [N, C, H, W]
        all_feats = np.concatenate(feat_list, axis=0)  # [N, C, H, W]
        N, C, H, W = all_feats.shape
        # [H*W, N, C]
        patches = all_feats.transpose(0, 2, 3, 1).reshape(N, H * W, C).transpose(1, 0, 2)
        means = np.zeros((H * W, C))
        inv_covs = np.zeros((H * W, C, C))
        eps = 1e-4
        for p in range(H * W):
            patch_data = patches[p]  # [N, C]
            mu = patch_data.mean(axis=0)
            cov = np.cov(patch_data, rowvar=False)
            cov += eps * np.eye(C)
            means[p] = mu
            inv_covs[p] = np.linalg.inv(cov)
        params[layer_idx] = (means, inv_covs)
    return params


def score_padim(encoder: torch.nn.Module, loader: DataLoader,
                gaussian_params: dict[int, tuple[np.ndarray, np.ndarray]],
                device: torch.device, layers: tuple = (1, 2),
                ) -> tuple[list[float], list[np.ndarray], list[str]]:
    """Compute PaDiM anomaly scores using vectorized Mahalanobis distance."""
    encoder.eval()
    image_scores: list[float] = []
    pixel_maps: list[np.ndarray] = []
    frame_ids: list[str] = []

    with torch.no_grad():
        for rgb, _, _, fids in loader:
            rgb = rgb.to(device)
            feats = encoder(rgb)
            batch_maps = []
            for layer_idx in layers:
                (means, inv_covs) = gaussian_params[layer_idx]  # means:[P,C], inv_covs:[P,C,C]
                P, C = means.shape
                f = feats[layer_idx].cpu().numpy()  # [B, C, H, W]
                B, Cf, H, W_ = f.shape
                assert C == Cf and P == H * W_, f"Shape mismatch: means {P}x{C}, feat {H}x{W_}x{Cf}"

                f_patches = f.transpose(0, 2, 3, 1).reshape(B, H * W_, C)  # [B, P, C]
                diff = f_patches - means[None, :, :]  # [B, P, C]
                # Vectorized Mahalanobis: for each (b,p), diff[b,p] @ inv_covs[p] @ diff[b,p]
                # = sum_c(diff[b,p,c] * sum_c'(inv_covs[p,c,c'] * diff[b,p,c']))
                # = einsum('bpc,pcd,bpd->bp', diff, inv_covs, diff)
                scores = np.sqrt(np.maximum(
                    np.einsum('bpc,pcd,bpd->bp', diff, inv_covs, diff), 0.0
                ))  # [B, P]
                score_map = scores.reshape(B, H, W_)
                batch_maps.append(score_map)

            combined = np.max(batch_maps, axis=0)  # [B, H, W]
            for b in range(B):
                image_scores.append(float(np.max(combined[b])))
                pixel_maps.append(combined[b])
                frame_ids.append(fids[b])

    return image_scores, pixel_maps, frame_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="PaDiM baseline on Cam4")
    parser.add_argument("--data_root", type=str,
                        default=str(_PROJ_ROOT / "rail_mvtec_gt_test"))
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    train_set = RailRGBDataset(data_root, args.cam, "test")
    test_set = RailRGBDataset(data_root, args.cam, "test")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    from models.trd.encoder import ResNet50Encoder
    encoder = ResNet50Encoder(pretrained=True).to(device)

    print("Extracting features...")
    feats = extract_features(encoder, train_loader, device, max_samples=200)
    print("Estimating Gaussians...")
    params = estimate_gaussians(feats)
    for li, (mu, ic) in params.items():
        print(f"  Layer {li}: means {mu.shape}, inv_cov {ic.shape}")

    print("Scoring...")
    image_scores, pixel_maps, frame_ids = score_padim(
        encoder, test_loader, params, device)

    labels = [s[1] for s in test_set.samples]
    image_auroc = roc_auc_score(labels, image_scores)
    print(f"\nPaDiM Cam4 Image AUROC: {image_auroc:.4f}")

    # Pixel AUROC
    gt_root = data_root / "rail_mvtec" / args.cam / "ground_truth" / "broken"
    from PIL import Image
    pixel_labels = []
    pixel_scores_flat = []
    for (_, label, _, fid), pmap in zip(test_set.samples, pixel_maps):
        if label == 1:
            gt_path = gt_root / f"{fid}.png"
            if gt_path.exists():
                gt = np.asarray(Image.open(gt_path).convert("L").resize((512, 512))) > 128
                pixel_labels.append(gt.flatten())
            else:
                pixel_labels.append(np.zeros(512 * 512, dtype=bool))
        else:
            pixel_labels.append(np.zeros(512 * 512, dtype=bool))
        pmap_resized = np.array(Image.fromarray(pmap.astype(np.float32)).resize((512, 512)))
        pixel_scores_flat.append(pmap_resized.flatten())

    pixel_labels_all = np.concatenate(pixel_labels)
    pixel_scores_all = np.concatenate(pixel_scores_flat)
    pixel_auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    print(f"PaDiM Cam4 Pixel AUROC: {pixel_auroc:.4f}")

    result = {
        "method": "PaDiM",
        "cam": args.cam,
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "num_test_samples": len(test_set),
    }
    out_path = Path(args.output) if args.output else (
        _PROJ_ROOT / "results" / f"padim_{args.cam}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(str(out_path), "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
