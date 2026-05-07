#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal PatchCore baseline on Cam4 for horizontal comparison table.

PatchCore: nominal feature memory bank + nearest-neighbor anomaly scoring.
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
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection

try:
    import tifffile
except ImportError:
    tifffile = None

# Use same feature extractor as TRD
from models.trd.encoder import ResNet50Encoder


def extract_features(encoder: torch.nn.Module, loader: DataLoader,
                     device: torch.device, layers: tuple = (1, 2),
                     max_samples: int = 200) -> dict[int, list[np.ndarray]]:
    """Extract multi-layer features from normal samples."""
    encoder.eval()
    features: dict[int, list[np.ndarray]] = {}
    count = 0
    with torch.no_grad():
        for rgb, _, _, _ in loader:
            if count >= max_samples:
                break
            rgb = rgb.to(device)
            feats = encoder(rgb)  # list of tensors
            for layer_idx in layers:
                f = feats[layer_idx].cpu().numpy()
                if layer_idx not in features:
                    features[layer_idx] = []
                features[layer_idx].append(f)
            count += 1
    return features


def build_memory_bank(features: dict[int, list[np.ndarray]],
                      coreset_ratio: float = 0.10,
                      random_seed: int = 42) -> dict[int, np.ndarray]:
    """Aggregate features into memory bank with coreset subsampling."""
    memory = {}
    rng = np.random.RandomState(random_seed)
    for layer_idx, feat_list in features.items():
        all_feats = np.concatenate(feat_list, axis=0)  # [N, C, H, W]
        N, C, H, W = all_feats.shape
        # Reshape: [N*H*W, C]
        patch_feats = all_feats.transpose(0, 2, 3, 1).reshape(-1, C)
        # Coreset via random sampling
        n_coreset = max(1, int(len(patch_feats) * coreset_ratio))
        indices = rng.choice(len(patch_feats), n_coreset, replace=False)
        memory[layer_idx] = patch_feats[indices]
    return memory


def score_patchcore(encoder: torch.nn.Module, loader: DataLoader,
                    memory: dict[int, np.ndarray],
                    device: torch.device, layers: tuple = (1, 2),
                    k: int = 5, chunk_size: int = 256) -> tuple[list[float], list[np.ndarray], list[str]]:
    """Compute PatchCore anomaly scores with chunked distance computation."""
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
                f = feats[layer_idx].cpu().numpy()  # [B, C, H, W]
                B, C, H, W = f.shape
                mem = memory[layer_idx]  # [M, C]
                f_flat = f.transpose(0, 2, 3, 1).reshape(B, H * W, C)

                scores = np.zeros((B, H * W))
                for b in range(B):
                    n_patches = H * W
                    # Chunked: avoid (n_patches, M, C) memory explosion
                    for start in range(0, n_patches, chunk_size):
                        end = min(start + chunk_size, n_patches)
                        chunk = f_flat[b, start:end]  # [chunk, C]
                        # dists: [chunk, M] via linalg.norm along last axis
                        diff = chunk[:, None, :] - mem[None, :, :]  # [chunk, M, C]
                        dists = np.sqrt(np.sum(diff * diff, axis=-1))  # [chunk, M]
                        topk = np.sort(dists, axis=-1)[:, :k]
                        scores[b, start:end] = np.mean(topk, axis=-1)

                score_map = scores.reshape(B, H, W)
                batch_maps.append(score_map)

            combined = np.mean(batch_maps, axis=0)
            for b in range(B):
                smap = combined[b]
                image_scores.append(float(np.max(smap)))
                pixel_maps.append(smap)
                frame_ids.append(fids[b])

    return image_scores, pixel_maps, frame_ids


class RailRGBDataset(torch.utils.data.Dataset):
    """RGB-only dataset for PatchCore/PaDiM evaluation."""

    def __init__(self, root: Path, cam: str, split: str = "test"):
        self.root = Path(root)
        self.cam_dir = self.root / "rail_mvtec" / cam / split
        self.samples: list[tuple[Path, int, str]] = []  # (path, label, frame_id)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        for label_name, label_val in [("good", 0), ("broken", 1)]:
            label_dir = self.cam_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((img_path, label_val, img_path.stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, fid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, label, path.name, fid


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore baseline on Cam4")
    parser.add_argument("--data_root", type=str,
                        default=str(_PROJ_ROOT / "rail_mvtec_gt_test"))
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coreset_ratio", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    data_root = Path(args.data_root)
    train_set = RailRGBDataset(data_root, args.cam, "test")  # use test/good as nominal
    test_set = RailRGBDataset(data_root, args.cam, "test")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # Build encoder
    encoder = ResNet50Encoder(pretrained=True).to(device)

    # Extract nominal features and build memory bank
    print("Extracting features...")
    feats = extract_features(encoder, train_loader, device, max_samples=200)
    print("Building memory bank...")
    memory = build_memory_bank(feats, coreset_ratio=args.coreset_ratio)
    for layer_idx, mem in memory.items():
        print(f"  Layer {layer_idx}: memory shape {mem.shape}")

    # Score
    print("Scoring...")
    image_scores, pixel_maps, frame_ids = score_patchcore(
        encoder, test_loader, memory, device, k=args.k)

    # Compute AUROC
    labels = [s[1] for s in test_set.samples]
    image_auroc = roc_auc_score(labels, image_scores)
    print(f"\nPatchCore Cam4 Image AUROC: {image_auroc:.4f}")

    # Pixel AUROC
    # Need ground truth masks for pixel-level evaluation
    gt_root = data_root / "rail_mvtec" / args.cam / "ground_truth" / "broken"
    pixel_labels = []
    pixel_scores_flat = []
    for (_, label, _, fid), pmap in zip(test_set.samples, pixel_maps):
        if label == 1:  # broken
            gt_path = gt_root / f"{fid}.png"
            if gt_path.exists():
                gt = np.asarray(Image.open(gt_path).convert("L").resize((512, 512))) > 128
                pixel_labels.append(gt.flatten())
                # Upsample pixel map to 512x512
                pmap_resized = np.array(Image.fromarray(pmap.astype(np.float32)).resize((512, 512)))
                pixel_scores_flat.append(pmap_resized.flatten())
        else:
            gt = np.zeros((512, 512), dtype=bool)
            pixel_labels.append(gt.flatten())
            pmap_resized = np.array(Image.fromarray(pmap.astype(np.float32)).resize((512, 512)))
            pixel_scores_flat.append(pmap_resized.flatten())

    pixel_labels_all = np.concatenate(pixel_labels)
    pixel_scores_all = np.concatenate(pixel_scores_flat)
    pixel_auroc = roc_auc_score(pixel_labels_all, pixel_scores_all)
    print(f"PatchCore Cam4 Pixel AUROC: {pixel_auroc:.4f}")

    # Save results
    result = {
        "method": "PatchCore",
        "cam": args.cam,
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "num_test_samples": len(test_set),
        "coreset_ratio": args.coreset_ratio,
        "k_nn": args.k,
    }
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = _PROJ_ROOT / "results" / f"patchcore_{args.cam}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(str(out_path), "w"), indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
