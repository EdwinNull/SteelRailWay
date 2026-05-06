#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PatchCore baseline on the rail Cam4 view.

This version fixes three issues in the earlier draft:
1. the feature-layer indices now match the local TRD encoder output;
2. the memory bank and evaluation split are cleanly separated;
3. evaluation follows the repo's patch-wise rail protocol instead of
   squeezing a 6000x900 rail strip directly into a square image.

Recommended comparison protocol for Cam4:
  - evaluation root: ``rail_mvtec_gt_test``
  - memory-bank root: ``rail_mvtec_gt_test_aug_cam4_normal50``
  - bank selection: ``manifest_added_good``

The augmented root contains 42 extra normal Cam4 images copied from the
training-domain normal pool. Using only those ``is_added=True`` samples for
the memory bank avoids leakage into the original 8-good / 10-broken Cam4
evaluation set.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader

from datasets.rail_dataset import RailDualModalDataset
from models.trd.encoder import ResNet50Encoder


MANIFEST_SELECTIONS = {
    "manifest_added_good",
    "manifest_original_good",
    "manifest_original_eval",
}


def parse_view_id(cam: str | int) -> int:
    if isinstance(cam, int):
        return cam
    cam = str(cam).strip().lower()
    if cam.startswith("cam"):
        cam = cam[3:]
    return int(cam)


def default_manifest_path(root: Path, view_id: int) -> Path:
    return root / f"cam{view_id}_augmented_manifest.json"


def load_manifest_frame_ids(manifest_path: Path, selection: str) -> set[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    selected: set[str] = set()
    for item in items:
        label = int(item["label"])
        is_added = bool(item["is_added"])
        frame_id = str(item["frame_id"])
        if selection == "manifest_added_good":
            if label == 0 and is_added:
                selected.add(frame_id)
        elif selection == "manifest_original_good":
            if label == 0 and not is_added:
                selected.add(frame_id)
        elif selection == "manifest_original_eval":
            if label == 1 or (label == 0 and not is_added):
                selected.add(frame_id)
        else:
            raise ValueError(f"Unsupported manifest selection: {selection}")
    return selected


def filter_samples(samples: list[dict], selection: str, manifest_path: Path | None) -> list[dict]:
    if selection == "all":
        return list(samples)
    if selection == "good":
        return [sample for sample in samples if int(sample["label"]) == 0]
    if selection == "broken":
        return [sample for sample in samples if int(sample["label"]) == 1]
    if selection in MANIFEST_SELECTIONS:
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for selection {selection}: {manifest_path}")
        frame_ids = load_manifest_frame_ids(manifest_path, selection)
        return [sample for sample in samples if sample["frame_id"] in frame_ids]
    raise ValueError(f"Unsupported sample selection: {selection}")


def build_rgb_patch_dataset(
    *,
    root: Path,
    view_id: int,
    img_size: int,
    patch_size: int,
    patch_stride: int,
    selection: str,
    manifest_path: Path | None,
    preload: bool,
    preload_workers: int,
) -> RailDualModalDataset:
    dataset = RailDualModalDataset(
        train_root=str(_PROJ_ROOT),
        test_root=str(root),
        view_id=view_id,
        split="test",
        img_size=img_size,
        depth_norm="zscore",
        use_patch=True,
        patch_size=patch_size,
        patch_stride=patch_stride,
        preload=preload,
        preload_workers=preload_workers,
    )
    dataset.samples = filter_samples(dataset.samples, selection, manifest_path)
    if not dataset.samples:
        raise RuntimeError(f"No samples left after selection={selection} under {root}")
    return dataset


def dataset_summary(dataset: RailDualModalDataset) -> dict[str, int]:
    num_images = len(dataset.samples)
    num_good = sum(int(sample["label"]) == 0 for sample in dataset.samples)
    num_broken = sum(int(sample["label"]) == 1 for sample in dataset.samples)
    return {
        "num_images": int(num_images),
        "num_good_images": int(num_good),
        "num_broken_images": int(num_broken),
        "num_patch_samples": int(len(dataset)),
        "num_patches_per_image": int(dataset.num_patches),
    }


def build_loader(
    dataset: RailDualModalDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def parse_layers(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    layers = tuple(int(v) for v in values)
    if not layers:
        raise ValueError("At least one layer index is required")
    if min(layers) < 0 or max(layers) > 2:
        raise ValueError(f"Encoder emits 3 feature maps only; got layers={layers}")
    return layers


def extract_patch_embeddings(
    encoder: torch.nn.Module,
    rgb: torch.Tensor,
    *,
    layers: tuple[int, ...],
    target_map_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    feats = encoder(rgb)
    selected = []
    for layer_idx in layers:
        feat = feats[layer_idx]
        if target_map_size > 0 and feat.shape[-1] != target_map_size:
            feat = F.adaptive_avg_pool2d(feat, (target_map_size, target_map_size))
        selected.append(feat)
    embedding = torch.cat(selected, dim=1)
    embedding = F.normalize(embedding, dim=1)
    batch, channels, height, width = embedding.shape
    embedding = embedding.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return embedding, channels, height, width


def collect_memory_features(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
) -> tuple[np.ndarray, int]:
    encoder.eval()
    chunks: list[np.ndarray] = []
    channels = 0
    with torch.no_grad():
        for batch in loader:
            rgb = batch["intensity"].to(device, non_blocking=True)
            embedding, channels, _, _ = extract_patch_embeddings(
                encoder,
                rgb,
                layers=layers,
                target_map_size=target_map_size,
            )
            chunks.append(
                embedding.reshape(-1, embedding.shape[-1]).detach().cpu().numpy().astype(np.float32)
            )
    if not chunks:
        raise RuntimeError("No memory-bank features were extracted")
    return np.concatenate(chunks, axis=0), int(channels)


def build_memory_bank(
    features: np.ndarray,
    *,
    coreset_ratio: float,
    coreset_size: int,
    projection_dim: int,
    random_seed: int,
) -> tuple[np.ndarray, SparseRandomProjection | None, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got {features.shape}")

    projector: SparseRandomProjection | None = None
    projected = features.astype(np.float32, copy=False)
    if projection_dim > 0 and projection_dim < projected.shape[1]:
        projector = SparseRandomProjection(
            n_components=projection_dim,
            dense_output=True,
            random_state=random_seed,
        )
        projected = projector.fit_transform(projected).astype(np.float32, copy=False)

    total = int(projected.shape[0])
    if coreset_size > 0:
        keep = min(total, int(coreset_size))
    else:
        keep = max(1, int(round(total * float(coreset_ratio))))
        keep = min(total, keep)

    rng = np.random.RandomState(random_seed)
    if keep == total:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(total, size=keep, replace=False))
    return projected[indices], projector, indices


def batched_knn_distances(
    index: NearestNeighbors,
    queries: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start:start + chunk_size]
        distances, _ = index.kneighbors(chunk, return_distance=True)
        chunks.append(distances.astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def score_patchcore(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    index: NearestNeighbors,
    projector: SparseRandomProjection | None,
    device: torch.device,
    layers: tuple[int, ...],
    target_map_size: int,
    img_size: int,
    query_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    encoder.eval()
    frame_scores: dict[str, list[float]] = {}
    frame_labels: dict[str, int] = {}
    patch_score_chunks: list[np.ndarray] = []
    patch_gt_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["intensity"].to(device, non_blocking=True)
            labels = batch["label"].cpu().numpy().astype(np.int64)
            gt = batch["gt"].cpu().numpy().astype(np.float32)
            frame_ids = list(batch["frame_id"])

            embedding, _, h, w = extract_patch_embeddings(
                encoder,
                rgb,
                layers=layers,
                target_map_size=target_map_size,
            )
            queries = embedding.reshape(-1, embedding.shape[-1]).detach().cpu().numpy().astype(np.float32)
            if projector is not None:
                queries = projector.transform(queries).astype(np.float32, copy=False)

            distances = batched_knn_distances(index, queries, chunk_size=query_chunk_size)
            patch_maps = distances.mean(axis=1).reshape(len(frame_ids), h, w)
            patch_maps = torch.from_numpy(patch_maps).unsqueeze(1)
            patch_maps = F.interpolate(
                patch_maps,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).numpy()

            patch_score_chunks.append(patch_maps.reshape(len(frame_ids), -1))
            patch_gt_chunks.append(gt.reshape(len(frame_ids), -1))

            for idx, frame_id in enumerate(frame_ids):
                frame_scores.setdefault(frame_id, []).append(float(patch_maps[idx].max()))
                frame_labels[frame_id] = int(labels[idx])

    ordered_frame_ids = sorted(frame_scores.keys())
    image_scores = np.array([max(frame_scores[fid]) for fid in ordered_frame_ids], dtype=np.float64)
    image_labels = np.array([frame_labels[fid] for fid in ordered_frame_ids], dtype=np.int64)
    pixel_scores = np.concatenate(patch_score_chunks, axis=0).reshape(-1)
    pixel_labels = np.concatenate(patch_gt_chunks, axis=0).reshape(-1)
    return image_scores, image_labels, ordered_frame_ids, pixel_scores, pixel_labels


def compute_safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def write_scores_csv(path: Path, frame_ids: list[str], labels: np.ndarray, scores: np.ndarray) -> None:
    order = np.argsort(-scores)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "frame_id", "label", "score"])
        writer.writeheader()
        for rank, idx in enumerate(order):
            writer.writerow(
                {
                    "rank": int(rank),
                    "frame_id": frame_ids[int(idx)],
                    "label": int(labels[int(idx)]),
                    "score": f"{float(scores[int(idx)]):.8f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore baseline on Cam4")
    parser.add_argument("--eval_root", type=str, default=str(_PROJ_ROOT / "rail_mvtec_gt_test"))
    parser.add_argument(
        "--bank_root",
        type=str,
        default=str(_PROJ_ROOT / "rail_mvtec_gt_test_aug_cam4_normal50"),
        help="Root used to build the memory bank",
    )
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target_map_size", type=int, default=32)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--coreset_ratio", type=float, default=0.02)
    parser.add_argument("--coreset_size", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--query_chunk_size", type=int, default=4096)
    parser.add_argument("--random_seed", type=int, default=42)
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
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    eval_root = Path(args.eval_root)
    bank_root = Path(args.bank_root) if args.bank_root else eval_root
    bank_manifest = Path(args.bank_manifest) if args.bank_manifest else default_manifest_path(bank_root, view_id)
    eval_manifest = Path(args.eval_manifest) if args.eval_manifest else default_manifest_path(eval_root, view_id)

    print(f"Using device: {device}")
    print(f"Cam view: Cam{view_id}")
    print(f"Eval root: {eval_root}")
    print(f"Bank root: {bank_root}")
    print(f"Layers: {layers} -> target_map_size={args.target_map_size}")

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

    bank_summary = dataset_summary(bank_dataset)
    eval_summary = dataset_summary(eval_dataset)
    print(f"Memory bank dataset: {bank_summary}")
    print(f"Evaluation dataset: {eval_summary}")

    bank_loader = build_loader(
        bank_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    eval_loader = build_loader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    encoder = ResNet50Encoder(pretrained=True).to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    print("Extracting memory-bank features...")
    memory_full, embedding_dim = collect_memory_features(
        encoder,
        bank_loader,
        device=device,
        layers=layers,
        target_map_size=args.target_map_size,
    )
    print(f"Full memory vectors: {memory_full.shape}")

    memory_bank, projector, coreset_indices = build_memory_bank(
        memory_full,
        coreset_ratio=args.coreset_ratio,
        coreset_size=args.coreset_size,
        projection_dim=args.projection_dim,
        random_seed=args.random_seed,
    )
    projected_dim = int(memory_bank.shape[1])
    print(f"Coreset memory vectors: {memory_bank.shape}")

    nn_index = NearestNeighbors(
        n_neighbors=int(args.k),
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nn_index.fit(memory_bank)

    print("Scoring evaluation set...")
    image_scores, image_labels, frame_ids, pixel_scores, pixel_labels = score_patchcore(
        encoder,
        eval_loader,
        index=nn_index,
        projector=projector,
        device=device,
        layers=layers,
        target_map_size=args.target_map_size,
        img_size=args.img_size,
        query_chunk_size=args.query_chunk_size,
    )

    image_auroc = compute_safe_auroc(image_labels, image_scores)
    pixel_auroc = compute_safe_auroc(pixel_labels.astype(np.int64), pixel_scores)

    print()
    print(f"PatchCore Cam{view_id} Image AUROC: {image_auroc if image_auroc is not None else 'N/A'}")
    print(f"PatchCore Cam{view_id} Pixel AUROC: {pixel_auroc if pixel_auroc is not None else 'N/A'}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = _PROJ_ROOT / "results" / f"patchcore_cam{view_id}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scores_csv:
        scores_csv_path = Path(args.scores_csv)
    else:
        scores_csv_path = out_path.with_name(out_path.stem + "_scores.csv")
    write_scores_csv(scores_csv_path, frame_ids, image_labels, image_scores)

    result = {
        "method": "PatchCore",
        "cam": f"cam{view_id}",
        "view_id": int(view_id),
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "layers": list(layers),
        "target_map_size": int(args.target_map_size),
        "embedding_dim_before_projection": int(embedding_dim),
        "embedding_dim_after_projection": int(projected_dim),
        "projection_dim_arg": int(args.projection_dim),
        "coreset_ratio": float(args.coreset_ratio),
        "coreset_size_arg": int(args.coreset_size),
        "memory_vectors_full": int(memory_full.shape[0]),
        "memory_vectors_coreset": int(memory_bank.shape[0]),
        "memory_coreset_indices_count": int(len(coreset_indices)),
        "k_nn": int(args.k),
        "img_size": int(args.img_size),
        "patch_size": int(args.patch_size),
        "patch_stride": int(args.patch_stride),
        "query_chunk_size": int(args.query_chunk_size),
        "random_seed": int(args.random_seed),
        "eval_root": str(eval_root),
        "bank_root": str(bank_root),
        "bank_selection": str(args.bank_selection),
        "eval_selection": str(args.eval_selection),
        "bank_manifest": str(bank_manifest) if bank_manifest.exists() else "",
        "eval_manifest": str(eval_manifest) if eval_manifest.exists() else "",
        "bank_summary": bank_summary,
        "eval_summary": eval_summary,
        "num_eval_images": int(len(image_labels)),
        "num_eval_abnormal": int((image_labels == 1).sum()),
        "num_eval_normal": int((image_labels == 0).sum()),
        "scores_csv": str(scores_csv_path),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Scores saved to {scores_csv_path}")


if __name__ == "__main__":
    main()
