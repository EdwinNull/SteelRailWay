#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Branch-wise AUROC diagnostics for Cam1/Cam4/Cam5 baseline vs Cam4 PEFT."""

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
import gc
import json
from datetime import datetime
from types import SimpleNamespace

import torch

from scripts.eval.eval_from_ckpt import evaluate_from_args


def find_latest_ckpt(runs_root: Path, view_id: int) -> Path | None:
    patterns = [
        runs_root / f"Cam{view_id}" / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
        runs_root / f"*cam{view_id}_*" / f"best_cam{view_id}.pth",
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in runs_root.glob(str(pattern.relative_to(runs_root))):
            resolved = path.resolve()
            if resolved not in seen:
                candidates.append(path)
                seen.add(resolved)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.parent.stat().st_mtime, p.parent.name), reverse=True)
    return candidates[0]


def load_ckpt_map(value: str | None) -> dict[int, str]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    return {int(k): str(v) for k, v in raw.items() if v}


def resolve_default_cam4_peft_ckpt(peft_root: Path) -> str | None:
    candidates = sorted(
        peft_root.glob("cam4_p1_*/final/final_peft_cam4.pth"),
        key=lambda p: (p.parent.parent.stat().st_mtime, p.parent.parent.name),
        reverse=True,
    )
    if not candidates:
        return None
    return str(candidates[0])


def default_out_dir(depth_peft_ckpt: str | None) -> Path:
    if depth_peft_ckpt:
        peft_path = Path(depth_peft_ckpt)
        if peft_path.exists():
            return peft_path.parents[1] / "diagnostics" / "branch_auc"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/rail_peft") / f"branch_auc_{timestamp}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB/Depth/Fusion AUROC breakdown for Cam1/Cam4/Cam5."
    )
    parser.add_argument("--runs_root", type=str, default="outputs/rail_all")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 4, 5])
    parser.add_argument("--ckpt_map", type=str, default=None,
                        help='可选：JSON 文件或 JSON 字符串，如 {"1": "path/to/best_cam1.pth"}')
    parser.add_argument("--depth_peft_ckpt", type=str, default=None,
                        help="可选：Cam4 final PEFT；不提供时自动从 outputs/rail_peft 查找最新结果")
    parser.add_argument("--train_root", type=str, default="data_20260327")
    parser.add_argument("--test_root", type=str, default="rail_mvtec_gt_test")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--depth_norm", type=str, default="zscore",
                        choices=["zscore", "minmax", "log"])
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_patch", action="store_true", default=True)
    parser.add_argument("--no_patch", action="store_false", dest="use_patch")
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=8)
    parser.add_argument("--channels_last", action="store_true", default=True)
    parser.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    parser.add_argument("--append_log", action="store_true", default=False)
    return parser


def make_eval_args(args, ckpt: str, view_id: int, config_dir: Path, depth_peft_ckpt: str | None):
    return SimpleNamespace(
        ckpt=str(ckpt),
        train_root=args.train_root,
        test_root=args.test_root,
        view_id=view_id,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        depth_norm=args.depth_norm,
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        preload=args.preload,
        preload_workers=args.preload_workers,
        precision=args.precision,
        channels_last=args.channels_last,
        output_log=None,
        append_log=args.append_log,
        scores_csv=None,
        scores_dir=str(config_dir),
        result_json=str(config_dir / "result.json"),
        depth_peft_ckpt=str(depth_peft_ckpt or ""),
        score_source="fusion",
    )


def branch_metric(result: dict, source: str):
    return result.get("auroc_by_source", {}).get(source)


def format_metric(value):
    return "" if value is None else f"{float(value):.8f}"


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "view_id",
        "config_name",
        "ckpt",
        "depth_peft_ckpt",
        "auroc_rgb",
        "auroc_depth",
        "auroc_fusion",
        "delta_rgb_vs_baseline",
        "delta_depth_vs_baseline",
        "delta_fusion_vs_baseline",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_txt(path: Path, args, depth_peft_ckpt: str | None, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Branch-wise AUROC diagnostics\n")
        f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"runs_root: {args.runs_root}\n")
        f.write(f"train_root: {args.train_root}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"views: {' '.join(map(str, args.views))}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"precision: {args.precision}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"num_workers: {args.num_workers}\n")
        f.write(f"use_patch: {args.use_patch}\n")
        f.write(f"depth_peft_ckpt: {depth_peft_ckpt or 'N/A'}\n\n")
        f.write(
            "Columns: view_id, config_name, auroc_rgb, auroc_depth, auroc_fusion, "
            "delta_rgb_vs_baseline, delta_depth_vs_baseline, delta_fusion_vs_baseline\n"
        )
        for row in rows:
            f.write(
                f"Cam{row['view_id']} {row['config_name']}: "
                f"rgb={row['auroc_rgb']}, depth={row['auroc_depth']}, fusion={row['auroc_fusion']}, "
                f"d_rgb={row['delta_rgb_vs_baseline']}, d_depth={row['delta_depth_vs_baseline']}, "
                f"d_fusion={row['delta_fusion_vs_baseline']}\n"
            )


def main():
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    ckpt_map = load_ckpt_map(args.ckpt_map)
    depth_peft_ckpt = args.depth_peft_ckpt or resolve_default_cam4_peft_ckpt(Path("outputs/rail_peft"))
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(depth_peft_ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")
    if depth_peft_ckpt:
        print(f"Depth PEFT: {depth_peft_ckpt}")
    else:
        print("Depth PEFT: N/A (baseline only)")

    configs = [("baseline", None)]
    if depth_peft_ckpt:
        configs.append(("with_cam4peft", depth_peft_ckpt))

    rows = []
    baseline_by_view = {}
    for view_id in args.views:
        ckpt = ckpt_map.get(view_id)
        if not ckpt:
            found = find_latest_ckpt(runs_root, view_id)
            if found is None:
                raise FileNotFoundError(f"Checkpoint not found for Cam{view_id} under {runs_root}")
            ckpt = str(found)

        for config_name, peft_path in configs:
            print("\n" + "=" * 72)
            print(f"Evaluating Cam{view_id} / {config_name}")
            print("=" * 72)
            config_dir = out_dir / f"cam{view_id}_{config_name}"
            config_dir.mkdir(parents=True, exist_ok=True)
            eval_args = make_eval_args(args, ckpt, view_id, config_dir, peft_path)
            result = evaluate_from_args(eval_args)

            row = {
                "view_id": int(view_id),
                "config_name": config_name,
                "ckpt": str(ckpt),
                "depth_peft_ckpt": str(peft_path or ""),
                "auroc_rgb": format_metric(branch_metric(result, "rgb")),
                "auroc_depth": format_metric(branch_metric(result, "depth")),
                "auroc_fusion": format_metric(branch_metric(result, "fusion")),
                "delta_rgb_vs_baseline": "",
                "delta_depth_vs_baseline": "",
                "delta_fusion_vs_baseline": "",
            }
            rows.append(row)
            if config_name == "baseline":
                baseline_by_view[view_id] = {
                    "rgb": branch_metric(result, "rgb"),
                    "depth": branch_metric(result, "depth"),
                    "fusion": branch_metric(result, "fusion"),
                }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for row in rows:
        baseline = baseline_by_view.get(row["view_id"], {})
        for source in ("rgb", "depth", "fusion"):
            metric_text = row[f"auroc_{source}"]
            if not metric_text:
                row[f"delta_{source}_vs_baseline"] = ""
                continue
            metric_value = float(metric_text)
            baseline_value = baseline.get(source)
            if baseline_value is None:
                row[f"delta_{source}_vs_baseline"] = ""
            else:
                row[f"delta_{source}_vs_baseline"] = f"{metric_value - float(baseline_value):.8f}"

    summary_csv = out_dir / "summary.csv"
    summary_txt = out_dir / "summary.txt"
    write_summary_csv(summary_csv, rows)
    write_summary_txt(summary_txt, args, depth_peft_ckpt, rows)

    print("\n" + "=" * 72)
    print("Branch-wise diagnostics finished")
    print("=" * 72)
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
