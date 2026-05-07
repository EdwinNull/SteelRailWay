#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility wrapper for legacy PatchCore Cam4 entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore baseline compatibility wrapper")
    parser.add_argument("--train_root", type=str, default="/data1/Leaddo_data/20260327-resize512")
    parser.add_argument("--test_root", "--eval_root", dest="test_root", type=str, default="/home/root123/LF/WYM/SteelRailWay/rail_mvtec_gt_test")
    parser.add_argument("--cam", type=str, default="cam4")
    parser.add_argument("--view_id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=900)
    parser.add_argument("--patch_stride", type=int, default=850)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--preload_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--scores_csv", type=str, default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--train_sample_num", type=int, default=1500)
    parser.add_argument("--train_val_test_split", type=float, nargs=3, default=[0.9, 0.1, 0.0])
    parser.add_argument("--sampling_mode", type=str, default="uniform_time")
    parser.add_argument("--projection_dim", "--patchcore_projection_dim", dest="projection_dim", type=int, default=128)
    parser.add_argument("--coreset_ratio", "--patchcore_coreset_ratio", dest="coreset_ratio", type=float, default=0.01)
    parser.add_argument("--coreset_size", "--patchcore_coreset_size", dest="coreset_size", type=int, default=0)
    parser.add_argument("--k", "--patchcore_k", dest="k", type=int, default=5)
    parser.add_argument("--query_chunk_size", "--patchcore_query_chunk_size", dest="query_chunk_size", type=int, default=2048)
    parser.add_argument("--layers", "--patchcore_layers", dest="layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--target_map_size", "--patchcore_target_map_size", dest="target_map_size", type=int, default=32)
    args = parser.parse_args()

    view_id = args.view_id
    if view_id is None:
        cam = str(args.cam).strip().lower()
        view_id = int(cam[3:]) if cam.startswith("cam") else int(cam)

    cmd = [
        sys.executable,
        str(PROJ_ROOT / "scripts" / "baselines" / "run_rgb_baseline.py"),
        "--method",
        "patchcore",
        "--train_root",
        args.train_root,
        "--test_root",
        args.test_root,
        "--view_id",
        str(view_id),
        "--img_size",
        str(args.img_size),
        "--train_sample_num",
        str(args.train_sample_num),
        "--train_val_test_split",
        *[str(x) for x in args.train_val_test_split],
        "--sampling_mode",
        args.sampling_mode,
        "--patch_size",
        str(args.patch_size),
        "--patch_stride",
        str(args.patch_stride),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--preload_workers",
        str(args.preload_workers),
        "--patchcore_projection_dim",
        str(args.projection_dim),
        "--patchcore_coreset_ratio",
        str(args.coreset_ratio),
        "--patchcore_coreset_size",
        str(args.coreset_size),
        "--patchcore_k",
        str(args.k),
        "--patchcore_query_chunk_size",
        str(args.query_chunk_size),
        "--patchcore_target_map_size",
        str(args.target_map_size),
        "--patchcore_layers",
        *[str(x) for x in args.layers],
    ]
    if args.preload:
        cmd.append("--preload")
    if args.smoke:
        cmd.append("--smoke")
    if args.output:
        cmd.extend(["--result_json", args.output])
    if args.scores_csv:
        cmd.extend(["--scores_csv", args.scores_csv])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
