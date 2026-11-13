"""CamRec YOLO training entry with optional max-performance GPU presets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO for CamRec multi-class intelligent camera detection."
    )
    parser.add_argument(
        "--data",
        default=str(Path("datasets") / "camrec_all_classes_balanced" / "dataset.yaml"),
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Model weights or model config, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for Ultralytics AutoBatch (recommended on 16GB GPU).",
    )
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0 or cpu.")
    parser.add_argument(
        "--project",
        default=str(Path("runs") / "train"),
        help="Project output directory.",
    )
    parser.add_argument("--name", default="yolo11s_camrec_maxperf", help="Run name.")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 = auto (min(12, cpu_count-2) on Windows).",
    )
    parser.add_argument(
        "--cache",
        default="disk",
        choices=["false", "True", "ram", "disk"],
        help="Dataset cache: false | True/ram (needs free RAM) | disk (safer on 32GB host).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic Mixed Precision (default: on).",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile (can help throughput; first epoch slower; may be unstable).",
    )
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cosine LR schedule (default: on for long runs).",
    )
    parser.add_argument(
        "--multi-scale",
        type=float,
        default=0.0,
        help="Multi-scale training scale range (0 disables; e.g. 0.5 is heavy).",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer: auto / AdamW / SGD / etc.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deterministic ops (slower). Default off for max speed.",
    )
    parser.add_argument(
        "--perf",
        choices=["default", "max"],
        default="max",
        help="Preset: max enables TF32/cudnn.benchmark and aggressive defaults.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use (debug only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume last interrupted run in the same project/name.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing run directory.",
    )
    return parser.parse_args()


def validate_dataset_yaml(data_path: Path) -> Path:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    base_dir = data_path.parent
    root_value = payload.get("path", ".")
    dataset_root = Path(root_value)
    if not dataset_root.is_absolute():
        dataset_root = (base_dir / root_value).resolve()
    else:
        dataset_root = dataset_root.resolve()

    for split_name in ("train", "val"):
        split_value = payload.get(split_name)
        if not split_value:
            raise ValueError(f"Dataset yaml is missing '{split_name}'")
        split_dir = (dataset_root / split_value).resolve()
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split path not found: {split_dir}")

    return data_path.resolve()


def resolve_cache(cache_arg: str):
    if cache_arg in ("false", "False", "0", "none", "None"):
        return False
    if cache_arg in ("True", "true", "ram"):
        return "ram"
    if cache_arg == "disk":
        return "disk"
    return False


def resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    # Windows multiprocessing is heavier; leave headroom for OS + training process.
    cpu = os.cpu_count() or 8
    return max(4, min(12, cpu - 2))


def enable_max_gpu_mode(device: str) -> None:
    """Push PyTorch/cuDNN toward throughput (safe for training quality)."""
    if not torch.cuda.is_available():
        print("[perf] CUDA not available; skipping GPU max mode.")
        return

    # Prefer GPU 0 when device is numeric.
    if str(device).isdigit():
        torch.cuda.set_device(int(device))

    # TF32 on Ampere/Ada/Blackwell: large matmul speedup, negligible detect impact.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as exc:  # pragma: no cover - version dependent
        print(f"[perf] set_float32_matmul_precision skipped: {exc}")

    # Autotune best convolution algorithms for fixed shapes.
    torch.backends.cudnn.benchmark = True
    # Prefer speed over strict reproducibility.
    torch.backends.cudnn.deterministic = False

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    print(
        f"[perf] GPU0={props.name}  VRAM={total_gb:.1f} GiB  "
        f"capability={props.major}.{props.minor}  "
        f"tf32=on  cudnn.benchmark=on"
    )


def print_train_plan(args: argparse.Namespace, workers: int, cache) -> None:
    print("========== CamRec train plan ==========")
    print(f"  model     : {args.model}")
    print(f"  data      : {args.data}")
    print(f"  imgsz     : {args.imgsz}")
    print(f"  batch     : {args.batch}  (-1 = AutoBatch fill VRAM)")
    print(f"  epochs    : {args.epochs}  patience={args.patience}")
    print(f"  device    : {args.device}")
    print(f"  workers   : {workers}")
    print(f"  cache     : {cache}")
    print(f"  amp       : {args.amp}")
    print(f"  compile   : {args.compile}")
    print(f"  cos_lr    : {args.cos_lr}")
    print(f"  multi_scale: {args.multi_scale}")
    print(f"  deterministic: {args.deterministic}")
    print(f"  perf      : {args.perf}")
    print(f"  project/name: {args.project}/{args.name}")
    print("=======================================")


def main() -> None:
    args = parse_args()
    data_path = validate_dataset_yaml(Path(args.data))
    workers = resolve_workers(args.workers)
    cache = resolve_cache(str(args.cache))

    if args.perf == "max":
        enable_max_gpu_mode(args.device)
        # Prefer non-deterministic high throughput unless user forces otherwise.
        if args.deterministic is False or args.perf == "max":
            # keep argparse value; max preset already defaults deterministic=False
            pass

    print_train_plan(args, workers, cache)

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        workers=workers,
        cache=cache,
        amp=args.amp,
        compile=args.compile,
        cos_lr=args.cos_lr,
        multi_scale=args.multi_scale,
        close_mosaic=args.close_mosaic,
        optimizer=args.optimizer,
        deterministic=args.deterministic,
        fraction=args.fraction,
        exist_ok=args.exist_ok,
        # Plots/val keep quality signal; disable only if pure throughput profiling.
        plots=True,
        val=True,
        pretrained=True,
        seed=0,
    )
    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)
    print("[done] training finished.")
    return results


if __name__ == "__main__":
    main()
