"""Train the leaf detector from the downloaded dataset and export it for the Pi.

Run this on a machine with a GPU. It trains yolo11n on dataset/data.yaml, keeps
the best weights in models/leaf.pt, then exports NCNN and ONNX at a size the
Raspberry Pi can run in real time.

    python train_and_export.py
    python train_and_export.py --epochs 100 --device 0 --export-imgsz 320
    python train_and_export.py --skip-train --weights models/leaf.pt

The Pi should receive only the exported files: PyTorch does not fit on a small
SD card, and the NCNN export is roughly twice as fast on ARM CPUs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "dataset" / "data.yaml"
MODELS = HERE / "models"


def human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:6.1f} TiB"


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_dataset(data: Path) -> bool:
    if not data.is_file():
        print(f"dataset config not found: {data}", file=sys.stderr)
        print("download it with the Roboflow SDK or from the Universe page", file=sys.stderr)
        return False

    root = data.parent
    counts = {}
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        n_img = len(list(images.glob("*"))) if images.is_dir() else 0
        n_lbl = len(list(labels.glob("*.txt"))) if labels.is_dir() else 0
        counts[split] = n_img
        print(f"  {split:6} images={n_img:5} labels={n_lbl:5}")
        if n_img == 0:
            print(f"split '{split}' has no images", file=sys.stderr)
            return False
        if n_img != n_lbl:
            print(f"warning: {split} has {n_img} images but {n_lbl} labels", file=sys.stderr)
    return True


def count_instances(data: Path) -> dict[str, int]:
    """Tally labelled instances per class id to surface class imbalance."""
    totals: dict[str, int] = {}
    for f in data.parent.glob("*/labels/*.txt"):
        for line in f.read_text(errors="replace").splitlines():
            parts = line.split()
            if parts:
                totals[parts[0]] = totals.get(parts[0], 0) + 1
    return dict(sorted(totals.items()))


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        # Leaves have no canonical orientation, so full rotation is valid.
        degrees=180,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished without best.pt in {result.save_dir}")
    return best


def export(best: Path, args: argparse.Namespace) -> dict[str, Path]:
    from ultralytics import YOLO

    model = YOLO(str(best))
    produced: dict[str, Path] = {}
    for fmt in args.formats:
        print(f"exporting {fmt} at imgsz={args.export_imgsz} ...")
        out = Path(model.export(format=fmt, imgsz=args.export_imgsz))
        produced[fmt] = out
        print(f"  {fmt:<6} {out}  {human(tree_size(out))}")
    return produced


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the leaf detector and export it for the Raspberry Pi."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' otherwise")
    parser.add_argument("--name", default="leaf", help="run name under runs/")
    parser.add_argument(
        "--export-imgsz",
        type=int,
        default=320,
        help="inference size baked into the export; 320 keeps real time on a Pi 5",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx"],
        choices=["ncnn", "onnx", "tflite", "openvino"],
        help="export formats (default: onnx, what the Pi runtime loads)",
    )
    parser.add_argument("--skip-train", action="store_true", help="export existing weights")
    parser.add_argument("--weights", type=Path, help="weights used with --skip-train")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics is not installed. Run: pip install ultralytics", file=sys.stderr)
        return 1

    MODELS.mkdir(exist_ok=True)

    if args.skip_train:
        best = args.weights or (MODELS / "leaf.pt")
        if not best.is_file():
            print(f"weights not found: {best}", file=sys.stderr)
            return 1
    else:
        print(f"dataset {args.data}")
        if not check_dataset(args.data):
            return 1

        totals = count_instances(args.data)
        print(f"  instances per class id: {totals}")
        if len(totals) > 1 and min(totals.values()) < 200:
            print(
                "  warning: one class has very few instances; expect weak detection "
                "for it and treat its boxes as a bonus rather than a reliable target"
            )

        best = train(args)

    target = MODELS / "leaf.pt"
    if best.resolve() != target.resolve():
        shutil.copy2(best, target)
    print(f"\nkept weights at {target}  ({human(target.stat().st_size)})")

    produced = export(target, args)

    print("\ncopy these to the Pi (no PyTorch needed there):")
    for fmt, path in produced.items():
        print(f"  {path}")
    print("\nthen point the script at the export, e.g. on the Pi:")
    print(f"  python3 capture.py --model {next(iter(produced.values())).name} --imgsz {args.export_imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())