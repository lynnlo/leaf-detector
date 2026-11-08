"""Export a trained YOLO leaf model into formats the Raspberry Pi can run.

Run this on a desktop or training machine. Export needs ultralytics and PyTorch,
which do not fit on a Raspberry Pi SD card; the Pi only receives the exported
files plus a small runtime.

Typical use:
    python export_for_pi.py models/leaf.pt --sample capture.jpg
    python export_for_pi.py models/leaf.pt --formats ncnn onnx --imgsz 320

The NCNN export is the fastest CPU option on ARM. ONNX is the portable fallback.
TFLite INT8 needs a dataset --data pointing at data.yaml so it can calibrate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def human(num_bytes: float) -> str:
    """Render a byte count as KiB/MiB/GiB."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:6.1f} TiB"


def tree_size(path: Path) -> int:
    """Total size in bytes of a file or of every file inside a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def report(label: str, path: Path) -> None:
    kind = "dir " if path.is_dir() else "file"
    print(f"  {label:<6} {kind} {path.name:<32} {human(tree_size(path))}")


def export(
    model_path: Path,
    formats: list[str],
    imgsz: int,
    half: bool,
    data_yaml: Path | None,
) -> dict[str, Path]:
    """Export the model once per requested format and return the artefacts."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    produced: dict[str, Path] = {}

    for fmt in formats:
        kwargs: dict[str, object] = {"format": fmt, "imgsz": imgsz}
        if half and fmt in {"onnx", "ncnn", "engine"}:
            kwargs["half"] = True
        if fmt == "tflite":
            if data_yaml is None:
                print("  skipping tflite: --data data.yaml is required for int8")
                continue
            kwargs["int8"] = True
            kwargs["data"] = str(data_yaml)

        print(f"exporting {fmt} ...")
        out = model.export(**kwargs)
        produced[fmt] = Path(out)
        report(fmt, produced[fmt])

    return produced


def verify(
    model_path: Path,
    produced: dict[str, Path],
    sample: Path,
    imgsz: int,
    confidence: float,
) -> bool:
    """Compare detections from the source weights against each export."""
    from ultralytics import YOLO

    if not sample.is_file():
        print(f"cannot verify: sample image not found at {sample}")
        return False

    reference = YOLO(str(model_path)).predict(
        str(sample), imgsz=imgsz, conf=confidence, verbose=False
    )[0]
    ref_count = 0 if reference.boxes is None else len(reference.boxes)
    print(f"\nreference {model_path.name}: {ref_count} detections")

    ok = True
    for fmt, path in produced.items():
        result = YOLO(str(path)).predict(
            str(sample), imgsz=imgsz, conf=confidence, verbose=False
        )[0]
        count = 0 if result.boxes is None else len(result.boxes)
        match = count == ref_count
        ok = ok and match
        print(f"  {fmt:<6} {count:3d} detections  {'OK' if match else 'MISMATCH'}")
        if not match:
            print(f"         expected {ref_count}; lower --conf or check imgsz")
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained YOLO model for the Raspberry Pi."
    )
    parser.add_argument("model", help="path to the trained .pt weights")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx"],
        choices=["ncnn", "onnx", "tflite", "openvino"],
        help="export formats (default: onnx, what the Pi runtime loads)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="inference size; 320 for real-time on a Pi 5 CPU (default: 320)",
    )
    parser.add_argument(
        "--half", action="store_true", help="export fp16 weights where supported"
    )
    parser.add_argument(
        "--data", type=Path, help="data.yaml, required for the tflite int8 export"
    )
    parser.add_argument(
        "--sample", type=Path, help="image used to check the exports agree"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="confidence for verification (default: 0.25)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
        help="directory the artefacts are reported into (default: models)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_path = Path(args.model)

    if not model_path.is_file():
        print(f"weights not found: {model_path}", file=sys.stderr)
        return 1

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ultralytics is not installed. Run: pip install ultralytics",
            file=sys.stderr,
        )
        return 1

    print(f"source {model_path}  ({human(model_path.stat().st_size)})  imgsz={args.imgsz}")
    produced = export(
        model_path, args.formats, args.imgsz, args.half, args.data
    )
    if not produced:
        print("nothing was exported", file=sys.stderr)
        return 1

    print("\nartefacts to copy to the Pi:")
    total = 0
    for path in produced.values():
        size = tree_size(path)
        total += size
        print(f"  {path}")
    print(f"  total {human(total)}")

    if args.sample:
        if not verify(model_path, produced, args.sample, args.imgsz, args.conf):
            print("\nverification failed", file=sys.stderr)
            return 2

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())