"""Drive to a detected leaf and sweep it up with a rotating brush.

Three motors across two L298N boards:
  * board 1 drives the two drivetrain motors
  * board 2 drives the single brush motor on channel A

Both boards use the same L298N class from drive.py; the second one is built with
different pins and no second channel. The look-decide-drive cycle itself lives in
capture.drive_loop, shared with capture.py, so the two scripts cannot drift apart.

Motors only move when --drive is passed.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from capture import Camera, DriveParams, LoopOptions, drive_loop
from drive import L298N
from leaf_detector import (
    DEFAULT_CLASSES,
    DEFAULT_WORKFLOW_ID,
    DEFAULT_WORKSPACE,
    RoboflowError,
    build_detector,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "leaf.onnx"

# Second L298N board: one brush motor on channel A. Channel B is not used, so
# its three pins (BCM 13, 19, 26) stay free. Remove this board's ENA jumper.
BRUSH_ENA = 12
BRUSH_IN1 = 16
BRUSH_IN2 = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive to a detected leaf and sweep it up."
    )
    parser.add_argument(
        "--backend",
        choices=["roboflow", "onnx"],
        default="roboflow",
        help="roboflow calls the hosted workflow; onnx runs a local export "
        "(default: roboflow)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key",
        help="Roboflow API key; defaults to the ROBOFLOW_API_KEY environment variable",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--workflow-id", default=DEFAULT_WORKFLOW_ID)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--speed", type=int, default=40, help="drive speed percent")
    parser.add_argument(
        "--brush-speed", type=int, default=100, help="brush speed percent"
    )
    parser.add_argument(
        "--close",
        type=float,
        default=0.55,
        help="box height / frame height that counts as close enough to sweep",
    )
    parser.add_argument(
        "--sweep-time",
        type=float,
        default=1.5,
        help="seconds spent driving over the leaf",
    )
    parser.add_argument(
        "--hold", type=float, default=0.4, help="seconds per steering command"
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cast around when no leaf is visible instead of standing still "
        "(default: on; use --no-search to wait in place)",
    )
    parser.add_argument(
        "--search-turn",
        type=float,
        default=5.0,
        help="seconds of rotating in place before creeping forward (default: 5)",
    )
    parser.add_argument(
        "--max-sweeps", type=int, default=3,
        help="brush passes before giving up on one leaf and moving on; "
        "0 never gives up (default: 3)",
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="loop iterations, 0 for unlimited (default: 0)"
    )
    parser.add_argument("--drive", action="store_true", help="energise the motors")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--save", metavar="PATH")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="correct for the camera's mounting: vertical swaps top/bottom "
        "(upside-down mount), horizontal mirrors left/right, both rotates 180 "
        "degrees, none leaves the frame as captured (default: vertical)",
    )
    parser.add_argument(
        "--k-angle", type=float, default=DriveParams.k_angle, help="turn gain"
    )
    parser.add_argument(
        "--k-linear",
        type=float,
        default=DriveParams.k_linear,
        help="scales forward speed, so a centred target is approached at "
        "k-linear x --speed (default: 0.9)",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")
    if not 1 <= args.brush_speed <= 100:
        parser.error("--brush-speed must be between 1 and 100")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if not 0.0 < args.close <= 1.0:
        parser.error("--close must be greater than 0 and at most 1")
    if args.sweep_time < 0 or args.hold < 0:
        parser.error("--sweep-time and --hold cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.backend == "onnx":
        if not args.model.is_file():
            print(f"ONNX model not found: {args.model}", file=sys.stderr)
            return 1
    elif not (args.api_key or os.environ.get("ROBOFLOW_API_KEY")):
        print(
            "no Roboflow API key: set ROBOFLOW_API_KEY or pass --api-key",
            file=sys.stderr,
        )
        return 1

    try:
        detector = build_detector(
            args.backend,
            model_path=args.model,
            api_key=args.api_key,
            workspace=args.workspace,
            workflow_id=args.workflow_id,
            class_names=tuple(args.classes),
            confidence=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            timeout=args.timeout,
        )
    except ImportError as exc:
        print(f"a dependency is missing: {exc}", file=sys.stderr)
        return 1
    except RoboflowError as exc:
        print(f"could not build the detector: {exc}", file=sys.stderr)
        return 1

    params = DriveParams(
        k_angle=args.k_angle,
        k_linear=args.k_linear,
        v_max=args.speed / 100.0,
        stop_height_ratio=args.close,
    )

    camera = Camera(args.width, args.height, args.flip)
    if args.backend == "roboflow":
        source = f"workflow {args.workflow_id[:44]}"
    else:
        source = args.model.name
    print(f"camera backend: {camera.backend}   detector: {args.backend} ({source})")

    options = LoopOptions(
        steps=args.steps,
        hold=args.hold,
        search=args.search,
        search_turn=args.search_turn,
        max_sweeps=args.max_sweeps,
        sweep_time=args.sweep_time,
        brush_speed=args.brush_speed / 100.0,
        preview=args.preview,
        save=args.save,
    )

    with ExitStack() as stack:
        if args.drive:
            robot = L298N(stack)
            brush = L298N(
                stack,
                ena=BRUSH_ENA,
                in1=BRUSH_IN1,
                in2=BRUSH_IN2,
                enb=None,
                in3=None,
                in4=None,
            )
        else:
            robot = brush = None
            print("dry run: motors are not energised (pass --drive to move)")

        return drive_loop(camera, detector, params, options, driver=robot, brush=brush)


if __name__ == "__main__":
    raise SystemExit(main())