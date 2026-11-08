"""Drive to a detected leaf and sweep it up with two rotating brushes.

Four motors across two L298N boards:
  * board 1 drives the two drivetrain motors
  * board 2 drives the two brush motors

Both boards use the same L298N class from drive.py; the second one is just
constructed with different pins. The loop is deliberately short: steer toward
the nearest leaf until it fills enough of the frame, then drive over it with the
brushes spinning for a moment.

Motors only move when --drive is passed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from capture import Camera, DriveParams, annotate, choose_target, compute_command
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

# Second L298N board: brush motor A and brush motor B. Remove both jumpers.
BRUSH_ENA = 12
BRUSH_IN1 = 16
BRUSH_IN2 = 20
BRUSH_ENB = 13
BRUSH_IN3 = 19
BRUSH_IN4 = 26

# Set True if the two brushes face each other and must counter-rotate.
BRUSH_OPPOSED = False


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
        "--max-sweeps", type=int, default=5, help="give up after this many sweeps"
    )
    parser.add_argument("--steps", type=int, default=40, help="loop iterations")
    parser.add_argument("--drive", action="store_true", help="energise the motors")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--save", metavar="PATH")
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

    speed = args.speed / 100.0
    brush_speed = args.brush_speed / 100.0
    brush_second = -brush_speed if BRUSH_OPPOSED else brush_speed
    params = DriveParams(v_max=speed, stop_height_ratio=args.close)

    camera = Camera(args.width, args.height)
    if args.backend == "roboflow":
        source = f"workflow {args.workflow_id[:44]}"
    else:
        source = args.model.name
    print(f"camera backend: {camera.backend}   detector: {args.backend} ({source})")

    sweeps = 0
    last_frame = None

    with ExitStack() as stack:
        if args.drive:
            robot = L298N(stack)
            brush = L298N(
                stack,
                ena=BRUSH_ENA,
                in1=BRUSH_IN1,
                in2=BRUSH_IN2,
                enb=BRUSH_ENB,
                in3=BRUSH_IN3,
                in4=BRUSH_IN4,
            )
        else:
            robot = brush = None
            print("dry run: motors are not energised (pass --drive to move)")

        try:
            for step in range(args.steps):
                frame = camera.read()
                detections = detector.detect(frame)
                target = choose_target(detections)
                left, right, note = compute_command(
                    target, frame.shape[1], frame.shape[0], params
                )

                if target is None:
                    # Leaf gone: stop and wait for the next look.
                    state = "no leaf - stopped"
                    left = right = 0.0
                    sweeping = False

                elif "arrived" in note:
                    # Close enough: drive over it with the brushes spinning.
                    sweeps += 1
                    state = f"sweeping ({sweeps}/{args.max_sweeps})"
                    sweeping = True

                else:
                    state = "approaching"
                    sweeping = False

                print(f"[{step + 1}/{args.steps}] {state:<22} {note}")

                last_frame = annotate(frame, detections, target, state)
                if args.preview:
                    cv2.imshow("leaf-dec", last_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if sweeping:
                    if robot is not None:
                        brush.drive(brush_speed, brush_second)
                        robot.drive(speed * 0.6, speed * 0.6)
                        time.sleep(args.sweep_time)
                        robot.stop()
                        brush.stop()
                    if sweeps >= args.max_sweeps:
                        print("reached --max-sweeps, stopping")
                        break
                    continue

                if robot is not None and (left or right):
                    robot.drive(left, right)
                    time.sleep(args.hold)
                    robot.stop()

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if robot is not None:
                robot.stop()
                brush.stop()
            camera.close()
            detector.close()
            if args.preview:
                cv2.destroyAllWindows()
            if args.save and last_frame is not None:
                cv2.imwrite(args.save, last_frame)
                print(f"saved {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())