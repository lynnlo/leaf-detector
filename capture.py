"""Detect leaves from the Raspberry Pi camera and steer two motors toward them.

Each step captures a frame, runs the leaf detector, picks the nearest detection,
converts the box's horizontal offset into differential motor speeds, holds that
command briefly, stops, and repeats. Motor output goes through the L298N class in
drive.py, so the pinout lives in one place.

Inference runs through ONNX Runtime (leaf_detector.py), so the Pi needs neither
ultralytics nor PyTorch. The ONNX model is produced on a desktop machine with
export_for_pi.py and copied across.

Motors only move when --drive is passed. Without it the script prints the command
it would send and leaves the wheels still.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drive import L298N
from leaf_detector import DEFAULT_CLASSES, Detection, OnnxLeafDetector

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "leaf.onnx"


@dataclass(frozen=True)
class DriveParams:
    """Tuning for turning a detection box into wheel speeds."""

    k_angle: float = 1.2
    k_linear: float = 0.9
    v_max: float = 0.6
    center_tolerance: float = 0.08
    stop_height_ratio: float = 0.6


class Camera:
    """Source of BGR frames, preferring picamera2 and falling back to rpicam-still."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._picam = None
        self.backend = "rpicam-still"

        try:
            from picamera2 import Picamera2
        except ImportError:
            return

        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(0.5)  # Let auto-exposure settle before the first frame.
        self._picam = picam
        self.backend = "picamera2"

    def read(self) -> np.ndarray:
        if self._picam is not None:
            return self._picam.capture_array()

        result = subprocess.run(
            [
                "rpicam-still",
                "--nopreview",
                "--encoding",
                "jpg",
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--timeout",
                "200",
                "--output",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("rpicam-still produced no decodable frame")
        return frame

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None


def choose_target(detections: list[Detection]) -> Detection | None:
    """Return the nearest detection.

    On a downward-tilted camera a box whose bottom edge sits lower in the image
    is closer to the robot, so the lowest bottom edge wins. Box area breaks ties.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: (d.y2, d.area))


def compute_command(
    detection: Detection | None,
    frame_width: int,
    frame_height: int,
    params: DriveParams,
) -> tuple[float, float, str]:
    """Turn a detection box into (left, right, reason) wheel speeds in -1..1.

    A positive horizontal offset means the target sits right of centre; the left
    wheel is then driven faster than the right, which rotates the chassis toward
    the target. Forward speed falls off as the target moves off-centre so the
    robot turns before it drives.
    """
    if detection is None:
        return 0.0, 0.0, "no leaf in view"

    closeness = detection.height / max(1.0, float(frame_height))
    if closeness >= params.stop_height_ratio:
        return 0.0, 0.0, f"arrived (box fills {closeness:.0%} of frame height)"

    half_width = max(1.0, frame_width / 2.0)
    offset = (detection.center_x - frame_width / 2.0) / half_width
    offset = max(-1.0, min(1.0, offset))

    turn = params.k_angle * offset
    forward = params.v_max * (1.0 - min(abs(offset), 1.0))

    if abs(offset) <= params.center_tolerance:
        direction = "straight"
    elif offset > 0:
        direction = "right"
    else:
        direction = "left"

    left = max(-1.0, min(1.0, forward + turn))
    right = max(-1.0, min(1.0, forward - turn))
    reason = (
        f"{direction} offset={offset:+.2f} "
        f"closeness={closeness:.0%} conf={detection.confidence:.2f}"
    )
    return left, right, reason


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    target: Detection | None,
    note: str,
) -> np.ndarray:
    """Draw every detection, highlighting the chosen target, plus a status line."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for det in detections:
        chosen = det is target
        colour = (0, 255, 0) if chosen else (0, 165, 255)
        thickness = 3 if chosen else 1
        cv2.rectangle(
            canvas,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            colour,
            thickness,
        )
        cv2.putText(
            canvas,
            f"{det.label} {det.confidence:.2f}",
            (int(det.x1), max(12, int(det.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    cv2.line(
        canvas,
        (width // 2, height - 40),
        (width // 2, height),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        canvas,
        f"leaves={len(detections)} {note}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect leaves from the Pi camera and steer toward them."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX weights (default: {DEFAULT_MODEL.relative_to(HERE)})",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="class names in the model's output order",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35, help="detection confidence (default: 0.35)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="must match the size the model was exported at (default: 320)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="capture width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="capture height (default: 480)"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="detect-and-drive cycles (default: 10)"
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=0.4,
        help="seconds per drive command (default: 0.4)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=45,
        metavar="PERCENT",
        help="cap on wheel speed, 1-100 (default: 45)",
    )
    parser.add_argument(
        "--k-angle", type=float, default=DriveParams.k_angle, help="turn gain"
    )
    parser.add_argument(
        "--k-linear", type=float, default=DriveParams.k_linear, help="forward gain"
    )
    parser.add_argument(
        "--stop-height",
        type=float,
        default=DriveParams.stop_height_ratio,
        help="stop once the box fills this fraction of frame height (default: 0.6)",
    )
    parser.add_argument(
        "--drive",
        action="store_true",
        help="energise the motors; without this the command is only printed",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="rotate in place when no leaf is visible instead of stopping",
    )
    parser.add_argument("--preview", action="store_true", help="show an annotated window")
    parser.add_argument("--save", metavar="PATH", help="write the annotated last frame")
    args = parser.parse_args(argv)

    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model.is_file():
        print(f"ONNX model not found: {args.model}", file=sys.stderr)
        print(
            "Export it on a desktop machine with:\n"
            "  python export_for_pi.py models/leaf.pt --formats onnx --imgsz 320\n"
            "then copy the .onnx file into models/.",
            file=sys.stderr,
        )
        return 1

    try:
        detector = OnnxLeafDetector(
            args.model,
            class_names=tuple(args.classes),
            confidence=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
        )
    except ImportError:
        print("onnxruntime is not installed. Run: pip install onnxruntime", file=sys.stderr)
        return 1

    params = DriveParams(
        k_angle=args.k_angle,
        k_linear=args.k_linear,
        v_max=args.speed / 100.0,
        stop_height_ratio=args.stop_height,
    )

    camera = Camera(args.width, args.height)
    print(f"camera backend: {camera.backend}   model: {args.model.name}")

    with ExitStack() as stack:
        driver = L298N(stack) if args.drive else None
        if driver is None:
            print("dry run: motors are not energised (pass --drive to move)")

        search_speed = params.v_max * 0.4
        last_frame: np.ndarray | None = None

        try:
            for step in range(args.steps):
                frame = camera.read()
                detections = detector.detect(frame)
                target = choose_target(detections)
                left, right, note = compute_command(
                    target, frame.shape[1], frame.shape[0], params
                )

                if target is None and args.search:
                    left, right = -search_speed, search_speed
                    note = "searching: rotating in place"

                print(
                    f"[{step + 1}/{args.steps}] {note} "
                    f"left={left:+.2f} right={right:+.2f}"
                )

                last_frame = annotate(frame, detections, target, note)
                if args.preview:
                    cv2.imshow("leaf-dec", last_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if driver is not None and (left or right):
                    driver.drive(left, right)
                    time.sleep(args.hold)
                    driver.stop()

                if target is not None and left == 0.0 and right == 0.0:
                    print("target reached")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if driver is not None:
                driver.stop()
            camera.close()
            if args.preview:
                cv2.destroyAllWindows()
            if args.save and last_frame is not None:
                cv2.imwrite(args.save, last_frame)
                print(f"saved {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())