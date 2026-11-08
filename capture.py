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
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import cv2
import numpy as np

from drive import L298N
from leaf_detector import (
    DEFAULT_CLASSES,
    DEFAULT_WORKFLOW_ID,
    DEFAULT_WORKSPACE,
    Detection,
    RoboflowError,
    build_detector,
)

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
            print(
                "picamera2 is not installed; using the rpicam-still fallback",
                file=sys.stderr,
            )
            return

        # libcamera reports missing or busy cameras as a plain RuntimeError, so a
        # broad catch here is deliberate: the point is to fall back rather than
        # abort with a traceback.
        try:
            picam = Picamera2()
            config = picam.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(0.5)  # Let auto-exposure settle before the first frame.
        except Exception as exc:
            print(
                f"picamera2 could not start ({exc}); using the rpicam-still fallback",
                file=sys.stderr,
            )
            return

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
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            lines = result.stderr.decode("utf-8", "replace").strip().splitlines()
            detail = lines[-1] if lines else "no output"
            raise RuntimeError(
                f"rpicam-still failed with exit code {result.returncode}: {detail}"
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
    robot turns before it drives, and is scaled by k_linear so a centred leaf is
    approached at k_linear * v_max.
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
    forward = params.k_linear * params.v_max * (1.0 - min(abs(offset), 1.0))

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


@dataclass
class LoopOptions:
    """Tunables for drive_loop, gathered from whichever script is running."""

    steps: int = 0
    hold: float = 0.4
    search: bool = True
    search_turn: float = 5.0
    max_sweeps: int = 0
    sweep_time: float = 1.5
    brush_speed: float = 1.0
    preview: bool = False
    save: str | Path | None = None
    stop_on_arrival: bool = False


def drive_loop(
    camera,
    detector,
    params: DriveParams,
    options: LoopOptions,
    driver: L298N | None = None,
    brush: L298N | None = None,
) -> int:
    """Look for leaves, steer at them, and handle them until stopped.

    One cycle reads a frame, runs the detector, picks the nearest leaf and turns
    it into wheel speeds. With nothing in view the robot casts around: it
    rotates in place, and creeps forward when a full sweep of the view finds
    nothing, which takes it onto fresh ground instead of spinning on one spot.

    On reaching a leaf it sweeps when a brush is fitted, then backs off so the
    same leaf is not immediately retargeted. With no brush it simply backs off.

    Runs until the step budget is exhausted or the operator interrupts, so
    steps=0 keeps going indefinitely. Returns a process exit code.
    """
    search_speed = params.v_max * 0.4
    total = str(options.steps) if options.steps > 0 else "inf"
    steps = range(options.steps) if options.steps > 0 else count()

    sweeps = 0
    search_started: float | None = None
    last_frame = None

    try:
        for step in steps:
            frame = camera.read()
            detections = detector.detect(frame)
            target = choose_target(detections)
            left, right, note = compute_command(
                target, frame.shape[1], frame.shape[0], params
            )

            # Seeing a leaf ends the hunt, so restart the rotation clock.
            if target is not None:
                search_started = None

            sweeping = False

            if target is None:
                # A new leaf gets a fresh budget of brush passes.
                sweeps = 0
                if not options.search:
                    state = "no leaf - stopped"
                    left = right = 0.0
                elif search_started is None:
                    search_started = time.monotonic()
                    state = "searching: rotating"
                    left, right = -search_speed, search_speed
                elif time.monotonic() - search_started >= options.search_turn:
                    state = "searching: creeping forward"
                    left = right = search_speed * 1.2
                    search_started = time.monotonic()
                else:
                    state = "searching: rotating"
                    left, right = -search_speed, search_speed

            elif "arrived" in note:
                sweeps += 1
                if options.stop_on_arrival:
                    state = "arrived"
                else:
                    sweeping = True
                    state = (
                        f"sweeping ({sweeps}/{options.max_sweeps})"
                        if options.max_sweeps > 0
                        else f"sweeping ({sweeps})"
                    )

            else:
                state = "approaching"

            print(
                f"[{step + 1}/{total}] {state:<24} {note}  "
                f"L={left:+.2f} R={right:+.2f}"
            )

            last_frame = annotate(frame, detections, target, state)
            if options.preview:
                cv2.imshow("leaf-dec", last_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if options.stop_on_arrival and state == "arrived":
                print("target reached")
                break

            if sweeping:
                limit = options.max_sweeps
                reached_limit = limit > 0 and sweeps >= limit

                if driver is not None and brush is not None:
                    brush.drive(options.brush_speed)
                    driver.drive(params.v_max * 0.6, params.v_max * 0.6)
                    time.sleep(options.sweep_time)
                    driver.stop()
                    brush.stop()

                if reached_limit:
                    print(f"gave up on this leaf after {sweeps} sweeps; moving on")
                    if driver is not None:
                        # Back off, then pivot, so the leaf drops out of frame
                        # instead of being swept again straight away.
                        driver.drive(-params.v_max * 0.6, -params.v_max * 0.6)
                        time.sleep(options.sweep_time * 0.4)
                        driver.drive(-params.v_max * 0.5, params.v_max * 0.5)
                        time.sleep(options.sweep_time * 0.6)
                        driver.stop()
                    sweeps = 0
                continue

            if driver is not None and (left or right):
                driver.drive(left, right)
                time.sleep(options.hold)
                driver.stop()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if driver is not None:
            driver.stop()
        if brush is not None:
            brush.stop()
        camera.close()
        detector.close()
        if options.preview:
            cv2.destroyAllWindows()
        if options.save and last_frame is not None:
            cv2.imwrite(str(options.save), last_frame)
            print(f"saved {options.save}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect leaves from the Pi camera and steer toward them."
    )
    parser.add_argument(
        "--backend",
        choices=["roboflow", "onnx"],
        default="roboflow",
        help="roboflow calls the hosted workflow over the network; onnx runs a "
        "local export with no network (default: roboflow)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX weights, used by --backend onnx (default: {DEFAULT_MODEL.relative_to(HERE)})",
    )
    parser.add_argument(
        "--api-key",
        help="Roboflow API key; defaults to the ROBOFLOW_API_KEY environment variable",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"Roboflow workspace slug (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--workflow-id",
        default=DEFAULT_WORKFLOW_ID,
        help="Roboflow workflow slug",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="seconds to wait for one workflow call (default: 15)",
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
        "--steps", type=int, default=10, help="detect-and-drive cycles, 0 for unlimited (default: 10)"
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
        "--k-linear",
        type=float,
        default=DriveParams.k_linear,
        help="scales forward speed, so a centred target is approached at "
        "k-linear x --speed (default: 0.9)",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cast around when no leaf is visible instead of stopping "
        "(default: on; use --no-search to wait in place)",
    )
    parser.add_argument(
        "--search-turn",
        type=float,
        default=5.0,
        help="seconds of rotating in place before creeping forward (default: 5)",
    )
    parser.add_argument(
        "--stop-on-arrival",
        action="store_true",
        help="stop at the first leaf reached instead of backing off and moving on",
    )
    parser.add_argument("--preview", action="store_true", help="show an annotated window")
    parser.add_argument("--save", metavar="PATH", help="write the annotated last frame")
    args = parser.parse_args(argv)

    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if not 0.0 < args.stop_height <= 1.0:
        parser.error("--stop-height must be greater than 0 and at most 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.backend == "onnx":
        if not args.model.is_file():
            print(f"ONNX model not found: {args.model}", file=sys.stderr)
            print(
                "Export it on a desktop machine with:\n"
                "  python export_for_pi.py models/leaf.pt --formats onnx --imgsz 320\n"
                "then copy the .onnx file into models/.",
                file=sys.stderr,
            )
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
        stop_height_ratio=args.stop_height,
    )

    camera = Camera(args.width, args.height)
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
        # No brush on this build, so one pass counts as "handled" and the robot
        # backs off to hunt for the next leaf instead of idling on this one.
        max_sweeps=1,
        preview=args.preview,
        save=args.save,
        stop_on_arrival=args.stop_on_arrival,
    )

    with ExitStack() as stack:
        driver = L298N(stack) if args.drive else None
        if driver is None:
            print("dry run: motors are not energised (pass --drive to move)")
        return drive_loop(camera, detector, params, options, driver=driver)


if __name__ == "__main__":
    raise SystemExit(main())