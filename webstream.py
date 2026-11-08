"""Serve the leaf-detection camera feed, with boxes drawn, over HTTP.

Open http://<raspberry-pi-ip>:8000/ in any browser on the same network to watch
the annotated feed. Detection runs on its own thread so the video stays smooth
even though a workflow call takes about half a second.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from capture import Camera, DriveParams, annotate, choose_target, compute_command
from leaf_detector import (
    DEFAULT_CLASSES,
    DEFAULT_WORKFLOW_ID,
    DEFAULT_WORKSPACE,
    RoboflowError,
    build_detector,
)
from webvideo import CameraPublisher, MjpegServer, page

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "leaf.onnx"


class AnnotatedFeed:
    """Camera -> annotate -> JPEG, with detection on a separate thread."""

    def __init__(
        self,
        camera: Camera,
        detector,
        detect_every: float = 0.5,
        jpeg_quality: int = 80,
        params: DriveParams | None = None,
        fps: float = 15.0,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.detect_every = max(0.05, detect_every)
        self.params = params or DriveParams()

        self._lock = threading.Lock()
        self._detections: list = []
        self._target = None
        self._note = "starting up"
        self._detect_ms = 0.0
        self._error = ""
        self._stop = threading.Event()

        # The publisher owns the camera read and the JPEG encode; this class only
        # decides what to draw on each frame and when to run detection.
        self.publisher = CameraPublisher(
            camera, fps=fps, quality=jpeg_quality, render=self._render
        )
        self.source = self.publisher.source

    # -- background workers -------------------------------------------------
    def _render(self, frame):
        """Draw the newest detection result onto a frame before it is encoded."""
        with self._lock:
            detections, target, note = self._detections, self._target, self._note
        return annotate(frame, detections, target, note)

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.publisher.latest_raw()

            if frame is None:
                with self._lock:
                    self._note = "waiting for the camera"
                self._stop.wait(self.detect_every)
                continue

            started = time.time()
            try:
                detections = self.detector.detect(frame)
                target = choose_target(detections)
                _, _, note = compute_command(
                    target, frame.shape[1], frame.shape[0], self.params
                )
                error = ""
            except RoboflowError as exc:
                # A network hiccup should dim the labels, not kill the video.
                detections, target, note = [], None, "detector unavailable"
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 - keep the feed alive
                detections, target, note = [], None, "detector error"
                error = f"{type(exc).__name__}: {exc}"

            with self._lock:
                self._detections = detections
                self._target = target
                self._note = note
                self._error = error
                self._detect_ms = (time.time() - started) * 1000.0

            self._stop.wait(self.detect_every)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self.publisher.start()
        self._thread = threading.Thread(
            target=self._detect_loop, name="detect", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.publisher.stop()

    def stats(self) -> dict:
        with self._lock:
            leaves = len(self._detections)
            note = self._note
            detect_ms = self._detect_ms
            error = self._error
        return {
            "leaves": leaves,
            "note": note,
            "fps": round(self.publisher.measured_fps, 1),
            "detect_ms": round(detect_ms),
            "camera": self.camera.backend,
            "error": error or self.publisher.error,
        }


def build_page() -> str:
    body = """
<div class="feed"><img src="/stream" alt="annotated leaf feed"></div>
<div class="bar">
  <span class="stat">leaves <b id="leaves">-</b></span>
  <span class="stat">video <b id="fps">-</b> fps</span>
  <span class="stat">detect <b id="ms">-</b> ms</span>
  <span class="stat">camera <b id="cam">-</b></span>
  <span class="stat" id="note" style="color:#8fb6ff"></span>
</div>
<div class="bar"><span class="stat" id="err" style="color:#ff8a80"></span></div>
<script>
const el = id => document.getElementById(id);
async function poll() {
  try {
    const s = await (await fetch('/stats', {cache: 'no-store'})).json();
    el('leaves').textContent = s.leaves;
    el('fps').textContent = s.fps;
    el('ms').textContent = s.detect_ms;
    el('cam').textContent = s.camera;
    el('note').textContent = s.note;
    el('err').textContent = s.error || '';
  } catch (e) { el('err').textContent = 'server unreachable'; }
}
setInterval(poll, 1000); poll();
</script>
"""
    return page("Leaf detection - live feed", body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the annotated leaf-detection feed over HTTP."
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument(
        "--backend", choices=["roboflow", "onnx"], default="roboflow",
        help="roboflow calls the hosted workflow; onnx runs a local export",
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
    parser.add_argument(
        "--detect-every", type=float, default=0.5,
        help="seconds between detection passes (default: 0.5)",
    )
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (default: 80)")
    parser.add_argument(
        "--fps", type=float, default=15.0,
        help="maximum streamed frames per second (default: 15)",
    )
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="vertical",
        help="correct for the camera's mounting: vertical swaps top/bottom "
        "(upside-down mount), horizontal mirrors left/right, both rotates 180 "
        "degrees, none leaves the frame as captured (default: vertical)",
    )
    args = parser.parse_args(argv)

    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if not 1.0 <= args.fps <= 60.0:
        parser.error("--fps must be between 1 and 60")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.backend == "onnx":
        if not args.model.is_file():
            print(f"ONNX model not found: {args.model}", file=sys.stderr)
            return 1
    elif not (args.api_key or os.environ.get("ROBOFLOW_API_KEY")):
        print("no Roboflow API key: set ROBOFLOW_API_KEY or pass --api-key", file=sys.stderr)
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

    camera = Camera(args.width, args.height, args.flip)
    feed = AnnotatedFeed(
        camera,
        detector,
        detect_every=args.detect_every,
        jpeg_quality=args.quality,
        fps=args.fps,
    )
    feed.start()

    server = MjpegServer(
        feed.source,
        get_routes={
            "/": lambda handler: handler.send_html(build_page()),
            "/stream": lambda handler: handler.stream_mjpeg(),
            "/stats": lambda handler: handler.send_json(feed.stats()),
        },
        post_routes={},
        host=args.host,
        port=args.port,
    )

    print(f"camera backend: {camera.backend}   detector: {args.backend}")
    print(f"annotated feed on http://<raspberry-pi-ip>:{args.port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        feed.stop()
        server.server_close()
        camera.close()
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())