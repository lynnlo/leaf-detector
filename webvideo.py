"""Reusable threaded MJPEG-over-HTTP server.

Several browser tabs can watch the same camera at once: the producer publishes
JPEG frames into a FrameSource and every connected client is served the newest
one. Nothing here knows about cameras, motors, or leaf detection.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

BOUNDARY = "frame"


class FrameSource:
    """Holds the newest encoded JPEG and lets clients wait for the next one."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0

    def publish(self, jpeg: bytes) -> None:
        """Record a new frame and wake every waiting client."""
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def latest(self) -> tuple[int, bytes | None]:
        """The current sequence number and frame, without blocking."""
        with self._condition:
            return self._sequence, self._jpeg

    def wait_next(self, after: int, timeout: float = 5.0) -> tuple[int, bytes] | None:
        """Block until a frame newer than `after` exists, or the timeout passes."""
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > after, timeout=timeout)
            if self._jpeg is None or self._sequence <= after:
                return None
            return self._sequence, self._jpeg


class _Handler(BaseHTTPRequestHandler):
    server_version = "leafdec"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence per-request logging; a streaming client would flood the console."""
        pass

    def do_GET(self) -> None:
        route = self.server.get_routes.get(self.path.split("?")[0])
        if route is None:
            self.send_error(404)
            return
        route(self)

    def do_POST(self) -> None:
        route = self.server.post_routes.get(self.path.split("?")[0])
        if route is None:
            self.send_error(404)
            return
        route(self)

    def read_json(self) -> dict:
        """Parse the request body as JSON, tolerating an empty or malformed body."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        raw = self.rfile.read(min(length, 64 * 1024))
        try:
            parsed = json.loads(raw.decode("utf-8", "replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def send_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream_mjpeg(self) -> None:
        """Serve multipart/x-mixed-replace until the client disconnects."""
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True

        source: FrameSource = self.server.frame_source
        sequence, jpeg = source.latest()
        try:
            while True:
                result = source.wait_next(sequence, timeout=5.0)
                if result is None:
                    continue  # Quiet camera: hold the connection open.
                sequence, jpeg = result
                self.wfile.write(
                    f"--{BOUNDARY}\r\n".encode("ascii")
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.flush()
        except OSError:
            # The client went away mid-stream: a closed tab, a curl timeout, or
            # a dropped WiFi link. BrokenPipe, ConnectionReset and Windows'
            # ConnectionAborted all fall under OSError, and all just end here.
            return


class MjpegServer(ThreadingHTTPServer):
    """HTTP server exposing a FrameSource plus caller-supplied routes."""

    daemon_threads = True

    def __init__(
        self,
        frame_source: FrameSource,
        get_routes: dict[str, Callable[[_Handler], None]],
        post_routes: dict[str, Callable[[_Handler], None]],
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self.frame_source = frame_source
        self.get_routes = get_routes
        self.post_routes = post_routes
        super().__init__((host, port), _Handler)

    def serve_background(self) -> threading.Thread:
        """Run the server on its own daemon thread so a caller can also drive hardware."""
        thread = threading.Thread(target=self.serve_forever, name="mjpeg-server", daemon=True)
        thread.start()
        return thread

    def handle_error(self, request, client_address) -> None:
        """Ignore clients that vanish mid-request; report everything else.

        Closing a browser tab drops keep-alive connections, which Python reports
        as a connection reset. That is routine, not a fault, so it is silenced.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def page(title: str, body: str) -> str:
    """Wrap shared styling and layout around an app's own HTML."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px; min-height: 100vh;
    background: #101418; color: #e6e9ee;
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; align-items: center; gap: 14px;
  }}
  h1 {{ margin: 0; font-size: 15px; font-weight: 600; letter-spacing: .02em;
        color: #9fb0c4; text-transform: uppercase; }}
  .feed {{ width: 100%; max-width: 860px; border-radius: 10px; overflow: hidden;
           background: #000; border: 1px solid #23303c; line-height: 0; }}
  .feed img {{ width: 100%; height: auto; display: block; }}
  .bar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
          justify-content: center; max-width: 860px; }}
  .stat {{ font-variant-numeric: tabular-nums; color: #7f90a4; }}
  .stat b {{ color: #e6e9ee; font-weight: 600; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""