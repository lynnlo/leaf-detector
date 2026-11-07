import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import BinaryIO


def read_jpeg_frame(stream: BinaryIO, buffer: bytearray) -> bytes | None:
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"

    while True:
        start = buffer.find(start_marker)
        if start == -1:
            chunk = stream.read(4096)
            if not chunk:
                return None
            buffer.extend(chunk)
            start = buffer.find(start_marker)
            if start == -1:
                # Keep a possible first byte of a JPEG marker split across reads.
                keep = 1 if buffer[-1:] == b"\xff" else 0
                if keep:
                    buffer[:] = buffer[-1:]
                else:
                    buffer.clear()
                continue

        if start:
            del buffer[:start]

        end = buffer.find(end_marker, len(start_marker))
        if end != -1:
            frame_end = end + len(end_marker)
            frame = bytes(buffer[:frame_end])
            del buffer[:frame_end]
            return frame

        chunk = stream.read(4096)
        if not chunk:
            return None
        buffer.extend(chunk)


class VideoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        camera = self.server.camera
        buffer = self.server.jpeg_buffer
        while True:
            frame = read_jpeg_frame(camera.stdout, buffer)
            if frame is None:
                return
            try:
                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return


class VideoHTTPServer(HTTPServer):
    def __init__(self, camera: subprocess.Popen[bytes]) -> None:
        self.camera = camera
        self.jpeg_buffer = bytearray()
        super().__init__(("0.0.0.0", 8000), VideoHandler)


def main() -> None:
    camera = subprocess.Popen(
        [
            "rpicam-vid",
            "--nopreview",
            "--timeout",
            "0",
            "--codec",
            "mjpeg",
            "--width",
            "640",
            "--height",
            "480",
            "--framerate",
            "20",
            "--output",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    if camera.stdout is None:
        camera.terminate()
        raise RuntimeError("Could not open rpicam-vid output")

    try:
        server = VideoHTTPServer(camera)
        print("Streaming camera video at http://<raspberry-pi-ip>:8000/")
        with server:
            server.serve_forever()
    finally:
        if camera.poll() is None:
            camera.terminate()
            try:
                camera.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera.kill()
                camera.wait()


if __name__ == "__main__":
    main()
