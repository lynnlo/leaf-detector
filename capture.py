import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

def capture_frame() -> bytes:
    result = subprocess.run(
        [
            "rpicam-still",
            "--nopreview",
            "--timeout",
            "1000",
            "--output",
            "-",
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rpicam-still failed: {error or result.returncode}")
    if not result.stdout:
        raise RuntimeError("rpicam-still returned an empty image")
    return result.stdout


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, image: bytes, **kwargs):
        self.image = image
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(self.image)))
        self.end_headers()
        self.wfile.write(self.image)


def main() -> None:
    image = capture_frame()
    server = HTTPServer(
        ("0.0.0.0", 8000),
        lambda *args, **kwargs: Handler(*args, image=image, **kwargs),
    )
    print("Serving captured image on port 8000")
    with server:
        server.serve_forever()


if __name__ == "__main__":
    main()
