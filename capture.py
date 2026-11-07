import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import subprocess

# Capture single frame via raspivid/cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

_, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
b = buf.tobytes()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)

HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()