"""Drive the robot from a browser: WASD, arrow keys, or an on-screen joystick.

Open http://<raspberry-pi-ip>:8001/ and steer. The server applies a deadman
timeout, so if the browser closes, the tab loses focus, or the network drops,
the motors stop within `--deadman` seconds instead of running away.

Motors stay unpowered unless --drive is passed, matching capture.py/pickup.py.
"""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import ExitStack

from capture import Camera
from drive import L298N
from webvideo import CameraPublisher, MjpegServer, page

# Second L298N board: one brush motor on channel A. Channel B is unused.
BRUSH_ENA = 12
BRUSH_IN1 = 16
BRUSH_IN2 = 20


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class Teleop:
    """Turns joystick vectors into wheel speeds, with a deadman timeout."""

    def __init__(
        self,
        robot: L298N | None,
        brush: L298N | None,
        speed: float = 0.6,
        brush_speed: float = 1.0,
        deadman: float = 0.5,
        invert_left: bool = False,
        invert_right: bool = False,
    ) -> None:
        self.robot = robot
        self.brush = brush
        self.speed = speed
        self.brush_speed = brush_speed
        self.deadman = deadman
        self.invert_left = -1.0 if invert_left else 1.0
        self.invert_right = -1.0 if invert_right else 1.0

        self._lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._stamp = 0.0
        self._brush_on = False
        self._stop = threading.Event()

        self._left = 0.0
        self._right = 0.0
        self._brush_applied: bool | None = None
        self._commands = 0

    # -- inputs -------------------------------------------------------------
    def set_command(self, x: float, y: float) -> None:
        with self._lock:
            self._x = _clamp(x)
            self._y = _clamp(y)
            self._stamp = time.time()
            self._commands += 1

    def set_brush(self, on: bool) -> None:
        with self._lock:
            self._brush_on = bool(on)

    def release(self) -> None:
        """Drop the current command as if the stick were centred."""
        with self._lock:
            self._x = 0.0
            self._y = 0.0
            self._stamp = 0.0

    # -- control loop -------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                x, y, stamp = self._x, self._y, self._stamp
                brush_on = self._brush_on

            # A stale command means the operator stopped sending; halt.
            fresh = stamp > 0 and (time.time() - stamp) <= self.deadman
            if not fresh:
                x = y = 0.0

            left = _clamp((y + x) * self.speed) * self.invert_left
            right = _clamp((y - x) * self.speed) * self.invert_right

            if self.robot is not None:
                self.robot.drive(left, right)

            if self.brush is not None and brush_on != self._brush_applied:
                if brush_on:
                    self.brush.drive(self.brush_speed)
                else:
                    self.brush.stop()
                self._brush_applied = brush_on

            with self._lock:
                self._left, self._right = left, right

            self._stop.wait(0.05)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.robot is not None:
            self.robot.stop()
        if self.brush is not None:
            self.brush.stop()

    def stats(self) -> dict:
        with self._lock:
            return {
                "left": round(self._left, 2),
                "right": round(self._right, 2),
                "speed": round(self.speed, 2),
                "brush": self._brush_on,
                "commands": self._commands,
                "armed": self.robot is not None,
            }


def build_page() -> str:
    body = """
<div class="layout">
  <div class="controls">
    <div class="col">
      <div class="joy">
        <div id="stick" class="stick"><div id="knob" class="knob"></div></div>
      </div>
      <div class="stats">
        <span class="stat">L <b id="left">0.00</b></span>
        <span class="stat">R <b id="right">0.00</b></span>
        <span class="stat">speed <b id="spd">60</b>%</span>
      </div>
      <div class="stats">
        <span class="stat" id="mode"></span>
        <span class="stat" id="link">connecting</span>
      </div>
    </div>
    
    <div class="feed"><img src="/stream" alt="camera feed"></div>
  

    <div class="col right">
      <div class="dpad">
        <button class="pad up" data-dir="y+">&#9650;</button>
        <button class="pad left" data-dir="x-">&#9664;</button>
        <button class="pad right" data-dir="x+">&#9654;</button>
        <button class="pad down" data-dir="y-">&#9660;</button>
      </div>
      <button class="stop" id="stop">STOP</button>
      <button class="brush" id="brush">brushes off</button>
      <label class="slider-label">speed <input id="speed" type="range" min="10" max="100" value="60"></label>
    </div>
  </div>
</div>

<style>
  .layout { display: flex; gap: 16px; align-items: flex-start; justify-content: center;
            flex-wrap: wrap; width: 100%; max-width: 1080px; }
  .feed { flex: 1 1 460px; min-width: 300px; }
  .feed img { max-height: 56vh; width: 100%; object-fit: contain; }
  .controls { display: flex; gap: 16px; flex: 1 1 480px; min-width: 360px;
              align-items: flex-start; }
  .col { display: flex; flex-direction: column; gap: 12px; align-items: center; }
  .col.right { align-items: center; }

  .stick { position: relative; width: 160px; height: 160px; border-radius: 50%;
           background: radial-gradient(circle at 50% 50%, #1b232c 0%, #141a21 70%);
           border: 1px solid #2a3947; touch-action: none; user-select: none; }
  .knob { position: absolute; width: 56px; height: 56px; border-radius: 50%;
          background: #2d6cdf; left: 52px; top: 52px; pointer-events: none;
          box-shadow: 0 0 18px rgba(45,108,223,.55); }

  .stats { display: flex; flex-direction: column; align-items: center; gap: 2px; }
  .stat { color: #8a94a0; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
  .stat b { color: #e6e9ee; font-size: 18px; font-weight: 600; }

  .dpad { display: grid; grid-template-columns: 60px 60px; grid-template-rows: 60px 60px;
          gap: 6px; }
  .pad { width: 60px; height: 60px; font-size: 18px; border-radius: 12px;
         border: 1px solid #2a3947; background: #1b232c; color: #e6e9ee;
         cursor: pointer; touch-action: none; user-select: none; }
  .pad:active, .pad.on { background: #2d6cdf; border-color: #2d6cdf; }
  .stop { width: 110px; height: 42px; background: #4a1f24; border-color: #7a3038;
          font-weight: 700; letter-spacing: .04em; }
  .brush { width: 110px; height: 42px; background: #241f4a; border-color: #38307a;
            font-weight: 700; letter-spacing: .04em; }
  .slider-label { display: flex; align-items: center; gap: 8px;
                  color: #8a94a0; font-size: 12px; text-transform: uppercase;
                  letter-spacing: .06em; }
  .slider-label input { width: 80px; }
</style>

<script>
const el = id => document.getElementById(id);
const keys = new Set();
let padVec = null;          // joystick vector while touched, null otherwise
let brushOn = false;

const KEYMAP = {
  w: [0, 1], arrowup: [0, 1],
  s: [0, -1], arrowdown: [0, -1],
  a: [-1, 0], arrowleft: [-1, 0],
  d: [1, 0], arrowright: [1, 0],
};

function keyVector() {
  let x = 0, y = 0;
  for (const k of keys) { const v = KEYMAP[k]; if (v) { x += v[0]; y += v[1]; } }
  return { x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)) };
}

function current() { return padVec || keyVector(); }

addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (KEYMAP[k] || k === ' ') { e.preventDefault(); }
  if (k === ' ') { stopAll(); return; }
  if (KEYMAP[k]) {
    keys.add(k);
    send(current());   // react at once, and stay alive on key auto-repeat
  }
});
addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
addEventListener('blur', stopAll);

// Browsers throttle timers in background tabs to roughly 1 Hz, so the 20 Hz
// command loop cannot be trusted to keep the robot moving - or to stop it.
// Send an explicit stop the moment the page is hidden or unloaded.
document.addEventListener('visibilitychange', () => { if (document.hidden) stopAll(); });

// On-screen arrows
const held = new Set();
document.querySelectorAll('[data-dir]').forEach(btn => {
  const dir = btn.dataset.dir;
  const on = e => { e.preventDefault(); held.add(dir); btn.classList.add('on'); applyPad(); };
  const off = e => { e.preventDefault(); held.delete(dir); btn.classList.remove('on'); applyPad(); };
  btn.addEventListener('pointerdown', on);
  btn.addEventListener('pointerup', off);
  btn.addEventListener('pointerleave', off);
  btn.addEventListener('pointercancel', off);
});

function applyPad() {
  if (!held.size) { if (!stickActive) padVec = null; return; }
  let x = 0, y = 0;
  for (const d of held) { if (d === 'x+') x += 1; if (d === 'x-') x -= 1;
                          if (d === 'y+') y += 1; if (d === 'y-') y -= 1; }
  padVec = { x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)) };
}

// Joystick
const stick = el('stick'), knob = el('knob');
let stickActive = false;
const R = 63;
function centreKnob() { knob.style.left = '63px'; knob.style.top = '63px'; }

function moveStick(e) {
  const r = stick.getBoundingClientRect();
  let dx = e.clientX - (r.left + r.width / 2);
  let dy = e.clientY - (r.top + r.height / 2);
  const len = Math.hypot(dx, dy) || 1;
  const scaled = Math.min(len, R);
  dx = dx / len * scaled; dy = dy / len * scaled;
  knob.style.left = (63 + dx) + 'px';
  knob.style.top = (63 + dy) + 'px';
  const nx = dx / R, ny = -dy / R;               // screen y is inverted
  const mag = Math.hypot(nx, ny);
  padVec = mag > 1 ? { x: nx / mag, y: ny / mag } : { x: nx, y: ny };
}

stick.addEventListener('pointerdown', e => {
  stickActive = true; stick.setPointerCapture(e.pointerId); moveStick(e);
});
stick.addEventListener('pointermove', e => { if (stickActive) moveStick(e); });
const endStick = () => { stickActive = false; padVec = null; centreKnob(); };
stick.addEventListener('pointerup', endStick);
stick.addEventListener('pointercancel', endStick);

// Stop button
function stopAll() { keys.clear(); held.clear(); padVec = null; centreKnob();
  document.querySelectorAll('.pad.on').forEach(b => b.classList.remove('on'));
  send({ x: 0, y: 0 }); }
el('stop').addEventListener('click', stopAll);

// Speed slider
el('speed').addEventListener('input', e => {
  el('spd').textContent = e.target.value;
  fetch('/speed', { method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ percent: Number(e.target.value) }) });
});

// Brush toggle
el('brush').addEventListener('click', () => {
  brushOn = !brushOn;
  el('brush').textContent = brushOn ? 'brushes on' : 'brushes off';
  el('brush').classList.toggle('on', brushOn);
  fetch('/brush', { method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ on: brushOn }) });
});

function send(v) {
  fetch('/cmd', { method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify(v), keepalive: true }).catch(() => {});
}

// Command loop, 20 Hz
setInterval(() => { const v = current(); send(v); }, 50);
addEventListener('pagehide', () => { send({ x: 0, y: 0 }); });

// Stats
setInterval(async () => {
  try {
    const s = await (await fetch('/stats', {cache: 'no-store'})).json();
    el('left').textContent = s.left.toFixed(2);
    el('right').textContent = s.right.toFixed(2);
    el('spd').textContent = Math.round(s.speed * 100);
    el('mode').textContent = s.armed ? '' : 'DRY RUN - motors off';
    el('link').textContent = 'linked';
  } catch (e) { el('link').textContent = 'link lost'; }
}, 500);
</script>
"""
    return page("Leaf robot - teleop", body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the robot from a web browser.")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--speed", type=int, default=60, help="drive speed percent (10-100)")
    parser.add_argument("--brush-speed", type=int, default=100, help="brush speed percent")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality (default: 75)")
    parser.add_argument(
        "--fps", type=float, default=15.0,
        help="maximum streamed frames per second (default: 15)",
    )
    parser.add_argument(
        "--deadman", type=float, default=0.5,
        help="seconds without a command before the motors stop (default: 0.5)",
    )
    parser.add_argument("--invert-left", action="store_true", help="reverse the left motor")
    parser.add_argument("--invert-right", action="store_true", help="reverse the right motor")
    parser.add_argument(
        "--drive", action="store_true", help="energise the motors; without it nothing moves"
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
    if not 1 <= args.speed <= 100:
        parser.error("--speed must be between 1 and 100")
    if not 1 <= args.brush_speed <= 100:
        parser.error("--brush-speed must be between 1 and 100")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if not 1.0 <= args.fps <= 60.0:
        parser.error("--fps must be between 1 and 60")
    if args.deadman <= 0:
        parser.error("--deadman must be greater than 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    camera = Camera(args.width, args.height, args.flip)
    publisher = CameraPublisher(camera, fps=args.fps, quality=args.quality)
    publisher.start()

    with ExitStack() as stack:
        robot = L298N(stack) if args.drive else None
        brush = (
            L298N(
                stack,
                ena=BRUSH_ENA,
                in1=BRUSH_IN1,
                in2=BRUSH_IN2,
                enb=None,
                in3=None,
                in4=None,
            )
            if args.drive
            else None
        )

        teleop = Teleop(
            robot,
            brush,
            speed=args.speed / 100.0,
            brush_speed=args.brush_speed / 100.0,
            deadman=args.deadman,
            invert_left=args.invert_left,
            invert_right=args.invert_right,
        )
        teleop.start()

        def on_command(handler) -> None:
            body = handler.read_json()
            try:
                x = float(body.get("x", 0.0))
                y = float(body.get("y", 0.0))
            except (TypeError, ValueError):
                handler.send_error(400, "x and y must be numbers")
                return
            teleop.set_command(x, y)
            handler.send_json({"ok": True})

        def on_brush(handler) -> None:
            teleop.set_brush(bool(handler.read_json().get("on", False)))
            handler.send_json({"ok": True})

        def on_speed(handler) -> None:
            try:
                percent = float(handler.read_json().get("percent", 60))
            except (TypeError, ValueError):
                handler.send_error(400, "percent must be a number")
                return
            teleop.speed = max(0.1, min(1.0, percent / 100.0))
            handler.send_json({"ok": True, "speed": teleop.speed})

        server = MjpegServer(
            publisher.source,
            get_routes={
                "/": lambda handler: handler.send_html(build_page()),
                "/stream": lambda handler: handler.stream_mjpeg(),
                "/stats": lambda handler: handler.send_json(teleop.stats()),
            },
            post_routes={"/cmd": on_command, "/brush": on_brush, "/speed": on_speed},
            host=args.host,
            port=args.port,
        )

        print(f"camera backend: {camera.backend}")
        if args.drive:
            print("motors ARMED")
        else:
            print("dry run: motors are not energised (pass --drive to move)")
        print(f"teleop on http://<raspberry-pi-ip>:{args.port}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            publisher.stop()
            teleop.stop()
            server.server_close()
            camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())