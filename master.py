from __future__ import annotations

import asyncio
import base64
import ctypes
import json
import logging
import math
import os
import platform
import queue
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Optional, Set

# Suppress low-level ALSA / C-library error spew on Linux systems
if platform.system() == "Linux":
    try:
        asound = ctypes.cdll.LoadLibrary("libasound.so.2")
        asound.snd_lib_error_set_handler(None)
    except Exception:
        pass

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import sounddevice as sd  # type: ignore
except Exception:
    sd = None

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, StreamingResponse
except Exception as exc:
    print("FastAPI is required. Install with: pip install fastapi uvicorn", file=sys.stderr)
    raise


HOST = os.environ.get("RC_HOST", "0.0.0.0")
PORT = int(os.environ.get("RC_PORT", "8000"))

# Supports both integer indices (1) and explicit V4L2 device paths ("/dev/video1")
_raw_dev = os.environ.get("RC_VIDEO_DEVICE", "1")
VIDEO_DEVICE: Any = int(_raw_dev) if _raw_dev.isdigit() else _raw_dev

VIDEO_WIDTH = int(os.environ.get("RC_VIDEO_WIDTH", "480"))
VIDEO_HEIGHT = int(os.environ.get("RC_VIDEO_HEIGHT", "360"))
VIDEO_FPS = float(os.environ.get("RC_VIDEO_FPS", "30"))
JPEG_QUALITY = int(os.environ.get("RC_JPEG_QUALITY", "50"))
AUDIO_RATE = int(os.environ.get("RC_AUDIO_RATE", "48000"))

# Raised default audio block size to prevent ALSA buffer underruns
AUDIO_BLOCK = int(os.environ.get("RC_AUDIO_BLOCK", "1024"))
VERBOSE = os.environ.get("RC_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
VOICE_RMS_THRESHOLD = float(os.environ.get("RC_VOICE_RMS", "0.018"))
VOICE_HOLD_SECONDS = float(os.environ.get("RC_VOICE_HOLD", "0.38"))
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "AIzaSyBX2ybnshzC8rdm2M7RIrU77jva1KqfR3Y")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "levislawns")
_AUTH_CACHE: Dict[str, tuple[float, str]] = {}


class CancelledErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[0] is asyncio.CancelledError:
            return False
        message = record.getMessage()
        return "asyncio.exceptions.CancelledError" not in message


def quiet_stream_disconnect_logs() -> None:
    cancelled_filter = CancelledErrorFilter()
    for logger_name in ("uvicorn.error", "starlette.error", "fastapi"):
        logging.getLogger(logger_name).addFilter(cancelled_filter)


TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/"
    "9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Ap//xAAUEAEAAAAAAAAA"
    "AAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/"
    "2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def now_ms() -> int:
    return int(time.time() * 1000)


class ThreadedCamera:
    def __init__(self, device: Any, width: int, height: int, fps: float, quality: int) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self._cap: Any = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_jpeg = TINY_JPEG
        self._camera_ok = False
        self._open_attempted = False
        self._frame_no = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _open(self) -> None:
        self._open_attempted = True
        if cv2 is None:
            print("cv2 is not installed; using static fallback video frame.")
            return

        # Prefer V4L2 API explicitly on Linux to avoid unnecessary driver probes
        if platform.system() == "Linux" and hasattr(cv2, "CAP_V4L2"):
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(self.device)

        if not self._cap or not self._cap.isOpened():
            self._camera_ok = False
            if self._cap:
                with suppress(Exception):
                    self._cap.release()
                self._cap = None
            print(f"No camera detected at index/device {self.device}; using generated dummy video.")
            return
        self._camera_ok = True
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        with suppress(Exception):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Camera opened at device {self.device}.")

    def _dummy_frame(self) -> Optional[bytes]:
        if cv2 is None or np is None:
            return None
        h, w = self.height, self.width
        self._frame_no += 1
        t = self._frame_no / max(self.fps, 1)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x_grad = np.linspace(20, 130, w, dtype=np.uint8)
        y_grad = np.linspace(15, 90, h, dtype=np.uint8)
        frame[:, :, 0] = x_grad[None, :]
        frame[:, :, 1] = y_grad[:, None]
        frame[:, :, 2] = 35
        cx = int((math.sin(t * 1.7) * 0.35 + 0.5) * w)
        cy = int((math.cos(t * 1.2) * 0.28 + 0.5) * h)
        cv2.circle(frame, (cx, cy), max(18, min(w, h) // 12), (45, 210, 255), -1)
        cv2.putText(frame, "NO CAMERA", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (24, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return encoded.tobytes() if ok else None

    def _run(self) -> None:
        if self._cap is None and cv2 is not None and not self._open_attempted:
            self._open()
        interval = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            start = time.monotonic()
            jpeg: Optional[bytes] = None
            if self._camera_ok and self._cap:
                for _ in range(2):
                    self._cap.grab()
                ok, frame = self._cap.read()
                if ok and frame is not None and cv2 is not None:
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                    if ok:
                        jpeg = encoded.tobytes()
                else:
                    self._camera_ok = False
                    print("Camera read failed; switching to dummy video.")
            if jpeg is None:
                jpeg = self._dummy_frame() or TINY_JPEG
            with self._lock:
                self._last_jpeg = jpeg
            elapsed = time.monotonic() - start
            self._stop.wait(max(0.001, interval - elapsed))

    def frame(self) -> bytes:
        with self._lock:
            return self._last_jpeg

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._cap:
            with suppress(Exception):
                self._cap.release()


class MotorDriver:
    def __init__(self) -> None:
        self.platform = platform.system()
        self.mode = "simulator"
        self.left = 0.0
        self.right = 0.0
        self._gpio: Any = None
        self._drive_motor: Any = None
        self._steer_motor: Any = None
        self._init_gpio()

    def _init_gpio(self) -> None:
        is_linux = self.platform == "Linux"
        machine = platform.machine().lower()
        looks_pi = is_linux and ("arm" in machine or "aarch64" in machine)
        if not looks_pi:
            print(f"Motor driver in simulator mode on {self.platform}/{platform.machine()}.")
            return
        try:
            from gpiozero import Motor  # type: ignore

            drive_pins = (int(os.environ.get("RC_DRIVE_FWD", "17")), int(os.environ.get("RC_DRIVE_REV", "18")))
            steer_pins = (int(os.environ.get("RC_STEER_FWD", "22")), int(os.environ.get("RC_STEER_REV", "23")))
            self._drive_motor = Motor(forward=drive_pins[0], backward=drive_pins[1], pwm=True)
            self._steer_motor = Motor(forward=steer_pins[0], backward=steer_pins[1], pwm=True)
            self.mode = "gpiozero"
            print(f"Motor driver using gpiozero. Drive={drive_pins}, Steer={steer_pins}")
        except Exception as exc:
            print(f"GPIO unavailable ({exc}); motor driver using simulator mode.")

    def set_tank(self, throttle: float, turn: float) -> Dict[str, Any]:
        self.left = clamp(throttle)
        self.right = clamp(turn)
        if self.mode == "gpiozero":
            self._apply_gpio(self._drive_motor, self.left)
            self._apply_gpio(self._steer_motor, self.right)
        elif VERBOSE:
            print(f"[motor sim] throttle={self.left:+.2f} turn={self.right:+.2f}")
        return self.telemetry()

    def drive(self, throttle: float, turn: float) -> Dict[str, Any]:
        return self.set_tank(throttle, turn)

    def stop(self) -> None:
        self.set_tank(0.0, 0.0)
        for motor in (self._drive_motor, self._steer_motor):
            if motor is not None:
                with suppress(Exception):
                    motor.stop()

    def close(self) -> None:
        self.stop()
        for motor in (self._drive_motor, self._steer_motor):
            if motor is not None:
                with suppress(Exception):
                    motor.close()

    def telemetry(self) -> Dict[str, Any]:
        return {
            "ts": now_ms(),
            "platform": self.platform,
            "mode": self.mode,
            "left": round(self.left, 3),
            "right": round(self.right, 3),
        }

    @staticmethod
    def _apply_gpio(motor: Any, value: float) -> None:
        if motor is None:
            return
        speed = abs(clamp(value))
        if speed < 0.02:
            motor.stop()
        elif value > 0:
            motor.forward(speed)
        else:
            motor.backward(speed)


class AudioManager:
    def __init__(self, sample_rate: int, blocksize: int) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.available = sd is not None
        self.output_ok = False
        self.input_ok = False
        self._output_stream: Any = None
        self._input_stream: Any = None
        self._speaker_q: queue.Queue[bytes] = queue.Queue(maxsize=16)
        self._speaker_buffer = bytearray()
        self._client_speaking_until = 0.0
        self._host_speaking_until = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._host_listeners: Set[asyncio.Queue[bytes]] = set()
        self._lock = threading.Lock()
        self._last_input_rms = 0.0
        self._last_output_rms = 0.0
        self._device_summary = "sounddevice unavailable"
        if self.available:
            self._probe_devices()

    def _probe_devices(self) -> None:
        try:
            devices = sd.query_devices()
            default_in, default_out = sd.default.device
            self._device_summary = f"default input={default_in}, output={default_out}, devices={len(devices)}"
            print(f"Audio devices: {self._device_summary}")
        except Exception as exc:
            self.available = False
            self._device_summary = f"audio probe failed: {exc}"
            print(self._device_summary)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if not self.available or sd is None:
            print("Audio disabled: sounddevice is not available.")
            return
        self._start_output()
        self._start_input()

    def _start_output(self) -> None:
        if sd is None:
            return

        def callback(outdata: Any, frames: int, _time_info: Any, status: Any) -> None:
            needed = frames * 2
            while len(self._speaker_buffer) < needed:
                try:
                    self._speaker_buffer.extend(self._speaker_q.get_nowait())
                except queue.Empty:
                    break
            if len(self._speaker_buffer) < needed:
                outdata[:] = b"\x00" * needed
                self._last_output_rms = 0.0
                return
            data = bytes(self._speaker_buffer[:needed])
            del self._speaker_buffer[:needed]
            outdata[:] = data
            self._last_output_rms = pcm16_rms(data)

        try:
            self._output_stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.blocksize,
                latency="high" if platform.system() == "Linux" else "low",
                callback=callback,
            )
            self._output_stream.start()
            self.output_ok = True
            print("Host speaker output enabled.")
        except Exception as exc:
            self.output_ok = False
            print(f"Host speaker output disabled: {exc}")

    def _start_input(self) -> None:
        if sd is None:
            return

        def callback(indata: Any, frames: int, _time_info: Any, status: Any) -> None:
            data = bytes(indata)
            rms = pcm16_rms(data)
            now = time.monotonic()
            self._last_input_rms = rms
            if now < self._client_speaking_until:
                return
            if rms > VOICE_RMS_THRESHOLD:
                self._host_speaking_until = now + VOICE_HOLD_SECONDS
            with self._lock:
                listeners = list(self._host_listeners)
            if self._loop is None:
                return
            for listener in listeners:
                self._loop.call_soon_threadsafe(drop_put_nowait, listener, data)

        try:
            self._input_stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.blocksize,
                latency="high" if platform.system() == "Linux" else "low",
                callback=callback,
            )
            self._input_stream.start()
            self.input_ok = True
            print("Host microphone input enabled.")
        except Exception as exc:
            self.input_ok = False
            print(f"Host microphone input disabled: {exc}")

    def push_speaker_pcm(self, data: bytes) -> None:
        if not data or not self.output_ok:
            return
        rms = pcm16_rms(data)
        now = time.monotonic()
        if rms > VOICE_RMS_THRESHOLD:
            if now < self._host_speaking_until:
                return
            self._client_speaking_until = now + VOICE_HOLD_SECONDS
        
        while self._speaker_q.qsize() > 4:
            with suppress(queue.Empty):
                self._speaker_q.get_nowait()
        with suppress(queue.Full):
            self._speaker_q.put_nowait(data)

    def add_host_listener(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
        with self._lock:
            self._host_listeners.add(q)
        return q

    def remove_host_listener(self, q: asyncio.Queue[bytes]) -> None:
        with self._lock:
            self._host_listeners.discard(q)

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "input": self.input_ok,
            "output": self.output_ok,
            "sampleRate": self.sample_rate,
            "blocksize": self.blocksize,
            "inputRms": round(self._last_input_rms, 4),
            "outputRms": round(self._last_output_rms, 4),
            "clientSpeaking": time.monotonic() < self._client_speaking_until,
            "hostSpeaking": time.monotonic() < self._host_speaking_until,
            "devices": self._device_summary,
        }

    def stop(self) -> None:
        for stream in (self._input_stream, self._output_stream):
            if stream is not None:
                with suppress(Exception):
                    stream.stop()
                with suppress(Exception):
                    stream.close()


def drop_put_nowait(q: asyncio.Queue[bytes], data: bytes) -> None:
    if q.full():
        with suppress(asyncio.QueueEmpty):
            q.get_nowait()
    with suppress(asyncio.QueueFull):
        q.put_nowait(data)


def pcm16_rms(data: bytes) -> float:
    if not data:
        return 0.0
    if np is not None:
        arr = np.frombuffer(data, dtype=np.int16)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((arr.astype(np.float32) / 32768.0) ** 2)))
    count = min(len(data) // 2, 512)
    if count <= 0:
        return 0.0
    total = 0.0
    for i in range(count):
        sample = int.from_bytes(data[i * 2 : i * 2 + 2], "little", signed=True) / 32768.0
        total += sample * sample
    return math.sqrt(total / count)


def verify_firebase_token_sync(token: str) -> Optional[str]:
    if not token:
        return None
    cached = _AUTH_CACHE.get(token)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
    body = json.dumps({"idToken": token}).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        if VERBOSE:
            print(f"firebase auth failed: {exc}")
        return None
    users = payload.get("users") or []
    if not users:
        return None
    user = users[0]
    uid = user.get("localId")
    if not uid:
        return None
    _AUTH_CACHE[token] = (time.monotonic() + 120, uid)
    return uid


async def websocket_firebase_uid(ws: WebSocket) -> Optional[str]:
    token = ws.query_params.get("token", "")
    uid = await asyncio.to_thread(verify_firebase_token_sync, token)
    if not uid:
        await ws.close(code=1008, reason="login required")
        return None
    return uid


camera = ThreadedCamera(VIDEO_DEVICE, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, JPEG_QUALITY)
motors = MotorDriver()
audio = AudioManager(AUDIO_RATE, AUDIO_BLOCK)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    camera.start()
    audio.start(asyncio.get_running_loop())
    print(f"Open http://127.0.0.1:{PORT} on this machine, or http://<host-ip>:{PORT} on your LAN.")
    try:
        yield
    finally:
        print("Shutting down hardware streams.")
        motors.close()
        audio.stop()
        camera.stop()


app = FastAPI(title="RC Teleoperation", lifespan=lifespan)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>RC Teleoperation</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08090b;
      --panel: #11151a;
      --panel-2: #171d24;
      --line: #26313b;
      --text: #edf2f7;
      --muted: #94a3b8;
      --good: #42d392;
      --warn: #f6c177;
      --bad: #ff6b6b;
      --blue: #5db7ff;
      --cyan: #2dd4bf;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      height: 100%;
      min-height: 0;
    }
    .stage {
      position: relative;
      min-width: 0;
      min-height: 0;
      background: #050607;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
    }
    .topbar {
      position: absolute;
      left: 16px;
      top: 16px;
      right: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(8, 9, 11, .72);
      backdrop-filter: blur(10px);
      border-radius: 8px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--bad);
      box-shadow: 0 0 14px currentColor;
      color: var(--bad);
    }
    .dot.ok { background: var(--good); color: var(--good); }
    aside {
      min-height: 0;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 720;
    }
    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 12px;
    }
    .row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 30px;
      font-size: 13px;
      color: var(--muted);
    }
    .value {
      color: var(--text);
      font-variant-numeric: tabular-nums;
      text-align: right;
      overflow-wrap: anywhere;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    .key {
      aspect-ratio: 1.25;
      border: 1px solid var(--line);
      background: #0c1015;
      color: var(--text);
      border-radius: 8px;
      font-size: 18px;
      font-weight: 750;
      display: grid;
      place-items: center;
      user-select: none;
    }
    .key.active { border-color: var(--blue); background: #12304a; }
    .spacer { visibility: hidden; }
    button {
      min-height: 42px;
      border: 1px solid var(--line);
      background: #0c1015;
      color: var(--text);
      border-radius: 8px;
      font: inherit;
      cursor: pointer;
    }
    button.on { border-color: var(--good); background: #0e3326; }
    button.warn { border-color: var(--warn); }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .locked {
      opacity: .45;
      pointer-events: none;
    }
    input {
      min-height: 38px;
      width: 100%;
      border: 1px solid var(--line);
      background: #090c10;
      color: var(--text);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
    }
    .authGrid {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .buttonGrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .meter {
      height: 8px;
      width: 130px;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      background: #090c10;
    }
    .meter > i {
      display: block;
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--good));
    }
    .joystick {
      position: relative;
      width: min(220px, 68vw);
      aspect-ratio: 1;
      margin: 10px auto 0;
      border-radius: 50%;
      border: 1px solid var(--line);
      background:
        linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px),
        #0a0e13;
      background-size: 50% 50%;
      touch-action: none;
    }
    .stick {
      position: absolute;
      width: 34%;
      aspect-ratio: 1;
      border-radius: 50%;
      left: 33%;
      top: 33%;
      border: 1px solid rgba(255,255,255,.24);
      background: #1d6b89;
      box-shadow: 0 10px 26px rgba(0,0,0,.35);
      transform: translate(0,0);
      pointer-events: none;
    }
    @media (max-width: 860px) {
      body { overflow: auto; }
      main { grid-template-columns: 1fr; height: auto; min-height: 100%; }
      .stage { height: 58vh; min-height: 320px; }
      aside { border-left: 0; border-top: 1px solid var(--line); overflow: visible; }
    }
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <img class="video" src="/video_feed" alt="Live video">
      <div class="topbar">
        <div class="badge"><span id="controlDot" class="dot"></span><span id="controlStatus">control offline</span></div>
        <div class="badge"><span id="fps">video stream</span></div>
      </div>
    </section>
    <aside>
      <header>
        <h1>RC Teleoperation</h1>
        <div class="sub" id="platform">connecting</div>
      </header>

      <section class="panel">
        <div class="row"><span>Login</span><span class="value" id="authStatus">signed out</span></div>
        <div class="authGrid" id="loginPanel">
          <input id="emailInput" type="email" autocomplete="email" placeholder="Email">
          <input id="passwordInput" type="password" autocomplete="current-password" placeholder="Password">
          <button id="loginBtn">Sign In</button>
        </div>
        <button id="logoutBtn" style="display:none;margin-top:10px;width:100%;">Sign Out</button>
      </section>

      <section class="panel">
        <div class="buttonGrid">
          <button id="micBtn" disabled>Enable Mic</button>
          <button id="listenBtn" disabled>Listen Host</button>
        </div>
        <div class="row"><span>Client mic</span><span class="meter"><i id="micMeter"></i></span></div>
        <div class="row"><span>Host mic</span><span class="meter"><i id="hostMeter"></i></span></div>
        <div class="row"><span>Audio</span><span class="value" id="audioStatus">idle</span></div>
      </section>

      <section class="panel locked" id="drivePanel">
        <div class="row"><span>Drive PWM</span><span class="value" id="leftPwm">0.00</span></div>
        <div class="row"><span>Steer PWM</span><span class="value" id="rightPwm">0.00</span></div>
        <div class="row"><span>Input</span><span class="value" id="inputState">neutral</span></div>
        <div class="controls">
          <div class="key spacer"></div><div class="key" data-key="w">W</div><div class="key spacer"></div>
          <div class="key" data-key="a">A</div><div class="key" data-key="s">S</div><div class="key" data-key="d">D</div>
        </div>
        <div class="joystick" id="joy"><div class="stick" id="stick"></div></div>
      </section>
    </aside>
  </main>

  <script src="https://www.gstatic.com/firebasejs/10.12.2/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.12.2/firebase-auth-compat.js"></script>
  <script>
    const firebaseConfig = {
      apiKey: "AIzaSyBX2ybnshzC8rdm2M7RIrU77jva1KqfR3Y",
      authDomain: "levislawns.firebaseapp.com",
      projectId: "levislawns",
      storageBucket: "levislawns.firebasestorage.app",
      messagingSenderId: "514611346211",
      appId: "1:514611346211:web:61b1d48bae07b1f528aa96"
    };
    firebase.initializeApp(firebaseConfig);
    const auth = firebase.auth();

    const AUDIO_RATE = 48000;
    const VOICE_RMS_THRESHOLD = 0.018;
    const VOICE_HOLD_MS = 380;
    const keys = new Set();
    const keyMap = {ArrowUp:"w", ArrowLeft:"a", ArrowDown:"s", ArrowRight:"d", w:"w", a:"a", s:"s", d:"d", W:"w", A:"a", S:"s", D:"d"};
    const els = {
      controlDot: document.getElementById("controlDot"),
      controlStatus: document.getElementById("controlStatus"),
      platform: document.getElementById("platform"),
      leftPwm: document.getElementById("leftPwm"),
      rightPwm: document.getElementById("rightPwm"),
      inputState: document.getElementById("inputState"),
      micBtn: document.getElementById("micBtn"),
      listenBtn: document.getElementById("listenBtn"),
      audioStatus: document.getElementById("audioStatus"),
      micMeter: document.getElementById("micMeter"),
      hostMeter: document.getElementById("hostMeter"),
      authStatus: document.getElementById("authStatus"),
      loginPanel: document.getElementById("loginPanel"),
      emailInput: document.getElementById("emailInput"),
      passwordInput: document.getElementById("passwordInput"),
      loginBtn: document.getElementById("loginBtn"),
      logoutBtn: document.getElementById("logoutBtn"),
      drivePanel: document.getElementById("drivePanel"),
      joy: document.getElementById("joy"),
      stick: document.getElementById("stick")
    };
    let currentUser = null;
    let controlWs, micWs, hostWs, audioCtx, micStream, micNode, micSource;
    let joyVec = {x: 0, y: 0};
    let playTime = 0;
    let remoteSpeakingUntil = 0;
    let localSpeakingUntil = 0;

    function wsUrl(path) {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${location.host}${path}`;
    }

    async function authedWsUrl(path) {
      if (!currentUser) throw new Error("Please sign in first.");
      const token = await currentUser.getIdToken(true);
      return `${wsUrl(path)}?token=${encodeURIComponent(token)}`;
    }

    function displayPlatform(name) {
      if (name === "Darwin") return "macOS";
      if (name === "Linux") return "Linux";
      if (name === "Windows") return "Windows";
      return name || "unknown";
    }

    async function connectControl() {
      if (!currentUser) return;
      if (controlWs && (controlWs.readyState === WebSocket.OPEN || controlWs.readyState === WebSocket.CONNECTING)) return;
      controlWs = new WebSocket(await authedWsUrl("/ws/control"));
      controlWs.onopen = () => {
        els.controlDot.classList.add("ok");
        els.controlStatus.textContent = "control online";
        sendControl();
      };
      controlWs.onclose = () => {
        els.controlDot.classList.remove("ok");
        els.controlStatus.textContent = currentUser ? "control offline" : "login required";
        if (currentUser) setTimeout(() => connectControl().catch(() => {}), 700);
      };
      controlWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.telemetry) {
          els.leftPwm.textContent = msg.telemetry.left.toFixed(2);
          els.rightPwm.textContent = msg.telemetry.right.toFixed(2);
          els.platform.textContent = `${displayPlatform(msg.telemetry.platform)} · ${msg.telemetry.mode}`;
        }
        if (msg.audio) {
          const a = msg.audio;
          const floor = a.clientSpeaking ? "client talking" : (a.hostSpeaking ? "host talking" : "open");
          els.audioStatus.textContent = `${floor} · ${a.input ? "host mic" : "no host mic"} · ${a.output ? "speaker" : "no speaker"}`;
          els.hostMeter.style.width = `${Math.min(100, a.inputRms * 420)}%`;
        }
      };
    }

    function commandVector() {
      if (!currentUser) return {x: 0, y: 0};
      let x = joyVec.x;
      let y = joyVec.y;
      if (keys.has("w")) y += 1;
      if (keys.has("s")) y -= 1;
      if (keys.has("a")) x -= 1;
      if (keys.has("d")) x += 1;
      x = Math.max(-1, Math.min(1, x));
      y = Math.max(-1, Math.min(1, y));
      return {x, y};
    }

    function sendControl() {
      const v = commandVector();
      els.inputState.textContent = `${v.y.toFixed(2)}, ${v.x.toFixed(2)}`;
      document.querySelectorAll(".key[data-key]").forEach(k => k.classList.toggle("active", keys.has(k.dataset.key)));
      if (currentUser && controlWs && controlWs.readyState === WebSocket.OPEN) {
        controlWs.send(JSON.stringify({x: v.x, y: v.y, keys: Array.from(keys), ts: Date.now()}));
      }
    }

    setInterval(sendControl, 60);

    addEventListener("keydown", (e) => {
      if (!currentUser) return;
      if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA")) return;
      const k = keyMap[e.key];
      if (!k) return;
      e.preventDefault();
      keys.add(k);
      sendControl();
    }, {passive: false});
    addEventListener("keyup", (e) => {
      if (document.activeElement && (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA")) return;
      const k = keyMap[e.key];
      if (!k) return;
      e.preventDefault();
      keys.delete(k);
      sendControl();
    }, {passive: false});

    function setupJoystick() {
      let active = false;
      const update = (clientX, clientY) => {
        const r = els.joy.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const max = r.width * 0.34;
        let dx = clientX - cx;
        let dy = clientY - cy;
        const mag = Math.hypot(dx, dy);
        if (mag > max) { dx *= max / mag; dy *= max / mag; }
        joyVec.x = dx / max;
        joyVec.y = -dy / max;
        els.stick.style.transform = `translate(${dx}px, ${dy}px)`;
        sendControl();
      };
      els.joy.addEventListener("pointerdown", e => {
        if (!currentUser) return;
        active = true;
        els.joy.setPointerCapture(e.pointerId);
        update(e.clientX, e.clientY);
      });
      els.joy.addEventListener("pointermove", e => { if (active) update(e.clientX, e.clientY); });
      const end = e => {
        active = false;
        joyVec = {x: 0, y: 0};
        els.stick.style.transform = "translate(0,0)";
        sendControl();
      };
      els.joy.addEventListener("pointerup", end);
      els.joy.addEventListener("pointercancel", end);
    }

    function ensureAudioContext() {
      if (!audioCtx) {
        const AudioCtor = window.AudioContext || window.webkitAudioContext;
        try {
          audioCtx = new AudioCtor({sampleRate: AUDIO_RATE});
        } catch (_) {
          audioCtx = new AudioCtor();
        }
      }
      if (audioCtx.state === "suspended") audioCtx.resume();
      return audioCtx;
    }

    function showAudioError(err, fallback) {
      const message = err && err.message ? err.message : (err ? String(err) : fallback);
      alert(message || fallback);
    }

    async function getClientMedia(constraints) {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return navigator.mediaDevices.getUserMedia(constraints);
      }
      const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
      if (legacyGetUserMedia) {
        return new Promise((resolve, reject) => legacyGetUserMedia.call(navigator, constraints, resolve, reject));
      }
      const secureHint = window.isSecureContext
        ? "This browser does not expose microphone capture."
        : "Microphone capture requires HTTPS, or http://localhost on most browsers.";
      throw new Error(secureHint);
    }

    function downsampleToPCM16(input, inRate, outRate) {
      if (!input.length || !inRate || !outRate) return new ArrayBuffer(0);
      const outLen = Math.max(1, Math.round(input.length * outRate / inRate));
      const pcm = new Int16Array(outLen);
      const scale = inRate / outRate;
      let rms = 0;
      for (let i = 0; i < outLen; i++) {
        const pos = i * scale;
        const idx = Math.floor(pos);
        const frac = pos - idx;
        const a = input[Math.min(idx, input.length - 1)] || 0;
        const b = input[Math.min(idx + 1, input.length - 1)] || a;
        let s = a + (b - a) * frac;
        s = Math.max(-1, Math.min(1, s));
        rms += s * s;
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
      }
      els.micMeter.style.width = `${Math.min(100, Math.sqrt(rms / Math.max(1, outLen)) * 420)}%`;
      return pcm.buffer;
    }

    async function toggleMic() {
      if (!currentUser) throw new Error("Please sign in first.");
      if (micWs) {
        micWs.close();
        micWs = null;
        if (micNode) micNode.disconnect();
        if (micSource) micSource.disconnect();
        if (micStream) micStream.getTracks().forEach(t => t.stop());
        els.micBtn.classList.remove("on");
        els.micBtn.textContent = "Enable Mic";
        return;
      }
      const ctx = ensureAudioContext();
      micStream = await getClientMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        },
        video: false
      });
      micWs = new WebSocket(await authedWsUrl("/ws/audio_in"));
      micWs.binaryType = "arraybuffer";
      await new Promise(resolve => micWs.onopen = resolve);
      micSource = ctx.createMediaStreamSource(micStream);
      micNode = ctx.createScriptProcessor(1024, 1, 1);
      micNode.onaudioprocess = (event) => {
        if (!micWs || micWs.readyState !== WebSocket.OPEN) return;
        if (performance.now() < remoteSpeakingUntil) return;
        const data = event.inputBuffer.getChannelData(0);
        const pcm = downsampleToPCM16(data, ctx.sampleRate, AUDIO_RATE);
        if (!pcm.byteLength) return;
        const rms = pcm16Rms(pcm);
        if (rms > VOICE_RMS_THRESHOLD) localSpeakingUntil = performance.now() + VOICE_HOLD_MS;
        if (rms > 0.004 || performance.now() < localSpeakingUntil) micWs.send(pcm);
      };
      micSource.connect(micNode);
      const silentMonitor = ctx.createGain();
      silentMonitor.gain.value = 0;
      micNode.connect(silentMonitor);
      silentMonitor.connect(ctx.destination);
      els.micBtn.classList.add("on");
      els.micBtn.textContent = "Mic On";
      micWs.onclose = () => {
        micWs = null;
        els.micBtn.classList.remove("on");
        els.micBtn.textContent = "Enable Mic";
      };
    }

    function playPCM16(arrayBuffer) {
      const ctx = ensureAudioContext();
      const pcm = new Int16Array(arrayBuffer);
      if (!pcm.length) return;
      const packetRms = pcm16Rms(arrayBuffer);
      if (packetRms > VOICE_RMS_THRESHOLD) remoteSpeakingUntil = performance.now() + VOICE_HOLD_MS;
      const buffer = ctx.createBuffer(1, pcm.length, AUDIO_RATE);
      const out = buffer.getChannelData(0);
      let rms = 0;
      for (let i = 0; i < pcm.length; i++) {
        const s = pcm[i] / 32768;
        out[i] = s;
        rms += s * s;
      }
      els.hostMeter.style.width = `${Math.min(100, Math.sqrt(rms / pcm.length) * 420)}%`;
      const src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      if (playTime < ctx.currentTime || playTime - ctx.currentTime > 0.35) {
        playTime = ctx.currentTime + 0.045;
      }
      const startAt = Math.max(ctx.currentTime + 0.025, playTime);
      src.start(startAt);
      playTime = startAt + buffer.duration;
      if (playTime - ctx.currentTime > 0.32) playTime = ctx.currentTime + 0.06;
    }

    function pcm16Rms(arrayBuffer) {
      const pcm = new Int16Array(arrayBuffer);
      if (!pcm.length) return 0;
      let total = 0;
      for (let i = 0; i < pcm.length; i++) {
        const s = pcm[i] / 32768;
        total += s * s;
      }
      return Math.sqrt(total / pcm.length);
    }

    async function toggleListen() {
      if (!currentUser) throw new Error("Please sign in first.");
      if (hostWs) {
        hostWs.close();
        hostWs = null;
        els.listenBtn.classList.remove("on");
        els.listenBtn.textContent = "Listen Host";
        return;
      }
      ensureAudioContext();
      hostWs = new WebSocket(await authedWsUrl("/ws/audio_out"));
      hostWs.binaryType = "arraybuffer";
      hostWs.onmessage = event => playPCM16(event.data);
      hostWs.onopen = () => {
        playTime = audioCtx.currentTime + 0.03;
        els.listenBtn.classList.add("on");
        els.listenBtn.textContent = "Listening";
      };
      hostWs.onclose = () => {
        hostWs = null;
        els.listenBtn.classList.remove("on");
        els.listenBtn.textContent = "Listen Host";
      };
    }

    els.micBtn.addEventListener("click", () => toggleMic().catch(err => showAudioError(err, "Microphone could not be started.")));
    els.listenBtn.addEventListener("click", () => toggleListen().catch(err => showAudioError(err, "Host audio could not be started.")));
    els.loginBtn.addEventListener("click", () => {
      auth.signInWithEmailAndPassword(els.emailInput.value.trim(), els.passwordInput.value)
        .catch(err => showAudioError(err, "Login failed."));
    });
    els.logoutBtn.addEventListener("click", () => auth.signOut());
    auth.onAuthStateChanged(user => {
      currentUser = user;
      els.authStatus.textContent = user ? (user.email || "signed in") : "signed out";
      els.loginPanel.style.display = user ? "none" : "grid";
      els.logoutBtn.style.display = user ? "block" : "none";
      els.micBtn.disabled = !user;
      els.listenBtn.disabled = !user;
      els.drivePanel.classList.toggle("locked", !user);
      if (user) {
        connectControl().catch(err => showAudioError(err, "Control connection failed."));
      } else {
        keys.clear();
        if (controlWs) controlWs.close();
        if (micWs) micWs.close();
        if (hostWs) hostWs.close();
        els.controlStatus.textContent = "login required";
        els.controlDot.classList.remove("ok");
      }
    });
    setupJoystick();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/status")
async def status() -> Dict[str, Any]:
    return {"telemetry": motors.telemetry(), "audio": audio.status()}


async def mjpeg_generator():
    boundary = b"--frame"
    delay = 1.0 / max(VIDEO_FPS, 1)
    last_frame = None
    try:
        while True:
            frame = camera.frame()
            if frame is not last_frame:
                last_frame = frame
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nCache-Control: no-store, no-cache, must-revalidate\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(delay)
    except (asyncio.CancelledError, Exception):
        return


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Connection": "close",
        },
    )


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket) -> None:
    await ws.accept()
    uid = await websocket_firebase_uid(ws)
    if not uid:
        return
    await ws.send_json({"telemetry": motors.telemetry(), "audio": audio.status()})
    last_msg = time.monotonic()
    watchdog_task = asyncio.create_task(control_watchdog(ws, lambda: last_msg))
    try:
        while True:
            raw = await ws.receive_text()
            last_msg = time.monotonic()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            x = clamp(float(msg.get("x", msg.get("turn", 0.0))))
            y = clamp(float(msg.get("y", msg.get("throttle", 0.0))))
            telemetry = motors.drive(y, x)
            await ws.send_json({"telemetry": telemetry, "audio": audio.status()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"control websocket error: {exc}")
    finally:
        watchdog_task.cancel()
        with suppress(Exception):
            motors.stop()


async def control_watchdog(ws: WebSocket, last_msg_getter: Any) -> None:
    try:
        while True:
            await asyncio.sleep(0.1)
            if time.monotonic() - last_msg_getter() > 0.65:
                motors.stop()
    except asyncio.CancelledError:
        return


@app.websocket("/ws/audio_in")
async def ws_audio_in(ws: WebSocket) -> None:
    await ws.accept()
    uid = await websocket_firebase_uid(ws)
    if not uid:
        return
    try:
        while True:
            data = await ws.receive_bytes()
            audio.push_speaker_pcm(data)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"browser-to-host audio websocket error: {exc}")


@app.websocket("/ws/audio_out")
async def ws_audio_out(ws: WebSocket) -> None:
    await ws.accept()
    uid = await websocket_firebase_uid(ws)
    if not uid:
        return
    listener = audio.add_host_listener()
    try:
        while True:
            data = await listener.get()
            await ws.send_bytes(data)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"host-to-browser audio websocket error: {exc}")
    finally:
        audio.remove_host_listener(listener)


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; exiting.")
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(Exception):
            signal.signal(sig, handle_signal)


def main() -> None:
    install_signal_handlers()
    quiet_stream_disconnect_logs()
    try:
        import uvicorn
    except Exception:
        print("uvicorn is required. Install with: pip install uvicorn", file=sys.stderr)
        raise
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("RC_LOG_LEVEL", "warning"),
        access_log=VERBOSE,
    )


if __name__ == "__main__":
    main()
