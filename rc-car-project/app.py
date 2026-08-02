import cv2
import os
import subprocess
import shutil
import json
import asyncio
import threading
import time
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False
    print("ℹ️ Running in Mock Mode (Non-Pi Environment). Motor signals will print to console.")

IN1, IN2, ENA = 17, 27, 22
IN3, IN4, ENB = 23, 24, 25

pwm_a, pwm_b = None, None

def setup_gpio():
    global pwm_a, pwm_b
    if not IS_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    pins = [IN1, IN2, ENA, IN3, IN4, ENB]
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    pwm_a = GPIO.PWM(ENA, 100)
    pwm_b = GPIO.PWM(ENB, 100)
    pwm_a.start(75)
    pwm_b.start(75)

def drive_motors(command: str):
    if not IS_PI:
        print(f"🚗 [MOCK MOTOR ACTION]: Executing command '{command}'")
        return

    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.LOW)

    if 'w' in command and 'a' in command:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(35); pwm_b.ChangeDutyCycle(85)
    elif 'w' in command and 'd' in command:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(85); pwm_b.ChangeDutyCycle(35)
    elif 'w' in command:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(80); pwm_b.ChangeDutyCycle(80)
    elif 's' in command:
        GPIO.output(IN2, GPIO.HIGH); GPIO.output(IN4, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(70); pwm_b.ChangeDutyCycle(70)
    elif 'a' in command:
        GPIO.output(IN2, GPIO.HIGH); GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(75); pwm_b.ChangeDutyCycle(75)
    elif 'd' in command:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN4, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(75); pwm_b.ChangeDutyCycle(75)
    else:
        pwm_a.ChangeDutyCycle(0); pwm_b.ChangeDutyCycle(0)

setup_gpio()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera = cv2.VideoCapture(0, cv2.CAP_V4L2 if IS_PI else cv2.CAP_ANY)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
def camera_loop():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            latest_frame = frame
        time.sleep(0.033) # Strictly paced to ~30 FPS

threading.Thread(target=camera_loop, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(file_path, "r") as f:
        return f.read()

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Operator connected via WebSocket!")

    if IS_PI:
        # Raspberry Pi (ALSA settings with low-latency flags)
        record_cmd = [
            "ffmpeg", "-f", "alsa", "-fflags", "nobuffer", "-flags", "low_delay",
            "-thread_queue_size", "32", "-flush_packets", "1", "-i", "default",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"
        ]
        play_cmd = [
            "ffmpeg", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
            "-f", "alsa", "-fflags", "nobuffer", "-flags", "low_delay",
            "-thread_queue_size", "32", "-flush_packets", "1", "default"
        ]
    else:
        # macOS (AVFoundation settings using index :1 for MacBook Air Microphone)
        record_cmd = [
            "ffmpeg", "-f", "avfoundation", "-i", ":1",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"
        ]
        # macOS output can bypass local FFmpeg player process since the browser handles audio web contexts directly
        play_cmd = None

    audio_process = await asyncio.create_subprocess_exec(
        *record_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    ) if record_cmd and shutil.which("ffmpeg") else None

    player_process = subprocess.Popen(
        play_cmd,
        stdin=subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    ) if play_cmd and shutil.which("ffmpeg") else None

    async def stream_audio():
        while audio_process and audio_process.returncode is None:
            try:
                chunk = await audio_process.stdout.read(256)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)
            except Exception:
                break

    async def stream_video():
        while True:
            try:
                if latest_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
                    b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                    await websocket.send_text(json.dumps({"type": "video", "data": b64_frame}))
                await asyncio.sleep(0.033) # 30 FPS pacing
            except Exception:
                break

    audio_task = asyncio.create_task(stream_audio())
    video_task = asyncio.create_task(stream_video())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message:
                payload = json.loads(message["text"])
                if payload.get("type") == "drive":
                    drive_motors(payload.get("command", "stop"))

            elif "bytes" in message and player_process and player_process.stdin:
                try:
                    player_process.stdin.write(message["bytes"])
                    player_process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        print("❌ Operator disconnected.")
        audio_task.cancel()
        video_task.cancel()
        drive_motors("stop")
        if audio_process:
            try: audio_process.terminate()
            except Exception: pass
        if player_process:
            player_process.terminate()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=None
    )
