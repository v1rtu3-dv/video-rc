import cv2
import os
import subprocess
import shutil
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# --- Raspberry Pi GPIO Hardware Setup ---
try:
    import RPi.GPIO as GPIO
    IS_PI = True
except ImportError:
    IS_PI = False
    print("ℹ️ Running in Mock Mode (Non-Pi Environment). Motor signals will print to console.")

# L298N Motor Driver Pin Map (BCM Pin Numbers)
# Left Motor (IN1, IN2, ENA) | Right Motor (IN3, IN4, ENB)
IN1, IN2, ENA = 17, 27, 22
IN3, IN4, ENB = 23, 24, 25

pwm_a = None
pwm_b = None

def setup_gpio():
    global pwm_a, pwm_b
    if not IS_PI:
        return
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Configure Direction & Enable Pins as Outputs
    pins = [IN1, IN2, ENA, IN3, IN4, ENB]
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    # Enable PWM Speed Control (100 Hz frequency)
    pwm_a = GPIO.PWM(ENA, 100)
    pwm_b = GPIO.PWM(ENB, 100)
    pwm_a.start(75)  # Default 75% speed
    pwm_b.start(75)

def drive_motors(command: str):
    """
    Translates WASD keyboard strings into L298N H-Bridge logic states.
    """
    if not IS_PI:
        print(f"[MOCK MOTOR ACTION]: Executing command '{command}'")
        return

    # Clear all direction pins
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

    if 'w' in command and 'a' in command:      # Forward-Left
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(35)
        pwm_b.ChangeDutyCycle(85)
    elif 'w' in command and 'd' in command:    # Forward-Right
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(85)
        pwm_b.ChangeDutyCycle(35)
    elif 'w' in command:                       # Forward
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(80)
        pwm_b.ChangeDutyCycle(80)
    elif 's' in command:                       # Reverse
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN4, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(70)
        pwm_b.ChangeDutyCycle(70)
    elif 'a' in command:                       # Spin Left in Place
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(75)
        pwm_b.ChangeDutyCycle(75)
    elif 'd' in command:                       # Spin Right in Place
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN4, GPIO.HIGH)
        pwm_a.ChangeDutyCycle(75)
        pwm_b.ChangeDutyCycle(75)
    else:                                      # Stop Motors
        pwm_a.ChangeDutyCycle(0)
        pwm_b.ChangeDutyCycle(0)

# Initialize Hardware Setup
setup_gpio()

# --- FastAPI Web App Initialization ---
app = FastAPI()

# Camera Configuration
camera = cv2.VideoCapture(1)  # Index 1 on Mac; change to 0 if needed on Pi
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_video_frames():
    """Captures camera frames, encodes to JPEG, and yields HTTP boundary chunks."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Compress frame to JPEG (Quality 60 keeps latency sub-100ms over ngrok)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(file_path, "r") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Operator connected via WebSocket")

    # Audio playback process setup (Pipes audio directly to speaker)
    audio_process = None
    if shutil.which("ffplay"):
        audio_process = subprocess.Popen(
            [
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-probesize", "32", "-analyzeduration", "0", "-i", "pipe:0"
            ],
            stdin=subprocess.PIPE
        )

    try:
        while True:
            message = await websocket.receive()
            
            # 1. Drive Commands
            if "text" in message:
                payload = json.loads(message["text"])
                if payload.get("type") == "drive":
                    command = payload.get("command", "stop")
                    drive_motors(command)
                
            # 2. Downward Voice Stream
            elif "bytes" in message:
                audio_chunk = message["bytes"]
                if audio_process and audio_process.stdin:
                    try:
                        audio_process.stdin.write(audio_chunk)
                        audio_process.stdin.flush()
                    except BrokenPipeError:
                        audio_process = subprocess.Popen(
                            [
                                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                                "-probesize", "32", "-analyzeduration", "0", "-i", "pipe:0"
                            ],
                            stdin=subprocess.PIPE
                        )
                        audio_process.stdin.write(audio_chunk)
                        audio_process.stdin.flush()

    except WebSocketDisconnect:
        print("Operator disconnected")
        drive_motors("stop")
        if audio_process:
            audio_process.terminate()

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        if IS_PI:
            GPIO.cleanup()
