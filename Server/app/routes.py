# app/routes.py
from flask import Blueprint, render_template, Response
import cv2
import numpy as np
import time
import threading
from queue import Queue, Full
from ultralytics import YOLO
import os
import logging
from collections import deque
import torch
import VideoCaptureCV
main = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraMonitor')

# Paths and configuration
MODEL_PATH = os.path.join(os.getcwd(), 'server', 'app', 'model-weights', 
                         'yolov12s13032025_000207_ver-dataset6', '_ver2',
                         'runs', 'detect', 'train', 'weights', 'best.pt')
RTSP_URL = 'rtsp://admin:L2FCD876@172.16.15.104:554/cam/realmonitor?channel=1&subtype=0&tcp'
FRAME_SIZE = (640, 480)
MAX_QUEUE_SIZE = 2  # Increased queue size for better buffer management

class VideoCaptureCV:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = True
        self.queue = Queue(maxsize=5)  # Larger buffer for network fluctuations
        self.reconnect_attempts = 0
        self.connect()
        self.last_valid_frame = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.reconnect_attempts = 0
            logger.info("RTSP connection established")
        else:
            logger.error("Failed to connect to RTSP stream")
            self.reconnect_attempts += 1

    def read_frames(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.handle_reconnection()
                continue

            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame read error, attempting reconnect...")
                self.handle_reconnection()
                continue

            try:
                frame = cv2.resize(frame, FRAME_SIZE)
                if self.queue.full():
                    self.queue.get_nowait()  # Discard oldest frame
                self.queue.put(frame)
                self.last_valid_frame = frame
                self.reconnect_attempts = 0
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                time.sleep(0.1)

    def handle_reconnection(self):
        backoff = min(2 ** self.reconnect_attempts, 30)
        logger.info(f"Reconnecting in {backoff}s...")
        time.sleep(backoff)
        self.connect()

    def read(self):
        return self.queue.get() if not self.queue.empty() else self.last_valid_frame

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Video connection released")

class HybridProcessor:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.half = self.device.startswith('cuda')
            self.model.to(self.device)
            self.model.fuse()
            self.model.conf = 0.3  # Increased confidence threshold
            self.model.iou = 0.45  # Adjusted IoU threshold
            logger.info(f"YOLO model loaded on {self.device}, FP16: {self.half}")
        except Exception as e:
            logger.critical(f"Model load failed: {str(e)}")
            raise

        self.queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.output_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.thread = threading.Thread(target=self.process_batch, daemon=True)
        self.thread.start()

    def process_batch(self):
        while True:
            frames = []
            while len(frames) < 4 and not self.queue.empty():  # Small batch processing
                frames.append(self.queue.get())
            
            if frames:
                try:
                    results = self.model(frames, imgsz=640, verbose=False, half=self.half)
                    for frame, result in zip(frames, results):
                        processed = self.draw_detections(frame, result) # Vẽ bouding box
                        self.output_queue.put(processed)
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    for frame in frames:
                        self.output_queue.put(frame)

    def draw_detections(self, frame, results):
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                cls_id = int(box.cls)
                label = f"{results.names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def put_frame(self, frame):
        try:
            self.queue.put_nowait(frame)
        except Full:
            pass  # Intentional frame drop under heavy load

    def get_frame(self):
        return self.output_queue.get_nowait() if not self.output_queue.empty() else None

# Initialize system components
try:
    video_capture = VideoCaptureCV(RTSP_URL)
    cv_thread = threading.Thread(target=video_capture.read_frames, daemon=True)
    cv_thread.start()
    yolo_processor = HybridProcessor(MODEL_PATH)
except Exception as e:
    logger.critical(f"System initialization failed: {str(e)}")
    exit(1)

def processing_loop():
    while True:
        frame = video_capture.read()
        if frame is not None:
            yolo_processor.put_frame(frame)
        time.sleep(0.001)  # Reduced sleep for better responsiveness

threading.Thread(target=processing_loop, daemon=True).start()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/video_feed')
def video_feed():
    def generate():
        fps_history = deque(maxlen=10)
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            frame = yolo_processor.get_frame()
            
            if frame is not None:
                # Calculate dynamic FPS
                delta = current_time - last_frame_time
                fps = 1 / delta if delta > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history) if fps_history else 0
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Optimized JPEG encoding
                _, jpeg = cv2.imencode('.jpg', frame, 
                                      [int(cv2.IMWRITE_JPEG_QUALITY), 75,
                                       int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' 
                      + jpeg.tobytes() + b'\r\n')
                last_frame_time = current_time
            else:
                time.sleep(0.005)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')