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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraMonitor')


FRAME_SIZE = (640, 640)
MAX_QUEUE_SIZE = 2  # Increased queue size for better buffer management

class HybridProcessor:
    def __init__(self, model_path):
        try:
            # self.model = YOLO(model_path) // Phiên bản yolo cao hơn v5
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
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