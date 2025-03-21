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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CameraMonitor')
FRAME_SIZE = (640, 640)

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
