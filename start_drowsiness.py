#!/usr/bin/env python3
import os
import time
import sys
import logging
from datetime import datetime

# Setup logging
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness_log.txt")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Wait for system to fully boot before starting
time.sleep(10)

try:
    logging.info("Starting drowsiness detection service")
    
    # Import your application modules
    import cv2
    import mediapipe as mp
    from drowsy_detection import VideoFrameHandler
    
    # Initialize video handler
    video_handler = VideoFrameHandler()
    thresholds = {
        "EAR_THRESH": 0.20,
        "WAIT_TIME": 3.0,
    }
    
    # Retry camera connection if it fails initially
    max_retries = 5
    retries = 0
    cap = None
    
    while retries < max_retries:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            logging.warning(f"Camera not available, retry {retries+1}/{max_retries}")
            retries += 1
            time.sleep(3)
        except Exception as e:
            logging.error(f"Error connecting to camera: {e}")
            retries += 1
            time.sleep(3)
    
    if not cap or not cap.isOpened():
        logging.error("Failed to open camera after multiple attempts")
        sys.exit(1)
    
    logging.info("Camera connected successfully")
    
    # Main loop
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame, retrying...")
                time.sleep(1)
                continue
            
            processed_frame, alarm_on = video_handler.process(frame, thresholds)
            
            if alarm_on:
                logging.info("ALARM: DROWSINESS DETECTED!")
            
        except KeyboardInterrupt:
            logging.info("Service stopped by user")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(1)
    
except Exception as e:
    logging.critical(f"Critical error: {e}")
finally:
    if 'cap' in locals() and cap:
        cap.release()
    logging.info("Service stopped")