import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import mediapipe as mp
import time
import numpy as np

from drowsy_detection import VideoFrameHandler

video_handler = VideoFrameHandler()
thresholds = {
    "EAR_THRESH": 0.20,  # Eyes are considered closed if EAR is below this value.
    "WAIT_TIME": 3.0,    # Seconds closed to trigger the drowsiness alarm.
}
vid = cv2.VideoCapture(1)

def video_frame_callback(frame):
    framex, play_alarm = video_handler.process(frame, thresholds)  # Process the frame
    return framex

while True:
    grabbed, img = vid.read()
    if not grabbed:
        break
    imgx = video_frame_callback(img)
    cv2.imshow('Video', imgx)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
