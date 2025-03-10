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
vid = cv2.VideoCapture(0)

# Optional: Create a video writer to save output
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

print("Starting drowsiness detection. Press Ctrl+C to exit.")
try:
    while True:
        grabbed, img = vid.read()
        if not grabbed:
            break
        
        # Process the frame (no display needed)
        processed_frame, alarm_on = video_handler.process(img, thresholds)
        
        # Optional: Save to video file
        # out.write(processed_frame)
        
        # Optional: Save periodic screenshots
        # if int(time.time()) % 5 == 0:  # Save every 5 seconds
        #     cv2.imwrite(f'frame_{int(time.time())}.jpg', processed_frame)
        
        # Print status to console instead of showing window
        if alarm_on:
            print("\rALARM: DROWSINESS DETECTED!", end="")
        else:
            print("\rMonitoring... (Press Ctrl+C to exit)", end="")
            
except KeyboardInterrupt:
    print("\nExiting drowsiness detection.")
finally:
    vid.release()
    # if 'out' in locals():
    #     out.release()
