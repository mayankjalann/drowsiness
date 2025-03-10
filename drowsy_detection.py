import cv2
import time
import numpy as np
import mediapipe as mp
import platform
import os
import subprocess
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

# Set the full path to your custom audio file.
# On your Raspberry Pi, update this path if needed.
AUDIO_FILE = "/home/ragpi/drowsy/beep1.mp3"

# Initialize platform-specific audio modules.
if platform.system() == "Windows":
    import winsound
elif platform.system() == "Linux":
    try:
        import pygame
        pygame.init()
        pygame.mixer.init()
        BEEP_SOUND = pygame.mixer.Sound(AUDIO_FILE)
    except Exception as e:
        print("Error initializing pygame for audio:", e)
        BEEP_SOUND = None

def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh

def distance(point_1, point_2):
    return sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)
        # Compute distances between landmarks for EAR calculation.
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except Exception as e:
        print("Error in get_ear:", e)
        ear = 0.0
        coords_points = None
    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_coords = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_coords = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0
    return Avg_EAR, (left_coords, right_coords)

def plot_eye_landmarks(frame, left_coords, right_coords, color):
    for coords in [left_coords, right_coords]:
        if coords:
            for coord in coords:
                cv2.circle(frame, coord, 2, color, -1)
    # Flip the frame for a selfie-view display.
    frame = cv2.flip(frame, 1)
    return frame

def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    return cv2.putText(image, text, origin, font, fntScale, color, thickness)

class VideoFrameHandler:
    def __init__(self):
        # Landmark indices for left and right eyes.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)    # Color for closed/drowsy state.
        self.GREEN = (0, 255, 0)  # Color for open/alert state.

        # Initialize Mediapipe FaceMesh model.
        try:
            self.facemesh_model = get_mediapipe_app()
        except Exception as e:
            print("Error initializing Mediapipe FaceMesh:", e)
            exit()

        # State tracker for drowsiness.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,      # Total time (in seconds) with eyes closed.
            "COLOR": self.GREEN,
            "play_alarm": False,
        }
        self.EAR_txt_pos = (10, 30)

        # Variables for blink detection.
        self.BLINK_CONSEC_FRAMES = 3   # Maximum consecutive frames for a closure to be considered a blink.
        self.closed_frames = 0         # Counter for consecutive frames with eyes closed.
        self.blink_counter = 0         # Total blink count.

        # For computing blink rate.
        self.session_start_time = time.perf_counter()  # Marks the beginning of the session.
        self.blink_rate_threshold = 15.0  # Minimum acceptable blink rate (blinks per minute).

        # Time tracker for beeps.
        self.last_beep_time = 0

    def process(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = True
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
        BLINK_RATE_txt_pos = (10, int(frame_h // 2 * 1.95))

        results = self.facemesh_model.process(frame)
        current_time = time.perf_counter()

        low_blink_alarm = False  # Flag to trigger beep on low blink rate.

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coords = calculate_avg_ear(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h,
            )
            frame = plot_eye_landmarks(frame, coords[0], coords[1], self.state_tracker["COLOR"])

            # ----- Blink Detection -----
            if EAR < thresholds["EAR_THRESH"]:
                self.closed_frames += 1
            else:
                # Count as a blink if eyes were closed for a short period.
                if 0 < self.closed_frames <= self.BLINK_CONSEC_FRAMES:
                    self.blink_counter += 1
                self.closed_frames = 0

            # ----- Drowsiness Detection -----
            if EAR < thresholds["EAR_THRESH"]:
                self.state_tracker["DROWSY_TIME"] += current_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = current_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, self.state_tracker["COLOR"])
            else:
                self.state_tracker["start_time"] = current_time
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            # ----- Blink Rate Calculation -----
            elapsed_time = current_time - self.session_start_time
            blink_rate = (self.blink_counter / elapsed_time) * 60 if elapsed_time > 10 else 0

            if elapsed_time > 10 and blink_rate < self.blink_rate_threshold:
                low_blink_alarm = True
                low_blink_alert = f"Low blink rate! ({blink_rate:.1f} BPM) Blink more!"
                plot_text(frame, low_blink_alert, BLINK_RATE_txt_pos, self.RED)
            else:
                blink_rate = f"{blink_rate:.1f}" if elapsed_time > 10 else "Calculating..."

            # ----- Overlay Text Information -----
            EAR_txt = f"Probability: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"Drowsy: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            blink_txt = f"Blinks: {self.blink_counter}"
            blink_rate_txt = f"Blink Rate: {blink_rate} BPM"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, blink_txt, (10, 100), self.state_tracker["COLOR"])
            plot_text(frame, blink_rate_txt, (10, 140), self.state_tracker["COLOR"])
        else:
            # Reset timers if no face is detected.
            self.state_tracker["start_time"] = current_time
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            frame = cv2.flip(frame, 1)

        # ----- Beep Sound Trigger -----
        if self.state_tracker["play_alarm"] or low_blink_alarm:
            if current_time - self.last_beep_time >= 1:
                try:
                    if platform.system() == "Windows":
                        winsound.Beep(1000, 200)
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["afplay", AUDIO_FILE])
                    elif platform.system() == "Linux":
                        if 'BEEP_SOUND' in globals() and BEEP_SOUND:
                            try:
                                BEEP_SOUND.play()
                            except Exception:
                                # Fallback methods for headless Pi
                                try:
                                    # Try using system beep
                                    os.system('echo -e "\a"')
                                except:
                                    # Last resort - just log it
                                    pass
                        else:
                            try:
                                # Try different audio players
                                subprocess.Popen(["aplay", "-q", AUDIO_FILE], 
                                                stderr=subprocess.DEVNULL)
                            except:
                                try:
                                    subprocess.Popen(["mpg123", "-q", AUDIO_FILE], 
                                                    stderr=subprocess.DEVNULL)
                                except:
                                    # If all audio methods fail, just log
                                    pass
                    else:
                        os.system('echo -e "\a"')
                except Exception as e:
                    # Don't print to console, just continue
                    pass
                self.last_beep_time = current_time

        return frame, self.state_tracker["play_alarm"]

if __name__ == "__main__":
    # Define thresholds for EAR and wait time (seconds).
    thresholds = {
        "EAR_THRESH": 0.25,
        "WAIT_TIME": 2.0,  # seconds of eyes closed before alarm is triggered
    }

    video_handler = VideoFrameHandler()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open video capture device.")
        exit()

    print("Starting video capture. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        processed_frame, alarm_on = video_handler.process(frame, thresholds)
        cv2.imshow("Drowsiness Detector", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()