import cv2
import numpy as np

class LivenessDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.blink_counter = 0
        self.closed_eyes_frames = 0
        self.EYES_CLOSED_FRAMES = 2
        self.BLINK_THRESHOLD = 1

    def is_live(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) == 0:
            self.closed_eyes_frames += 1
        else:
            if self.closed_eyes_frames >= self.EYES_CLOSED_FRAMES:
                self.blink_counter += 1
            self.closed_eyes_frames = 0

        return self.blink_counter >= self.BLINK_THRESHOLD
