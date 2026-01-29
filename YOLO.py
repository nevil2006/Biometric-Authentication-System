import cv2
from ultralytics import YOLO

# Load your trained model ONCE
model = YOLO(r"C:\Users\ADMIN\OneDrive\Desktop\biometric autheication\best.pt")

def detect_face(frame):
    """
    Takes a frame (from camera or image)
    Returns cropped face image or None
    """

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop face region
            face = frame[y1:y2, x1:x2]

            return face  # return FIRST detected face

    return None
