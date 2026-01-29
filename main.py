import cv2
from YOLO import detect_face
from liveness import LivenessDetector

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create FULL SCREEN window
window_name = "Biometric Authentication"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

liveness = LivenessDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)

    if face is not None:
        if liveness.is_live(face):
            text = "LIVE PERSON"
            color = (0, 255, 0)
        else:
            text = "CHECKING / FAKE"
            color = (0, 0, 255)
    else:
        text = "NO FACE DETECTED"
        color = (255, 0, 0)

    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow(window_name, frame)

    # Press Q or ESC to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
