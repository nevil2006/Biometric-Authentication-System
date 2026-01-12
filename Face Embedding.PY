import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Load YOLO model
model = YOLO(r"C:\Users\ADMIN\OneDrive\Desktop\biometric autheication\best.pt")

# Load InsightFace model
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load image
image_path = r"C:\Users\ADMIN\OneDrive\Desktop\biometric autheication\Real Images\000001.jpg"
img = cv2.imread(image_path)

# Run YOLO detection
results = model(img)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # ---- CROP FACE ----
        face_crop = img[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # ---- GENERATE EMBEDDING ----
        faces = face_app.get(face_crop)

        if len(faces) == 0:
            continue

        embedding = faces[0].embedding  # 512D vector

        print("Embedding shape:", embedding.shape)
        print("First 10 values:", embedding[:10])

        # Draw box
        label = f"Face {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

output_path = "output_result.jpg"
cv2.imwrite(output_path, img)
print("Result saved as:", output_path)
