import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")   # path to your model

# Load image
image_path = "test.jpg"  # put your image path here
img = cv2.imread(image_path)

# Run detection
results = model(img)

# Draw detections
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

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

# Show output
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
