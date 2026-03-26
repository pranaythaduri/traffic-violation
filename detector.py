import cv2
import numpy as np
import os
from ultralytics import YOLO

# -------------------------------
# Ensure output directory exists
# -------------------------------
os.makedirs("output", exist_ok=True)

# -------------------------------
# Load YOLOv8 model
# -------------------------------
model = YOLO("models/yolov8n.pt")

# -------------------------------
# Load video
# -------------------------------
cap = cv2.VideoCapture("videos/traffic1.mp4")

if not cap.isOpened():
    print("❌ Error: Video file not found or cannot be opened.")
    exit()

# -------------------------------
# Video properties
# -------------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# -------------------------------
# Video writer
# -------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output/result.mp4",
    fourcc,
    fps if fps > 0 else 20,
    (width, height)
)

# -------------------------------
# Traffic signal simulation
# -------------------------------
signal_color = "RED"  # Change to GREEN for testing

# Wrong-way reference direction
REFERENCE_DIRECTION = "LEFT_TO_RIGHT"

def detect_direction(x1, x2):
    return "LEFT_TO_RIGHT" if x2 > x1 else "RIGHT_TO_LEFT"

# -------------------------------
# Main loop
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls]
            violation_text = ""

            # Focus only on relevant objects
            if label not in ["person", "motorcycle", "car", "truck"]:
                continue

            # Helmet violation (placeholder logic)
            if label == "motorcycle":
                violation_text = "Helmet Check Needed"

            # Signal jump detection
            if signal_color == "RED" and label in ["car", "motorcycle"]:
                if y2 < height // 2:
                    violation_text = "🚨 SIGNAL JUMP"

            # Wrong-way detection
            direction = detect_direction(x1, x2)
            if direction != REFERENCE_DIRECTION:
                violation_text = "🚫 WRONG WAY"

            # Draw bounding box
            color = (0, 0, 255) if violation_text else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {violation_text}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Display signal state
    cv2.putText(
        frame,
        f"Signal: {signal_color}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    out.write(frame)
    cv2.imshow("Traffic Violation Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
