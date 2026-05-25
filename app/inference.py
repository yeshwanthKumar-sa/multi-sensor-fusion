from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Allowed AV-related classes
ALLOWED_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person"
]


def run_inference(image):

    results = model(image)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            # Ignore irrelevant objects
            if label not in ALLOWED_CLASSES:
                continue

            detections.append({
                "label": label,
                "confidence": round(confidence * 100, 2)
            })

    return detections