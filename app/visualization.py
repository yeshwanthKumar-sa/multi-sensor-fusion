from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.5

ALLOWED_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person"
]


def draw_boxes(image):

    results = model(
        image,
        conf=CONFIDENCE_THRESHOLD
    )

    # Filter unwanted classes
    for result in results:

        keep_boxes = []

        for box in result.boxes:

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            if label in ALLOWED_CLASSES:
                keep_boxes.append(box)

        result.boxes = keep_boxes

    plotted_image = results[0].plot()

    return Image.fromarray(plotted_image)