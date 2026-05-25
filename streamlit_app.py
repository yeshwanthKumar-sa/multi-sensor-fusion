import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av
import time

from app.inference import run_inference
from app.visualization import draw_boxes

# Load YOLO model
model = YOLO("yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.5

st.set_page_config(
    page_title="Multi-Sensor Fusion",
    layout="wide"
)

st.title("Multi-Sensor Fusion Demo")
st.write("Autonomous Vehicle Perception System")

# -----------------------------------
# IMAGE UPLOAD SECTION
# -----------------------------------

st.header("Image Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    start_time = time.time()

    col1, col2 = st.columns([1.4, 1])

    with col1:

        visualized_image = draw_boxes(image)

        st.image(
            visualized_image,
            caption="Detection Output",
            width="stretch"
        )

    with col2:

        with st.spinner("Running YOLO inference..."):
            results = run_inference(image)

        inference_time = round(
            (time.time() - start_time) * 1000,
            2
        )

        st.success("Inference Complete!")

        st.subheader("Detection Results")

        if len(results) == 0:

            st.warning("No AV objects detected")

        else:

            detected_classes = []
            confidences = []

            for detection in results:

                detected_classes.append(
                    detection["label"]
                )

                confidences.append(
                    detection["confidence"]
                )

                st.success(
                    f"{detection['label']} detected "
                    f"({detection['confidence']}%)"
                )

            avg_confidence = round(
                sum(confidences) / len(confidences),
                2
            )

            st.markdown("---")

            st.subheader("Detection Analytics")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Objects Detected",
                    len(results)
                )

            with metric2:
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence}%"
                )

            st.metric(
                "Inference Time",
                f"{inference_time} ms"
            )

            st.markdown("### Detected Classes")

            unique_classes = list(
                set(detected_classes)
            )

            for cls in unique_classes:
                st.write(f"• {cls}")

# -----------------------------------
# WEBCAM SECTION
# -----------------------------------

st.markdown("---")

st.header("Live Webcam Detection")


def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    results = model(
        img,
        conf=CONFIDENCE_THRESHOLD
    )

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


webrtc_streamer(
    key="av-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

# -----------------------------------
# SYSTEM INFO
# -----------------------------------

st.markdown("---")

st.subheader("System Info")

st.write("Model: YOLOv8 Nano")
st.write("Detection Mode: AV Object Filtering")
st.write("Framework: Ultralytics YOLO")
st.write("Features: Image + Webcam Detection")
st.write("Status: Active")