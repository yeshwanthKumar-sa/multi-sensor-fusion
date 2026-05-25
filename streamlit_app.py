import streamlit as st
from PIL import Image
import time

from app.inference import run_inference
from app.visualization import draw_boxes

st.set_page_config(
    page_title="Multi-Sensor Fusion",
    layout="wide"
)

st.title("Multi-Sensor Fusion Demo")
st.write("Autonomous Vehicle Object Detection System")

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

            st.warning("No autonomous vehicle objects detected")

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

        st.markdown("---")

        st.subheader("System Info")

        st.write("Model: YOLOv8 Nano")
        st.write("Detection Mode: Autonomous Vehicle Filtering")
        st.write("Framework: Ultralytics YOLO")
        st.write("Status: Active")