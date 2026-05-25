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
st.write("Real-Time Object Detection using YOLOv8")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.4, 1])

    with col1:

        visualized_image = draw_boxes(image)

        st.image(
            visualized_image,
            caption="Detection Output",
            use_container_width=True
        )

    with col2:

        with st.spinner("Running YOLO inference..."):
            time.sleep(1)

        st.success("Inference Complete!")

        st.subheader("Detection Results")

        results = run_inference(image)

        if len(results) == 0:

            st.warning("No objects detected")

        else:

            for detection in results:

                st.success(
                    f"{detection['label']} detected "
                    f"({detection['confidence']}%)"
                )

        st.markdown("---")

        st.subheader("System Info")

        st.write("Model: YOLOv8 Nano")
        st.write("Inference: Real-Time Object Detection")
        st.write("Framework: Ultralytics YOLO")
        st.write("Status: Active")