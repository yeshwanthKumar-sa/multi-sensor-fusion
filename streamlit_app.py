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
st.write("Autonomous Vehicle Perception System")

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

        with st.spinner("Running sensor fusion inference..."):
            time.sleep(2)

        st.success("Inference Complete!")

        st.subheader("Detection Results")

        results = run_inference()

        for obj in results["objects"]:
            st.write(f"{obj} detected")

        st.write(f"Confidence Score: {results['confidence']}")

        st.markdown("---")

        st.subheader("System Info")

        st.write("Model: Multi-Sensor Fusion Model")
        st.write("Sensors: Camera, LiDAR, Radar")
        st.write("Status: Active")