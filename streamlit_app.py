import streamlit as st 
 
st.title("Multi-Sensor Fusion Demo") 
 
st.write("Autonomous Vehicle Perception System") 
 
uploaded_file = st.file_uploader("Upload Image") 
 
if uploaded_file: 
    st.success("Image uploaded successfully!") 
