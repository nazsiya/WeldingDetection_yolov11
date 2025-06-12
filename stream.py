import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# Title
st.title("🧠 Crack Detection using YOLOv8")
st.write("Upload an image to detect cracks using a trained YOLOv8 model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
 # Display uploaded image
 image = Image.open(uploaded_file)
 st.image(image, caption="Uploaded Image", use_column_width=True)

 # Load the YOLO model
 model_path = "best.pt" # Replace with actual path if needed
 model = YOLO(model_path)

 # Save uploaded image temporarily
 with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
    image.save(temp_file.name)
    temp_image_path = temp_file.name

 # Run detection
 st.subheader("Detection in progress...")
 results = model.predict(temp_image_path, conf=0.5)

 # Display image with detections
 for result in results:
    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1]) # Convert to PIL image
    st.image(im, caption="Detection Result", use_column_width=True)

    # Detection metrics
    st.subheader("Detection Metrics")
    st.write(f"✅ Number of defects found: **{len(result.boxes)}**")
    st.write("🔍 Class-wise detections:")

 for box in result.boxes:
    class_name = model.names[int(box.cls)]
    confidence = box.conf.item()
    st.write(f"- **{class_name}**: Confidence `{confidence:.2f}`")

 # Clean up
 os.remove(temp_image_path)