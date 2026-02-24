import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLO model (YOLOv8 nano - fast)
model = YOLO("yolov8n.pt")

st.title("🔍 YOLO Object Detection & Counting App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np)

    # Get annotated image
    annotated_frame = results[0].plot()

    # Get detected class IDs
    boxes = results[0].boxes
    class_ids = boxes.cls.tolist() if boxes is not None else []

    # Count objects
    object_count = len(class_ids)

    # Display results
    st.subheader("📸 Detection Result")
    st.image(annotated_frame, use_container_width=True)

    st.subheader("📊 Object Count")
    st.success(f"Total Objects Detected: {object_count}")

    # Detailed class-wise count
    if object_count > 0:
        names = results[0].names
        class_count = {}

        for cls_id in class_ids:
            class_name = names[int(cls_id)]
            class_count[class_name] = class_count.get(class_name, 0) + 1

        st.subheader("📋 Class-wise Count")
        for cls_name, count in class_count.items():
            st.write(f"{cls_name}: {count}")
