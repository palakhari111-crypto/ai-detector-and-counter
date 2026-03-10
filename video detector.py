import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(layout="wide")
st.title("🚀 YOLOv8 + DeepSORT Tracking Dashboard")

st.markdown("Upload a video to detect, track and count objects with unique IDs.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file.")
    else:
        st.success("Video loaded successfully!")

        # Initialize DeepSORT tracker
        tracker = DeepSort(max_age=30)

        # Layout
        col1, col2 = st.columns([3, 1])
        frame_placeholder = col1.empty()
        stats_placeholder = col2.empty()

        # Counters
        unique_ids = set()
        class_counts = defaultdict(set)

        fps = 0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            detections = []

            # -------------------- YOLO DETECTIONS --------------------
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Confidence filter
                if conf < 0.4:
                    continue

                detections.append((
                    [x1, y1, x2 - x1, y2 - y1],
                    conf,
                    cls
                ))

            # -------------------- DEEPSORT TRACKING --------------------
            tracks = tracker.update_tracks(detections, frame=frame)

            current_objects = 0

            for track in tracks:
                if not track.is_confirmed():
                    continue

                current_objects += 1

                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                l, t, r, b = int(l), int(t), int(r), int(b)

                unique_ids.add(track_id)

                cls_id = track.det_class
                class_name = model.names.get(cls_id, "Unknown")

                class_counts[class_name].add(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"{class_name} | ID {track_id}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # -------------------- FPS CALCULATION --------------------
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # -------------------- DASHBOARD --------------------
            stats_html = f"""
            <h3>📊 Tracking Dashboard</h3>
            <p><b>Objects in Current Frame:</b> {current_objects}</p>
            <p><b>Total Unique Objects:</b> {len(unique_ids)}</p>
            """

            for cls_name, ids in class_counts.items():
                stats_html += f"<p><b>{cls_name}:</b> {len(ids)}</p>"

            stats_placeholder.markdown(stats_html, unsafe_allow_html=True)

        cap.release()

        time.sleep(1)
        try:
            os.remove(video_path)
        except:
            pass

        st.success("Processing complete!")
