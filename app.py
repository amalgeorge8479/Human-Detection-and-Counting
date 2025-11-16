import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
from sort import Sort
import numpy as np

st.set_page_config(page_title="Human Detection and Counting", layout="wide")

st.markdown("""
    <style>
        body, .stApp {
            background-color: white !important;
            color: black !important;
        }
        .stButton>button {
            color: white !important;
            background-color: black !important;
            border-radius: 8px;
        }
        h1, h2, h3, p, label {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("👤 Human Detection and Counting (Live Detection)")
st.write("Upload a video below to see real-time human detection and counting directly on this page.")

uploaded_file = st.file_uploader("📂 Upload your video file", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("▶️ Start Live Detection"):
        st.info("🔍 Running live detection... Please wait.")

        stframe = st.empty()

        model = YOLO('best.pt')
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        cap = cv2.VideoCapture(tfile.name)

        CONF_THRESHOLD = 0.5
        MIN_BOX_AREA = 400
        MIN_TRACK_FRAMES = 5
        SPATIAL_THRESH = 220
        TEMPORAL_THRESH = 120
        MAX_POS = 300

        counted_ids = set()
        counted_positions = []
        tracking_duration = {}
        frame_count = 0

        def is_near_recent(cx, cy, cur_frame):
            for px, py, pf in counted_positions:
                if cur_frame - pf > TEMPORAL_THRESH:
                    continue
                if np.hypot(cx - px, cy - py) < SPATIAL_THRESH:
                    return True
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
            frame_p = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            frame_p = cv2.GaussianBlur(frame_p, (3, 3), 0)

            results = model(frame_p, conf=CONF_THRESHOLD, verbose=False, imgsz=640)
            detections = results[0].boxes.data.cpu().numpy() if len(results) else np.empty((0, 6))

            dets_for_sort = []
            boxes_abs = []
            scores = []
            for x1, y1, x2, y2, conf, cls in detections:
                if int(cls) != 0:
                    continue
                w, h = (x2 - x1), (y2 - y1)
                if w * h < MIN_BOX_AREA:
                    continue
                if (h / max(w, 1)) < 0.9:
                    continue
                boxes_abs.append([int(x1), int(y1), int(w), int(h)])
                scores.append(float(conf))
                dets_for_sort.append([x1, y1, x2, y2, conf])

            if dets_for_sort:
                indices = cv2.dnn.NMSBoxes(boxes_abs, scores, CONF_THRESHOLD, 0.45)
                if len(indices) > 0:
                    idxs = np.array(indices).flatten()
                    dets_for_sort = [dets_for_sort[i] for i in idxs]
                dets_for_sort = np.array(dets_for_sort)
            else:
                dets_for_sort = np.empty((0, 5))

            tracked = tracker.update(dets_for_sort)

            for x1, y1, x2, y2, tid in tracked:
                tid = int(tid)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tracking_duration[tid] = tracking_duration.get(tid, 0) + 1

                if tracking_duration[tid] >= MIN_TRACK_FRAMES and tid not in counted_ids and not is_near_recent(cx, cy, frame_count):
                    counted_ids.add(tid)
                    counted_positions.append((cx, cy, frame_count))
                    if len(counted_positions) > MAX_POS:
                        counted_positions.pop(0)

                color = (0, 255, 0) if tid in counted_ids else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            cv2.putText(frame, f'Total: {len(counted_ids)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success(f"✅ Detection Complete! Total People Detected: {len(counted_ids)}")
