import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import threading
from playsound import playsound

# page configuration
st.set_page_config(
    page_title="Kita Kita: Obstacle Detection",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom styling
st.markdown("""
<style>
    .stApp {background-color: #F4E7E1;}
    h1,h2,h3,h4,h5,p,li {color: #1D1D1B !important;}
    .top-header {text-align:center;padding:20px 0;color:#521C0D;}
    .title {font-size:50px;font-weight:bold;margin-bottom:-10px;color:#521C0D;}
    .subtitle {font-size:22px;margin-top:-5px;color:#521C0D;}
    .section-title {font-size:28px;font-weight:bold;color:#521C0D;margin-top:20px;}
    .circle-img {width:120px;height:120px;object-fit:cover;border-radius:50%;margin-bottom:10px;border:3px solid #00C2FF;}
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="top-header">
    <div class="title">Kita Kita</div>
    <div class="subtitle">Obstacle Detection for Safer Mobility</div>
</div>
""", unsafe_allow_html=True)

# yolo model integration
model = YOLO(r"D:\Desktop\my_model\my_model.pt")
try:
    model.to("cuda")
except:
    pass

# tabs
overview_tab, detect_tab, realtime_tab, team_tab = st.tabs(["Overview", "Picture", "Realtime Scan", "Team"])

# overview
with overview_tab :
    st.markdown('<h2 class="section-title">Project Overview</h2>', unsafe_allow_html=True)
    st.write("Kita Kita is a real-time obstacle detection system designed to assist visually impaired individuals.")
    st.markdown('<h2 class="section-title">Problem Statement</h2>', unsafe_allow_html=True)
    st.write("Sidewalks often have hazards like poles, vendors, parked vehicles. Canes can't detect obstacles at varying heights or distances.")
    st.markdown('<h2 class="section-title">Target Users</h2>', unsafe_allow_html=True)
    st.write("- Visually impaired\n- Users navigating unfamiliar environments\n- People in low-visibility settings")
    st.markdown('<h2 class="section-title">Proposed Solution</h2>', unsafe_allow_html=True)
    st.write("Kita Kita detects obstacles in real-time and warns users to move safely.")
    st.markdown('<h2 class="section-title">Technologies Used</h2>', unsafe_allow_html=True)
    st.write("- YOLOv11\n- Roboflow dataset\n- OpenCV\n- Streamlit")

# picture
with detect_tab:
    st.markdown('<h2 class="section-title">Picture Obstacle Detection</h2>', unsafe_allow_html=True)
    st.write("Picture scan for obstacles found in streets and sidewalks.")
    upload = st.checkbox("Upload Image")
    camera = st.checkbox("Camera")
    if upload :
        uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            results = model(img_array, conf=0.25, stream=True)
            annotated_image = None
            for r in results:
                annotated_image = r.plot()
                for box in r.boxes:
                    conf = float(box.conf[0])
                    bbox_height = box.xyxy[0][3] - box.xyxy[0][1]
            col1, col2 = st.columns(2)
            with col1: st.image(image, width="content")
            with col2: st.image(annotated_image, width="content")

    if camera :
        camera_image = st.camera_input("Take a photo")
        if camera_image:
            image = Image.open(camera_image)
            img_array = np.array(image)
            results = model(img_array, conf=0.25, stream=True)
            annotated_image = None
            for r in results:
                annotated_image = r.plot()
                for box in r.boxes:
                    conf = float(box.conf[0])
                    bbox_height = box.xyxy[0][3] - box.xyxy[0][1]
            col1, col2 = st.columns(2)
            with col1: st.image(image, width="content")
            with col2: st.image(annotated_image, width="content")

# realtime
with realtime_tab:
    st.markdown('<h2 class="section-title">Realtime Camera Obstacle Detection</h2>', unsafe_allow_html=True)
    st.write("Live scan with sound warning for close obstacles.")
    run_scan = st.checkbox("Realtime Scan", key="local_realtime_scan")
    upload = st.checkbox("Upload a Video")
    stframe = st.empty()

    # realtime webcam
    if run_scan:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): 
            st.error("Cannot access webcam.")
        else:
            frame_id = 0
            while st.session_state["local_realtime_scan"]:
                ret, frame = cap.read()
                if not ret: 
                    break
                frame_id += 1
                if frame_id % 2 != 0: 
                    continue
                h, w, _ = frame.shape
                scale = 640 / max(h, w)
                new_w, new_h = int(w*scale), int(h*scale)
                resized = cv2.resize(frame, (new_w, new_h))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame, conf=0.25, stream=True)
                annotated_frame = None
                for r in results:
                    annotated_frame = r.plot()
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        bbox_height = box.xyxy[0][3] - box.xyxy[0][1]
                        if conf >= 0.7 and label != "sidewalk" :
                            playsound("Notification Sound Effect.wav")
                stframe.image(annotated_frame, channels="RGB", width="content")
            cap.release()

    # video upload
    if upload:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4","avi","mov","mkv"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            video_placeholder = st.empty()
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if frame_id % 2 != 0: 
                    continue
                h, w, _ = frame.shape
                scale = 640 / max(h, w)
                new_w, new_h = int(w*scale), int(h*scale)
                resized = cv2.resize(frame, (new_w, new_h))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame, conf=0.25, stream=True)
                annotated_frame = None
                for r in results:
                    annotated_frame = r.plot()
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        bbox_height = box.xyxy[0][3] - box.xyxy[0][1]
                        if conf >= 0.7 and label != "sidewalk":
                            playsound("Notification Sound Effect.wav")
                video_placeholder.image(annotated_frame, channels="RGB")
            cap.release()
            tfile.close()
            st.success("Video processing complete!")

# team
with team_tab:
    st.markdown('<h2 class="section-title">Meet the Team</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(r"D:\Desktop\my_model\kylian.jpg", width=120)
        st.write("**Kylian Buton**")
    with col2:
        st.image(r"D:\Desktop\my_model\bea dee.jpg", width=120)
        st.write("**Beatrice Dee**")
    with col3:
        st.image(r"D:\Desktop\my_model\jesse.jpg", width=120)
        st.write("**Jesus Verzosa**")
