
import os
import time
import tempfile
from pathlib import Path
from threading import Lock

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Safety Monitoring",
    page_icon="🦺",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

CUSTOM_MODEL = MODEL_DIR / "best.pt"
DEFAULT_MODEL = "yolov8n.pt"

PPE_TERMS = (
    "no_helmet", "no-helmet", "no helmet",
    "no_hardhat", "no-hardhat", "no hard hat",
    "no_vest", "no-vest", "no vest",
    "without_helmet", "without_vest"
)

LIVE_LOG = []
LIVE_LOCK = Lock()


# ================= LOGIN & SESSION AUTHENTICATION =================
import hmac
import streamlit as st

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login screen
if not st.session_state.authenticated:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #07111f, #10243b, #0b1728);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .login-title {
            color: #f8fafc;
            text-align: center;
            font-size: 38px;
            font-weight: 800;
        }
        .login-subtitle {
            color: #9fb4cc;
            text-align: center;
            margin-bottom: 25px;
        }
        [data-testid="stForm"] {
            background: rgba(20, 39, 62, 0.85);
            padding: 28px;
            border: 1px solid #294762;
            border-radius: 18px;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #087ea4, #2563eb);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            min-height: 45px;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 1.15, 1])

    with center:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="login-title">🛡️ SafeVision AI</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="login-subtitle">'
            'AI-Powered Workplace Safety Monitoring'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("login_form"):
            st.subheader("Welcome back")
            st.write("Sign in to access your monitoring dashboard.")

            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password"
            )
            submitted = st.form_submit_button(
                "Sign in →",
                use_container_width=True
            )

            if submitted:
                try:
                    valid_username = hmac.compare_digest(
                        username, str(st.secrets["auth"]["username"])
                    )
                    valid_password = hmac.compare_digest(
                        password, str(st.secrets["auth"]["password"])
                    )
                except (KeyError, FileNotFoundError):
                    st.error(
                        "Login credentials are not configured. "
                        "Set them in .streamlit/secrets.toml."
                    )
                    st.stop()

                if valid_username and valid_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.caption("Authorized personnel only")

    st.stop()

# Logout button appears after successful login
with st.sidebar:
    st.success("Signed in")
    if st.button("🚪 Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ---------------- MODEL ----------------

@st.cache_resource
def load_model(model_path, modified_time):
    return YOLO(model_path)


def get_model():
    if CUSTOM_MODEL.exists():
        return load_model(
            str(CUSTOM_MODEL),
            CUSTOM_MODEL.stat().st_mtime
        )
    return load_model(DEFAULT_MODEL, 0)


def detect_frame(frame, model, confidence=0.35):
    result = model.predict(
        source=frame,
        conf=confidence,
        imgsz=640,
        verbose=False
    )[0]

    annotated = result.plot()
    records = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = str(result.names[class_id])
        score = float(box.conf[0].item())

        records.append({
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Detected Class": label,
            "Confidence": round(score, 4),
            "Potential Violation": (
                "Yes" if any(
                    term in label.lower()
                    for term in PPE_TERMS
                ) else "No"
            )
        })

    return annotated, records


def save_report(records, filename="safety_report.csv"):
    if not records:
        st.info("No objects were detected for this report.")
        return

    df = pd.DataFrame(records)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV report",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )


# ---------------- HEADER ----------------

st.title("🦺 AI-Powered Visual Safety Monitoring")
st.caption(
    "Visual processing • PPE detection • Safety analytics"
)

if CUSTOM_MODEL.exists():
    st.success("Custom PPE model loaded from models/best.pt")
else:
    st.warning(
        "Demo mode: using general YOLOv8 weights. "
        "These are NOT a PPE compliance model. "
        "Add your trained PPE weights as models/best.pt "
        "to enable PPE-specific detection."
    )

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )
    st.caption("Higher confidence may reduce false detections.")

    st.divider()
    st.write("**Current model**")
    st.write(
        "Custom PPE model"
        if CUSTOM_MODEL.exists()
        else "General YOLOv8 demo model"
    )


# ---------------- NAVIGATION ----------------

page = st.radio(
    "Choose a module",
    [
        "Image Analysis",
        "Video Analysis",
        "Live Webcam",
        "CCTV Stream",
        "Safety Reports"
    ],
    horizontal=True
)


# ---------------- IMAGE ANALYSIS ----------------

if page == "Image Analysis":
    st.subheader("Upload an image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG file",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original image", width="stretch")

        if st.button("Analyze Image", type="primary"):
            model = get_model()
            frame = cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2BGR
            )

            annotated, records = detect_frame(
                frame, model, confidence
            )

            st.subheader("Detection Results")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="AI detection output",
                width="stretch"
            )

            col1, col2 = st.columns(2)
            col1.metric("Objects detected", len(records))
            col2.metric(
                "Potential violation detections",
                sum(
                    r["Potential Violation"] == "Yes"
                    for r in records
                )
            )

            st.session_state["last_records"] = records
            save_report(records, "image_safety_report.csv")


# ---------------- VIDEO ANALYSIS ----------------

elif page == "Video Analysis":
    st.subheader("Upload a video")
    video = st.file_uploader(
        "Choose an MP4, AVI or MOV video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if video and st.button("Analyze Video", type="primary"):
        model = get_model()

        suffix = Path(video.name).suffix or ".mp4"
        input_path = None
        output_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_in:
                temp_in.write(video.read())
                input_path = temp_in.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Could not open this video file.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 20.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_out:
                output_path = temp_out.name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            all_records = []
            frame_count = 0
            progress = st.progress(0)
            status = st.empty()

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = 2000

            while cap.isOpened() and frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, records = detect_frame(
                    frame, model, confidence
                )
                writer.write(annotated)
                all_records.extend(records)
                frame_count += 1

                if total > 0:
                    progress.progress(
                        min(frame_count / min(total, max_frames), 1.0)
                    )

                if frame_count % 25 == 0:
                    status.write(
                        f"Processed {frame_count} frames..."
                    )

            cap.release()
            writer.release()
            progress.progress(1.0)

            st.success(f"Processed {frame_count} frames.")

            if frame_count:
                st.video(output_path)

            st.session_state["last_records"] = all_records
            save_report(all_records, "video_safety_report.csv")

        finally:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)


# ---------------- LIVE WEBCAM ----------------

elif page == "Live Webcam":
    st.subheader("Real-time browser webcam")
    st.info(
        "Allow camera permission in your browser. "
        "The Codespaces app must be accessed over HTTPS."
    )

    model = get_model()

    def webcam_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, records = detect_frame(
            img, model, confidence
        )

        with LIVE_LOCK:
            LIVE_LOG.extend(records)
            if len(LIVE_LOG) > 5000:
                del LIVE_LOG[:1000]

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )

    webrtc_streamer(
        key="safety-webcam",
        video_frame_callback=webcam_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True
    )

    if st.button("Create webcam report"):
        with LIVE_LOCK:
            webcam_records = list(LIVE_LOG)
        save_report(webcam_records, "webcam_safety_report.csv")


# ---------------- CCTV STREAM ----------------

elif page == "CCTV Stream":
    st.subheader("Accessible CCTV / RTSP stream")
    st.warning(
        "Use only a camera you are authorized to access. "
        "Do not share camera passwords or stream URLs."
    )

    rtsp_url = st.text_input(
        "Stream URL",
        placeholder="rtsp://camera-address:554/stream"
    )

    duration = st.slider(
        "Capture duration (seconds)",
        min_value=5,
        max_value=60,
        value=10
    )

    if st.button("Start CCTV Analysis", type="primary"):
        if not rtsp_url.strip():
            st.error("Enter an accessible stream URL.")
        else:
            model = get_model()
            cap = cv2.VideoCapture(rtsp_url)
            display = st.empty()
            status = st.empty()
            cctv_records = []
            started = time.time()
            count = 0

            if not cap.isOpened():
                st.error(
                    "Unable to connect. Check the URL, "
                    "network access and camera permissions."
                )
            else:
                try:
                    while time.time() - started < duration:
                        ok, frame = cap.read()
                        if not ok:
                            status.warning(
                                "Stream stopped or no frame received."
                            )
                            break

                        annotated, records = detect_frame(
                            frame, model, confidence
                        )
                        cctv_records.extend(records)
                        count += 1

                        display.image(
                            cv2.cvtColor(
                                annotated, cv2.COLOR_BGR2RGB
                            ),
                            caption="Live CCTV detections",
                            width="stretch"
                        )
                        status.write(
                            f"Frames processed: {count}"
                        )
                finally:
                    cap.release()

                st.session_state["last_records"] = cctv_records
                save_report(cctv_records, "cctv_safety_report.csv")


# ---------------- REPORTS ----------------

elif page == "Safety Reports":
    st.subheader("Detection Summary")

    records = st.session_state.get("last_records", [])

    if records:
        df = pd.DataFrame(records)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total detections", len(df))
        col2.metric(
            "Potential violation detections",
            int((df["Potential Violation"] == "Yes").sum())
        )
        col3.metric(
            "Distinct classes",
            df["Detected Class"].nunique()
        )

        st.bar_chart(df["Detected Class"].value_counts())
        save_report(records, "safety_summary.csv")
    else:
        st.info(
            "Analyze an image, video, webcam feed or CCTV stream "
            "first to populate the report."
        )

st.divider()
st.caption(
    "Milestone 1 • Visual Processing & Safety Detection. "
    "AI outputs must be reviewed by a qualified safety professional."
)
