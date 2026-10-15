
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


# ---------------- MODERN SAFEVISION THEME ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b192b 55%, #10243b 100%);
    color: #edf5ff;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.85);
}

[data-testid="stSidebar"] {
    background: #091525;
    border-right: 1px solid #1c3852;
}

h1, h2, h3 {
    color: #e8f5ff !important;
    font-weight: 750 !important;
}

p, label, .stMarkdown {
    color: #d1dfef;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #10243b, #142d47);
    border: 1px solid #244966;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetricLabel"] {
    color: #a8c3dc !important;
}

[data-testid="stMetricValue"] {
    color: #4dd8f5 !important;
    font-weight: 800;
}

.stButton > button,
[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #087ea4, #2563eb);
    color: white;
    border: 1px solid #3186b8;
    border-radius: 10px;
    min-height: 42px;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: #67e8f9;
    color: white;
    transform: translateY(-2px);
}

[data-testid="stDataFrame"] {
    border: 1px solid #244966;
    border-radius: 12px;
}

hr {
    border-color: #24415a;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #10243b;
    border-color: #31516c;
}

</style>
""", unsafe_allow_html=True)



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


# ---------------- PERSISTENT SCAN HISTORY ----------------

def persist_history(records, source):
    """Append completed scan results to a local CSV history file."""
    if not records:
        return

    history_path = REPORT_DIR / "detection_history.csv"

    history_df = pd.DataFrame(records).copy()
    history_df.insert(0, "Source", source)
    history_df.insert(
        1, "Scan ID", time.strftime("%Y%m%d-%H%M%S")
    )

    history_df.to_csv(
        history_path,
        mode="a",
        header=not history_path.exists(),
        index=False,
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



# ---------------- INTERACTIVE NAVIGATION ----------------

page = st.radio(
    "Navigate your workspace",
    [
        "Dashboard",
        "Image Analysis",
        "Video Analysis",
        "Live Webcam",
        "CCTV Stream",
        "Safety Reports",
    ],
    horizontal=True,
    key="navigation_page",
)

st.divider()


# ---------------- INTERACTIVE DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("📊 Safety Operations Overview")
    st.write(
        "Monitor PPE detections and review your latest safety analysis."
    )

    records = st.session_state.get("last_records", [])
    model_ready = CUSTOM_MODEL.exists()

    # Model status and scan summary
    col1, col2 = st.columns([2, 1])

    with col1:
        if model_ready:
            st.success("🟢 Custom PPE detection model is ready")
        else:
            st.warning("🟡 Demo model active — PPE results may be unreliable")

    with col2:
        st.info("Detection confidence: " + f"{confidence:.0%}")

    st.divider()

    if records:
        df = pd.DataFrame(records)

        total = len(df)
        violations = int(
            (df["Potential Violation"] == "Yes").sum()
        )
        people = int(
            (df["Detected Class"].str.lower() == "person").sum()
        )
        classes = df["Detected Class"].nunique()

        # Interactive KPI row
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("🔎 Total detections", total)
        k2.metric("⚠️ Potential violations", violations)
        k3.metric("👷 People detected", people)
        k4.metric("🏷️ Object classes", classes)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("### 📈 Detection breakdown")
            class_counts = (
                df["Detected Class"]
                .value_counts()
                .rename_axis("Object class")
                .to_frame("Detections")
            )
            st.bar_chart(class_counts)

        with right:
            st.markdown("### 🦺 PPE detection summary")

            labels = df["Detected Class"].str.lower()

            helmet_count = int(
                labels.isin(["helmet"]).sum()
            )
            no_helmet_count = int(
                labels.isin(["no-helmet", "no_helmet", "no helmet"]).sum()
            )
            vest_count = int(
                labels.isin(["vest"]).sum()
            )
            no_vest_count = int(
                labels.isin(["no-vest", "no_vest", "no vest"]).sum()
            )

            st.write(f"🟢 Helmet detections: **{helmet_count}**")
            st.write(f"🔴 No-helmet detections: **{no_helmet_count}**")
            st.write(f"🟢 Safety vest detections: **{vest_count}**")
            st.write(f"🔴 No-vest detections: **{no_vest_count}**")

            st.caption(
                "Counts reflect detected objects, not unique workers. "
                "Review detections before making safety decisions."
            )

        st.divider()

        st.markdown("### 🕒 Recent detection activity")
        st.dataframe(
            df.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download latest dashboard report",
            data=csv_data,
            file_name="safety_dashboard_report.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Your dashboard is ready. Analyze an image, video, "
            "webcam feed, or CCTV stream to see real detection statistics."
        )

        st.markdown("### 🚀 Quick actions")
        a, b, c = st.columns(3)

        with a:
            if st.button("🖼️ Analyze an image", use_container_width=True):
                st.session_state["navigation_page"] = "Image Analysis"
                st.rerun()

        with b:
            if st.button("🎬 Analyze a video", use_container_width=True):
                st.session_state["navigation_page"] = "Video Analysis"
                st.rerun()

        with c:
            if st.button("📋 View reports", use_container_width=True):
                st.session_state["navigation_page"] = "Safety Reports"
                st.rerun()
# Persistent history from previously completed scans
    st.divider()
    st.subheader("📚 Persistent Scan History")

    history_path = REPORT_DIR / "detection_history.csv"

    if history_path.exists():
        history_df = pd.read_csv(history_path)

        if not history_df.empty:
            f1, f2, f3 = st.columns(3)

            with f1:
                source_options = [
                    "All",
                    *sorted(history_df["Source"].dropna().unique().tolist())
                ]
                selected_source = st.selectbox(
                    "Filter by source", source_options
                )

            with f2:
                class_options = [
                    "All",
                    *sorted(
                        history_df["Detected Class"]
                        .dropna().astype(str).unique().tolist()
                    )
                ]
                selected_class = st.selectbox(
                    "Filter by detected class", class_options
                )

            with f3:
                selected_violation = st.selectbox(
                    "Filter by violation",
                    ["All", "Yes", "No"]
                )

            filtered = history_df.copy()

            if selected_source != "All":
                filtered = filtered[
                    filtered["Source"] == selected_source
                ]

            if selected_class != "All":
                filtered = filtered[
                    filtered["Detected Class"].astype(str)
                    == selected_class
                ]

            if selected_violation != "All":
                filtered = filtered[
                    filtered["Potential Violation"].astype(str)
                    == selected_violation
                ]

            st.metric("Matching history records", len(filtered))

            st.dataframe(
                filtered.iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

            history_csv = filtered.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download filtered history",
                data=history_csv,
                file_name="filtered_safety_history.csv",
                mime="text/csv",
                key="download_filtered_history"
            )

            if st.button("🗑️ Clear saved history"):
                history_path.unlink(missing_ok=True)
                st.success("Saved scan history cleared.")
                st.rerun()
        else:
            st.info("No saved scan history yet.")
    else:
        st.info(
            "No saved scan history yet. Complete an image, video, "
            "webcam, or CCTV scan to start building your history."
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
            persist_history(records, "Image")
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
            persist_history(all_records, "Video")
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
        persist_history(webcam_records, "Webcam")


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
                persist_history(cctv_records, "CCTV")
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
