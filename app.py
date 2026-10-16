
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

# ---------------- REAL-TIME SAFETY ALERTS ----------------

ALERT_FILE = REPORT_DIR / "safety_alerts.csv"

ALERT_LOG = []
ALERT_LOCK = Lock()

# Prevent repeated alerts for the same class and source.
ALERT_COOLDOWN_SECONDS = 10
LAST_ALERT_TIME = {}

ALERT_CLASSES = {
    "no-helmet",
    "no_helmet",
    "no helmet",
    "no-vest",
    "no_vest",
    "no vest",
    "no-hardhat",
    "no_hardhat",
    "no hard hat",
    "without_helmet",
    "without_vest",
}


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
            background:
                radial-gradient(ellipse at 50% 15%, rgba(198,255,77,.09), transparent 38%),
                linear-gradient(145deg, #090c0a, #111711 65%, #0b0e0b);
            color: #f1f5ed;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .login-title {
            color: #f1f5ed;
            text-align: center;
            font-size: 42px;
            font-weight: 850;
            letter-spacing: -1.5px;
        }
        .login-subtitle {
            color: #9aa79a;
            text-align: center;
            margin-bottom: 25px;
        }
        [data-testid="stForm"] {
            background: rgba(18, 22, 19, .96);
            padding: 30px;
            border: 1px solid #354236;
            border-radius: 20px;
            box-shadow: 0 24px 70px rgba(0,0,0,.3);
        }
        [data-testid="stForm"] h2,
        [data-testid="stForm"] h3,
        [data-testid="stForm"] p,
        [data-testid="stForm"] label {
            color: #f1f5ed !important;
        }
        .stButton > button,
        [data-testid="stFormSubmitButton"] button {
            background: #c6ff4d;
            color: #10150b;
            border: 1px solid #c6ff4d;
            border-radius: 11px;
            font-weight: 800;
            min-height: 46px;
            width: 100%;
        }
        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #d5ff80;
            color: #10150b;
            border-color: #d5ff80;
        }
        div[data-baseweb="input"] > div {
            background: #171d18;
            border-color: #354236;
            border-radius: 10px;
        }
        input {
            color: #f1f5ed !important;
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



# ---------------- DARK COMMAND CENTER THEME ----------------

st.markdown("""
<style>
/* ---------- DESIGN TOKENS ---------- */
:root {
    --bg: #0b0d0c;
    --surface: #121613;
    --surface-2: #181e19;
    --surface-3: #202820;
    --border: #2b352c;
    --lime: #c6ff4d;
    --lime-soft: #a9e83e;
    --text: #f1f5ed;
    --muted: #9aa79a;
    --danger: #ff646b;
    --warning: #ffc857;
    --blue: #75baff;
    --radius: 18px;
}

/* ---------- APPLICATION BACKGROUND ---------- */
.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(198,255,77,.055), transparent 32%),
        linear-gradient(145deg, #0b0d0c 0%, #0d110e 55%, #101510 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(11, 13, 12, .88);
    border-bottom: 1px solid rgba(198,255,77,.08);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

/* ---------- TYPOGRAPHY ---------- */
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.035em;
    font-weight: 750 !important;
}

h1 {
    font-size: clamp(2rem, 3vw, 2.8rem) !important;
    line-height: 1.12 !important;
}

h2 {
    font-size: 1.65rem !important;
}

h3 {
    font-size: 1.18rem !important;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #d0d8cf;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #0e120f;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--lime) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

/* ---------- TOP NAVIGATION ---------- */
div[role="radiogroup"] {
    gap: 8px;
    padding: 9px;
    background: rgba(18, 22, 19, .92);
    border: 1px solid var(--border);
    border-radius: 15px;
}

div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 8px 13px;
    transition: all .18s ease;
}

div[role="radiogroup"] label:hover {
    background: var(--surface-3);
    border-color: #3b493b;
}

div[role="radiogroup"] label p {
    color: #c4cec2 !important;
    font-size: .88rem;
    font-weight: 650;
}

div[role="radiogroup"] label:has(input:checked) {
    background: rgba(198,255,77,.12);
    border-color: rgba(198,255,77,.45);
}

div[role="radiogroup"] label:has(input:checked) p {
    color: var(--lime) !important;
}

/* ---------- METRIC / KPI CARDS ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #171d18, #111511);
    border: 1px solid #2c382d;
    border-radius: var(--radius);
    padding: 20px 21px;
    min-height: 125px;
    box-shadow: 0 10px 28px rgba(0,0,0,.14);
    transition: transform .18s ease, border-color .18s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(198,255,77,.55);
}

[data-testid="stMetricLabel"] {
    color: #a9b5a8 !important;
    font-size: .84rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: var(--lime) !important;
    font-size: clamp(1.6rem, 2.3vw, 2.25rem) !important;
    font-weight: 800 !important;
    letter-spacing: -.04em;
}

[data-testid="stMetricDelta"] {
    font-size: .8rem !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    color: #10150b;
    background: linear-gradient(135deg, var(--lime), #a8e832);
    border: 1px solid #c6ff4d;
    border-radius: 11px;
    min-height: 43px;
    padding: 8px 17px;
    font-weight: 750;
    transition: all .18s ease;
    box-shadow: 0 4px 15px rgba(198,255,77,.08);
}

.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #d5ff80;
    color: #0b0d0c;
    border-color: #e1ffad;
    transform: translateY(-1px);
}

.stButton > button:focus,
[data-testid="stDownloadButton"] button:focus {
    box-shadow: 0 0 0 3px rgba(198,255,77,.22);
}

/* Secondary buttons, where supported by Streamlit */
.stButton > button[kind="secondary"] {
    background: var(--surface-2);
    color: var(--text);
    border-color: var(--border);
}

/* ---------- INPUTS AND SELECTORS ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #171d18;
    border-color: #354236;
    border-radius: 10px;
    color: var(--text);
}

input, textarea {
    color: var(--text) !important;
}

[data-testid="stFileUploader"] {
    background: #141914;
    border: 1px dashed #465543;
    border-radius: 15px;
    padding: 10px;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--lime);
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--lime);
    border-color: var(--lime);
}

/* ---------- CONTENT PANELS / CONTAINERS ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(18, 22, 19, .78);
    border-color: var(--border);
    border-radius: var(--radius);
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 13px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- TABLES ---------- */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 13px;
    overflow: hidden;
}

[data-testid="stDataFrame"] * {
    border-color: #283129;
}

/* ---------- CHARTS ---------- */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPlotlyChart"] {
    background: #121713;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 10px;
}

/* ---------- IMAGES / VIDEO ---------- */
[data-testid="stImage"],
[data-testid="stVideo"] {
    border-radius: 15px;
    overflow: hidden;
}

img {
    border-radius: 10px;
}

/* ---------- DIVIDERS / PROGRESS ---------- */
hr {
    border-color: #2a332b !important;
}

[data-testid="stProgressBar"] > div > div {
    background: var(--lime);
}

/* ---------- STATUS / LINKS ---------- */
a {
    color: var(--lime) !important;
}

small, .muted {
    color: var(--muted);
}

/* ---------- MOBILE ---------- */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem .8rem 2rem;
    }

    div[role="radiogroup"] {
        flex-wrap: wrap;
    }

    [data-testid="stMetric"] {
        padding: 15px;
        min-height: 105px;
    }
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

def register_alerts(records, source):
    """Record potential PPE violations with a cooldown."""

    new_alerts = []
    now = time.time()

    with ALERT_LOCK:
        for record in records:
            detected_class = str(
                record.get("Detected Class", "")
            ).strip().lower()

            is_violation = (
                record.get("Potential Violation") == "Yes"
                or detected_class in ALERT_CLASSES
            )

            if not is_violation:
                continue

            alert_key = (source, detected_class)
            previous_time = LAST_ALERT_TIME.get(alert_key, 0)

            if now - previous_time < ALERT_COOLDOWN_SECONDS:
                continue

            LAST_ALERT_TIME[alert_key] = now

            alert = {
                "Timestamp": record.get(
                    "Timestamp",
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ),
                "Source": source,
                "Detected Class": record.get(
                    "Detected Class", "unknown"
                ),
                "Confidence": record.get("Confidence", 0),
                "Alert": "Potential PPE violation detected",
            }

            ALERT_LOG.append(alert)
            new_alerts.append(alert)

        # Keep the in-memory alert list bounded.
        if len(ALERT_LOG) > 5000:
            del ALERT_LOG[:1000]

        # Append new alerts to persistent CSV history.
        if new_alerts:
            alert_df = pd.DataFrame(new_alerts)
            write_header = not ALERT_FILE.exists()

            alert_df.to_csv(
                ALERT_FILE,
                mode="a",
                header=write_header,
                index=False
            )

    return new_alerts

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


# ---------------- COMMAND CENTER HEADER ----------------

st.markdown("""
<div style="
    display:flex;
    align-items:center;
    gap:16px;
    padding:8px 0 22px 0;
">
    <div style="
        width:58px;
        height:58px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:16px;
        background:rgba(198,255,77,.10);
        border:1px solid rgba(198,255,77,.35);
        font-size:30px;
    ">🦺</div>
    <div>
        <div style="
            color:#C6FF4D;
            font-size:11px;
            font-weight:800;
            letter-spacing:3px;
            text-transform:uppercase;
            margin-bottom:5px;
        ">SAFETY INTELLIGENCE PLATFORM</div>
        <div style="
            color:#F1F5ED;
            font-size:clamp(25px,3vw,36px);
            font-weight:850;
            letter-spacing:-1.3px;
            line-height:1.12;
        ">SafeVision <span style="color:#C6FF4D;">AI</span></div>
        <div style="
            color:#9AA79A;
            font-size:13px;
            margin-top:7px;
        ">Real-time PPE detection · Visual analytics · Incident monitoring</div>
    </div>
</div>
""", unsafe_allow_html=True)

if CUSTOM_MODEL.exists():
    st.success("AI ENGINE ONLINE  ·  Custom PPE model loaded")
else:
    st.warning(
        "DEMO MODEL ACTIVE  ·  General YOLOv8 weights are not a "
        "PPE compliance model. Add trained weights at models/best.pt."
    )



# ---------------- COMMAND CENTER NAVIGATION ----------------

page = st.radio(
    "WORKSPACE",
    [
        "Dashboard",
        "Image Analysis",
        "Video Analysis",
        "Live Webcam",
        "CCTV Stream",
        "Safety Alerts",
        "Safety Reports",
    ],
    horizontal=True,
    key="navigation_page",
    label_visibility="collapsed",
)

st.divider()





# ---------------- DARK COMMAND CENTER DASHBOARD ----------------

if page == "Dashboard":

    st.markdown("""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        flex-wrap:wrap;
        gap:12px;
        margin:5px 0 22px;
    ">
      <div>
        <div style="
            color:#C6FF4D;
            font-size:11px;
            font-weight:800;
            letter-spacing:2.5px;
        ">OPERATIONS OVERVIEW</div>
        <div style="
            color:#F1F5ED;
            font-size:30px;
            font-weight:800;
            letter-spacing:-1px;
            margin-top:5px;
        ">Safety Command Center</div>
        <div style="color:#9AA79A;font-size:13px;margin-top:5px;">
            Detection intelligence, PPE risk signals and monitoring history
        </div>
      </div>
      <div style="
          border:1px solid #354236;
          background:#141914;
          color:#C6FF4D;
          padding:9px 13px;
          border-radius:10px;
          font-size:12px;
          font-weight:700;
      ">● ANALYTICS WORKSPACE</div>
    </div>
    """, unsafe_allow_html=True)

    history_path = REPORT_DIR / "detection_history.csv"
    history_df = pd.DataFrame()

    if history_path.exists():
        try:
            history_df = pd.read_csv(history_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            history_df = pd.DataFrame()

    if history_df.empty:
        current_records = st.session_state.get("last_records", [])
        if current_records:
            history_df = pd.DataFrame(current_records)

    if history_df.empty:
        st.markdown("""
        <div style="
            padding:28px;
            background:#141914;
            border:1px solid #2b352c;
            border-radius:18px;
        ">
            <div style="font-size:32px;">◉</div>
            <div style="font-size:21px;font-weight:750;color:#F1F5ED;
                        margin-top:10px;">Waiting for detection data</div>
            <div style="color:#9AA79A;margin-top:6px;">
                Run an image, video, webcam or CCTV analysis to populate
                your operational dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Start a monitoring workflow")
        q1, q2, q3 = st.columns(3)

        with q1:
            if st.button("▧  Analyze image", use_container_width=True):
                st.session_state["navigation_page"] = "Image Analysis"
                st.rerun()

        with q2:
            if st.button("▶  Analyze video", use_container_width=True):
                st.session_state["navigation_page"] = "Video Analysis"
                st.rerun()

        with q3:
            if st.button("▤  View safety reports", use_container_width=True):
                st.session_state["navigation_page"] = "Safety Reports"
                st.rerun()

    else:
        df = history_df.copy()

        if "Source" not in df.columns:
            df["Source"] = "Latest scan"
        if "Timestamp" not in df.columns:
            df["Timestamp"] = pd.NaT
        if "Detected Class" not in df.columns:
            df["Detected Class"] = "Unknown"
        if "Potential Violation" not in df.columns:
            df["Potential Violation"] = "No"

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], errors="coerce"
        )
        df["Source"] = df["Source"].fillna("Unknown").astype(str)
        df["Detected Class"] = (
            df["Detected Class"].fillna("Unknown").astype(str)
        )
        df["Potential Violation"] = (
            df["Potential Violation"].fillna("No").astype(str)
        )

        # ---------------- FILTER BAR ----------------

        with st.container(border=True):
            st.markdown("#### ◈ Analytics filters")
            f1, f2, f3 = st.columns([1, 1, 1.4])

            with f1:
                source_options = sorted(df["Source"].unique().tolist())
                source_choice = st.selectbox(
                    "Monitoring source",
                    ["All sources"] + source_options,
                    key="analytics_source",
                )

            with f2:
                class_options = sorted(
                    df["Detected Class"].unique().tolist()
                )
                class_choice = st.selectbox(
                    "Detected class",
                    ["All classes"] + class_options,
                    key="analytics_class",
                )

            with f3:
                valid_dates = df["Timestamp"].dropna()
                date_range = None

                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    date_range = st.date_input(
                        "Date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="analytics_date_range",
                    )
                else:
                    st.caption("No valid timestamps available for date filtering.")

        filtered = df.copy()

        if source_choice != "All sources":
            filtered = filtered[filtered["Source"] == source_choice]

        if class_choice != "All classes":
            filtered = filtered[
                filtered["Detected Class"] == class_choice
            ]

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                filtered["Timestamp"].notna()
                & (filtered["Timestamp"].dt.date >= start_date)
                & (filtered["Timestamp"].dt.date <= end_date)
            ]

        normalized_violation = (
            filtered["Potential Violation"]
            .str.strip().str.lower().eq("yes")
        )
        normalized_class = (
            filtered["Detected Class"].str.strip().str.lower()
        )

        total = len(filtered)
        violations = int(normalized_violation.sum())
        people = int(normalized_class.eq("person").sum())
        violation_rate = (violations / total * 100) if total else 0
        distinct_classes = filtered["Detected Class"].nunique()

        st.markdown("### Operational snapshot")
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("◉ Total detections", f"{total:,}")
        k2.metric("⚠ Potential violations", f"{violations:,}")
        k3.metric("♙ Person detections", f"{people:,}")
        k4.metric("↗ Violation rate", f"{violation_rate:.1f}%")

        st.caption(
            f"{distinct_classes} detected classes in the filtered records. "
            "Violation rate is the percentage of object detections marked "
            "as potential violations, not a worker compliance rate."
        )

        st.markdown("### Detection intelligence")
        chart_left, chart_right = st.columns(2, gap="large")

        with chart_left:
            with st.container(border=True):
                st.markdown("#### Object class distribution")
                if not filtered.empty:
                    class_counts = (
                        filtered["Detected Class"]
                        .value_counts()
                        .rename_axis("Detected class")
                        .to_frame("Detections")
                    )
                    st.bar_chart(
                        class_counts,
                        color="#C6FF4D",
                        height=300,
                    )
                else:
                    st.info("No detections match the current filters.")

        with chart_right:
            with st.container(border=True):
                st.markdown("#### Violation classification")
                if not filtered.empty:
                    violation_counts = (
                        filtered["Potential Violation"]
                        .str.strip()
                        .str.title()
                        .value_counts()
                        .rename_axis("Classification")
                        .to_frame("Records")
                    )
                    st.bar_chart(
                        violation_counts,
                        color="#FF646B",
                        height=300,
                    )
                else:
                    st.info("No records match the current filters.")

        with st.container(border=True):
            st.markdown("#### Detection activity over time")
            trend_df = filtered.dropna(subset=["Timestamp"]).copy()

            if not trend_df.empty:
                trend_df["Date"] = trend_df["Timestamp"].dt.strftime(
                    "%Y-%m-%d"
                )
                trend_df["Is Violation"] = (
                    trend_df["Potential Violation"]
                    .str.strip().str.lower().eq("yes")
                )
                daily = trend_df.groupby("Date").agg(
                    Detections=("Detected Class", "size"),
                    Violations=("Is Violation", "sum"),
                )
                st.line_chart(
                    daily,
                    color=["#C6FF4D", "#FF646B"],
                    height=300,
                )
            else:
                st.info("No valid timestamps are available for a trend chart.")

        # ---------------- PPE STATUS SUMMARY ----------------

        st.markdown("### PPE signal summary")

        helmet_count = int(normalized_class.eq("helmet").sum())
        no_helmet_count = int(
            normalized_class.isin(
                ["no-helmet", "no_helmet", "no helmet",
                 "no-hardhat", "no_hardhat", "no hard hat"]
            ).sum()
        )
        vest_count = int(normalized_class.eq("vest").sum())
        no_vest_count = int(
            normalized_class.isin(
                ["no-vest", "no_vest", "no vest"]
            ).sum()
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            with st.container(border=True):
                st.caption("HELMET DETECTIONS")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;"
                    f"color:#C6FF4D'>{helmet_count:,}</div>",
                    unsafe_allow_html=True,
                )

        with p2:
            with st.container(border=True):
                st.caption("MISSING HELMET SIGNALS")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;"
                    f"color:#FF646B'>{no_helmet_count:,}</div>",
                    unsafe_allow_html=True,
                )

        with p3:
            with st.container(border=True):
                st.caption("VEST DETECTIONS")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;"
                    f"color:#C6FF4D'>{vest_count:,}</div>",
                    unsafe_allow_html=True,
                )

        with p4:
            with st.container(border=True):
                st.caption("MISSING VEST SIGNALS")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:800;"
                    f"color:#FF646B'>{no_vest_count:,}</div>",
                    unsafe_allow_html=True,
                )

        st.caption(
            "Counts represent detected objects, not unique workers. "
            "Potential violations should be reviewed by a qualified safety "
            "professional before operational action."
        )

        # ---------------- HISTORY AND EXPORT ----------------

        with st.container(border=True):
            st.markdown("#### Recent detection records")
            display_df = filtered.sort_values(
                "Timestamp", ascending=False, na_position="last"
            )
            st.dataframe(
                display_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "↓ Export filtered analytics CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="safevision_analytics.csv",
                mime="text/csv",
                key="download_advanced_analytics",
            )



# ---------------- IMAGE ANALYSIS ----------------

if page == "Image Analysis":
    st.subheader("Image Safety Analysis")
    st.caption("Upload a workplace image to detect potential PPE violations.")

    uploaded = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            image.load()

            # Prevent accidental processing of extremely large images.
            max_pixels = 20_000_000

            if image.width * image.height > max_pixels:
                st.error(
                    "This image is too large to process. "
                    "Please upload an image with fewer than 20 million pixels."
                )
            else:
                st.image(
                    image,
                    caption="Original image",
                    width="stretch",
                )

                if st.button("Analyze Image", type="primary"):
                    try:
                        with st.spinner("Loading AI model and analyzing image..."):
                            model = get_model()

                            frame = cv2.cvtColor(
                                np.array(image),
                                cv2.COLOR_RGB2BGR,
                            )

                            annotated, records = detect_frame(
                                frame,
                                model,
                            )

                        new_alerts = register_alerts(
                            records,
                            "Image Analysis",
                        )

                        for alert in new_alerts:
                            st.warning(
                                f"Potential violation: "
                                f"{alert['Detected Class']} "
                                f"({float(alert['Confidence']):.1%} confidence)"
                            )

                        st.subheader("Detection Results")

                        st.image(
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB,
                            ),
                            caption="AI detection output",
                            width="stretch",
                        )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Objects detected",
                            len(records),
                        )

                        col2.metric(
                            "Potential violations",
                            sum(
                                record["Potential Violation"] == "Yes"
                                for record in records
                            ),
                        )

                        st.session_state["last_records"] = records

                        persist_history(records, "Image")
                        save_report(records, "image_safety_report.csv")

                        st.success("Image analysis completed successfully.")

                    except Exception as exc:
                        st.error(
                            f"Image analysis failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        st.info(
                            "Check the Streamlit Cloud logs for the full "
                            "traceback. Do not share passwords or secrets."
                        )

        except Exception as exc:
            st.error(
                f"Could not read this image: "
                f"{type(exc).__name__}: {exc}"
            )
            st.info("Try uploading a valid JPG or PNG image.")


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
                    frame, model
                )
                register_alerts(records, "Video Analysis")
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
            img, model
        )
        register_alerts(records, "Live Webcam")

        with LIVE_LOCK:
            LIVE_LOG.extend(records)
            if len(LIVE_LOG) > 5000:
                del LIVE_LOG[:1000]

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )

    webcam_ctx = webrtc_streamer(
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

    if webcam_ctx.state.playing:
        st.success("🟢 Webcam monitoring is active")
    else:
        st.info("⚪ Webcam monitoring is stopped")

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
                            frame, model
                        )
                        register_alerts(records, "CCTV Stream")
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

# ---------------- SAFETY ALERTS ----------------

elif page == "Safety Alerts":
    st.subheader("🚨 Real-Time Safety Alerts")
    st.caption(
        "Potential PPE violations detected by the AI model. "
        "Review alerts before taking action."
    )

    # Load alerts saved across app reruns.
    if ALERT_FILE.exists():
        try:
            alerts_df = pd.read_csv(ALERT_FILE)
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError,
            UnicodeDecodeError
        ):
            st.warning(
                "The alert history file could not be read. "
                "Check reports/safety_alerts.csv."
            )
            alerts_df = pd.DataFrame()
    else:
        alerts_df = pd.DataFrame()

    if not alerts_df.empty:
        alerts_df["Timestamp"] = pd.to_datetime(
            alerts_df["Timestamp"], errors="coerce"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total alerts", len(alerts_df))

        col2.metric(
            "Helmet-related alerts",
            alerts_df["Detected Class"]
            .astype(str)
            .str.lower()
            .str.contains("helmet|hardhat", regex=True)
            .sum()
        )

        col3.metric(
            "Vest-related alerts",
            alerts_df["Detected Class"]
            .astype(str)
            .str.lower()
            .str.contains("vest", regex=True)
            .sum()
        )

        st.divider()
        st.markdown("### Filter alerts")

        sources = sorted(
            alerts_df["Source"].dropna().unique().tolist()
        )
        selected_source = st.selectbox(
            "Monitoring source",
            ["All sources"] + sources
        )

        classes = sorted(
            alerts_df["Detected Class"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        selected_class = st.selectbox(
            "Detected class",
            ["All classes"] + classes
        )

        filtered_alerts = alerts_df.copy()

        if selected_source != "All sources":
            filtered_alerts = filtered_alerts[
                filtered_alerts["Source"] == selected_source
            ]

        if selected_class != "All classes":
            filtered_alerts = filtered_alerts[
                filtered_alerts["Detected Class"].astype(str)
                == selected_class
            ]

        filtered_alerts = filtered_alerts.sort_values(
            "Timestamp", ascending=False, na_position="last"
        )

        st.markdown("### Alert history")

        if not filtered_alerts.empty:
            st.dataframe(
                filtered_alerts,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download alert history",
                data=filtered_alerts.to_csv(index=False).encode("utf-8"),
                file_name="safety_alert_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No alerts match the selected filters.")

    else:
        st.info(
            "No alerts have been recorded yet. Run Image Analysis, "
            "Video Analysis, Live Webcam, or CCTV Stream first."
        )

    st.divider()
    st.markdown("### Monitoring guidance")
    st.write(
        "- **Live Webcam:** Alerts are recorded while the camera callback "
        "processes frames."
    )
    st.write(
        "- **CCTV Stream:** Alerts are recorded during the capture session."
    )
    st.write(
        "- **Image/Video Analysis:** Alerts are recorded when the "
        "uploaded media is analyzed."
    )
    st.caption(
        "Alert records are stored in reports/safety_alerts.csv. "
        "They may be lost if the workspace or its files are deleted."
    )
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
