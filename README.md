# 🛡️ AI Safety Monitoring System

An AI-powered safety monitoring application designed to analyze images and videos using computer vision and identify potential safety violations or hazardous situations.

## 🚀 Live Demo

**Try the deployed application here:**

👉 https://ai-safety-monitoring.streamlit.app/

## 📌 Project Overview

The AI Safety Monitoring System uses artificial intelligence and computer vision to support automated safety monitoring. It processes uploaded images and videos and uses object detection models to help identify safety-related conditions.

The project aims to demonstrate how AI can assist in improving workplace safety, monitoring environments, and reducing the effort required for manual inspection.

## ✨ Features

- 🖼️ **Image Analysis:** Upload images for AI-powered safety analysis.
- 🎥 **Video Analysis:** Process video files to detect objects and potential safety concerns.
- 🤖 **AI-Based Detection:** Uses the YOLO object detection framework.
- 📊 **Safety Reports:** Supports generating or displaying analysis results and reports, depending on the enabled application features.
- 📹 **Live Monitoring:** Supports webcam or real-time video functionality where configured.
- 🌐 **Web Interface:** Access the application through a browser without installing it locally.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Streamlit** – Web application interface
- **Ultralytics YOLO** – Object detection
- **OpenCV** – Image and video processing
- **NumPy** – Numerical operations
- **Pandas** – Data handling
- **Pillow (PIL)** – Image processing
- **Streamlit WebRTC** – Real-time media functionality where configured

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/NehaSanjana24/AI-Safety-Monitoring.git
```

### 2. Navigate to the project directory

```bash
cd AI-Safety-Monitoring
```

### 3. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will usually open at:

http://localhost:8501

## 📂 Project Structure

```text
AI-Safety-Monitoring/
├── app.py
├── requirements.txt
├── models/
├── reports/
├── .streamlit/
└── README.md
```

*Note: Additional files and folders may be present in the repository.*

## 🎯 Applications

- Workplace and industrial safety monitoring
- Automated video and image inspection
- Computer vision experimentation
- AI-assisted hazard identification
- Safety monitoring prototypes and demonstrations

## 🔮 Future Enhancements

- Improve detection accuracy for safety-related events.
- Add more customizable safety rules and alerts.
- Enhance real-time monitoring and notifications.
- Improve report generation and analytics.
- Optimize processing speed and resource usage.
- Expand support for different environments and safety scenarios.

## ⚠️ Disclaimer

This project is intended for educational, experimental, and decision-support purposes. AI predictions may be inaccurate and should not replace professional safety inspections or established safety procedures. Detection capabilities depend on the model, input quality, and application configuration.

## 👩‍💻 Author

**Neha Sanjana**

GitHub: [NehaSanjana24](https://github.com/NehaSanjana24)

## 🔗 Project Links

- **Live Application:** https://ai-safety-monitoring.streamlit.app/
- **GitHub Repository:** https://github.com/NehaSanjana24/AI-Safety-Monitoring

---

⭐ If you find this project interesting, consider starring the repository on GitHub!
