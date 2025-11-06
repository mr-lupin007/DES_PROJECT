# 🎯 AI Presentation Coach Pro — Ultimate Edition v4.0
Advanced multimodal presentation analysis (speech + vision) with professional benchmarking, progress tracking, and AI-powered feedback.

**Team:** Surya • Pushpendra • Meet • Vasundhara • Ayush  
**Course:** DES646 – Design Engineering (2025)

---

## ✨ What it does
- **Live Practice:** Records webcam + mic locally, tracks eye contact via OpenCV (Haar cascades), detects basic emotions (smile proxy), transcribes speech, and computes WPM, fillers, vocabulary richness, etc.
- **Pro Benchmarks:** Compare against **TED**, **Business**, and **Academic** speaker profiles.
- **Analytics Dashboard:** Trend lines, progress, historical table, CSV export, and a one-click **full report**.
- **AI Feedback:** Rule-based report enhanced with a Gemini-style formatter (no external call required). You can plug in a Gemini API key for future integration.

> ℹ️ **Privacy by default:** Your transcripts, scores, and feedback are stored **locally** in `presentation_history.db` (SQLite). No cloud logging unless you deploy the app and choose to persist the DB.

---

## 🖼️ Screens & Flow
1. **🎤 Live Practice** → capture video/audio, compute metrics, show live eye-contact %, dominant emotion, and timers.
2. **📊 Analytics Dashboard** → trend of scores/WPM/eye-contact with recent improvements and session history.
3. **🧪 System Check** → camera & mic tests, dependency and DB health checks.
4. **⚙️ Settings** → optional Gemini API key, DB reset tools.
5. **ℹ️ About** → tech stack & features.

---

## 🧱 Architecture
- **UI:** Streamlit + custom CSS
- **CV:** OpenCV Haar cascades for face/eye/smile
- **ASR:** `SpeechRecognition` (Google Web Speech API)
- **NLP Metrics:** custom analysis of fillers / repetition / sentence stats
- **DB:** SQLite3 with **safe migrations** (adds missing columns automatically)
- **Viz:** Matplotlib + Seaborn styles

---

## ✅ Requirements
- **OS:** Windows / macOS / Linux
- **Python:** 3.8–3.12 (3.10+ recommended)
- **Hardware:** Webcam + Microphone (local runs only)
- **Internet:** Required for Google Web Speech transcription

**Python packages**
streamlit
opencv-python
numpy
pandas
matplotlib
seaborn
SpeechRecognition
pyaudio # see install notes below
requests
Pillow

yaml
Copy code

> 🧩 Linux needs system libs for OpenCV GUI: `sudo apt-get update && sudo apt-get install -y libgl1`

---

## ⚙️ Installation (Local)

### 1) Create & activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
2) Install dependencies
bash
Copy code
pip install --upgrade pip
pip install streamlit opencv-python numpy pandas matplotlib seaborn SpeechRecognition requests Pillow
Install PyAudio (microphone driver for SpeechRecognition)
Windows (easiest):

bash
Copy code
pip install pipwin
pipwin install pyaudio
If pipwin fails, download a prebuilt wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/ and install:

bash
Copy code
pip install <downloaded_wheel.whl>
macOS (Intel & Apple Silicon):

bash
Copy code
brew install portaudio
pip install pyaudio
Ubuntu/Debian:

bash
Copy code
sudo apt-get install -y portaudio19-dev python3-pyaudio  # system package
# or build from pip:
pip install pyaudio
If PyAudio is painful, you can first test UI without audio (you’ll see “Limited or No Speech Detected”). PyAudio is only required to record audio from your mic.

▶️ Run
Place your app.py in the project root (this repo). Then:

bash
Copy code
streamlit run app.py
Open the local URL (usually http://localhost:8501).

🔑 Optional: Gemini API Key
The app includes a polished, rule-based feedback generator; adding a Gemini key is optional.

In-app: Go to ⚙️ Settings → Google Gemini AI Integration → Enter API Key (stored in session only).

Alternative (recommended security): put the key into Streamlit Secrets:

Create .streamlit/secrets.toml:

toml
Copy code
GEMINI_API_KEY = "your_key_here"
Read it in code with st.secrets["GEMINI_API_KEY"] (you can add this later).

Current code formats “Gemini-style” feedback locally and does not call external APIs unless you integrate it.

🗄️ Database
File: presentation_history.db

Table: sessions

Automatic schema migration on launch (adds missing columns like ai_feedback, scores, etc.)

Reset: Settings → “Reset Statistics” (deletes all sessions)

If you ever see:
sqlite3.OperationalError: table sessions has no column named ai_feedback
this means you inserted before migration. The included init_db() runs at startup to fix this. If it persists:

Stop the app

Delete presentation_history.db

Run again

🌐 Deployment Notes (Important)
This app uses OpenCV VideoCapture(0) and PyAudio to access your local camera & mic from the server process.
That works great locally because the browser and Python run on the same machine.

On the cloud, your server cannot access a remote user’s webcam/mic with OpenCV/PyAudio. For browser-based capture you need WebRTC (e.g., streamlit-webrtc).

Best options
Local run (recommended for full features) — everything works.

Streamlit Community Cloud / Hugging Face / Render / Railway:

App will deploy, but webcam/mic features won’t work with current OpenCV/PyAudio approach.

To make it truly web-native, replace capture with streamlit-webrtc.

Why Netlify didn’t work
Netlify is a static hosting platform and cannot serve Python/Streamlit backends. Use Streamlit Cloud, Hugging Face Spaces (Streamlit runtime), Render, or Railway for Python apps.

Example: Render deployment (headless)
Push code to GitHub (include requirements.txt)

Create a Web Service

Start command:

bash
Copy code
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
Camera/mic will not work remotely without WebRTC.

🧪 Troubleshooting
Camera
“Cannot access webcam”

Close other apps using camera (Zoom/Meet)

OS permissions: allow Python/Terminal to access camera

Try VideoCapture(1) if you have multiple cameras

Microphone
“No microphones found” / empty list

OS sound settings → set a default input device

Grant mic permission to Python/Terminal

Install PyAudio properly (see above)

“RequestError” in speech recognition

Internet required (Google Web Speech)

API rate-limiting possible → try again later

Consider offline engines (e.g., Vosk) if needed

OpenCV on Linux
libGL.so.1 missing → sudo apt-get install -y libgl1

SQLite migration errors
Make sure init_db() runs before any inserts (already done in code)

Delete presentation_history.db and relaunch if schema is corrupted

🧪 System Check Page
Use 🧪 System Check to:

Verify camera & face/eye detection

Test microphone and short transcription

Inspect installed library versions

Confirm DB health and session count

📦 Suggested requirements.txt
ini
Copy code
streamlit==1.39.0
opencv-python==4.10.0.84
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
SpeechRecognition==3.10.4
pyaudio==0.2.14
requests==2.32.3
Pillow==10.3.0
If pyaudio fails on CI/cloud, remove it from requirements.txt to let UI load; audio features will be disabled.

🔭 Roadmap (nice-to-have)
WebRTC input via streamlit-webrtc (true browser-based camera/mic)

Offline ASR fallback (Vosk/Whisper-cpp)

True Gemini API integration (with st.secrets)

More robust emotion detection (CNN-based)

Multi-user auth + cloud DB option

📜 License
For academic use in DES646. If you plan to reuse outside the course, add an explicit license file (MIT/Apache-2.0) as needed.

🙌 Credits
Built with ❤️ using Streamlit, OpenCV, SpeechRecognition, SQLite, Matplotlib, and Seaborn.
