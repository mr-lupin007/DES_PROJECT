# 🎤 Emotion-Aware AI Presentation Coach

An interactive **AI-powered presentation coach** built with **Streamlit** that analyzes your speaking pace, filler words, eye contact, and facial expressions to help you become a more confident and effective presenter.

> ⚠️ **Important:** This app is designed to run **locally on your computer** because it needs direct access to your **webcam** and **microphone**.  
> Static hosts like **Netlify** cannot run this Streamlit app.

---

## ✨ Features

- 🎥 **Live Webcam Recording** – Captures your video using OpenCV.
- 🎙 **Speech-to-Text** – Uses Google Speech API via `SpeechRecognition` to transcribe audio.
- 🗣 **Speaking Pace Analysis** – Computes **words per minute (WPM)** and checks if you’re in the ideal 120–180 WPM range.
- 🧠 **Filler Word Detection** – Counts words like _“um, uh, like, so, actually, basically, literally, you know”_.
- 👀 **Eye Contact Estimation** – Approximates eye contact using OpenCV Haar cascades for face & eye detection.
- 🙂 **Simple Emotion Detection** – Categorizes expressions as **happy**, **confident**, or **neutral**.
- 📊 **Scoring Dashboard** – Gives you:
  - Overall score (/100)
  - Eye contact score (%)
  - Pace score
  - Fluency score
  - Emotion distribution pie chart
- 📈 **History & Progress Tracking** – Stores each session in a local **SQLite** database and visualizes:
  - Score progression
  - WPM over sessions
  - Filler word trends
  - Eye contact improvement
- 💬 **Personalized Feedback** – Text feedback + action items for your next session.

---

## 🛠 Tech Stack

- **Frontend / App Framework:** [Streamlit](https://streamlit.io/)
- **Computer Vision:** OpenCV (Haar cascades for face, eyes, and smile)
- **Speech Recognition:** `SpeechRecognition` (Google Web Speech API)
- **Data Storage:** SQLite (`presentation_history.db`)
- **Data Analysis & Plots:** Pandas, Matplotlib, NumPy
- **Language:** Python 3.8+

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/presentation-coach.git
cd presentation-coach
2. Create and activate a virtual environment (recommended)
Windows (PowerShell)
bash
Copy code
python -m venv venv
venv\Scripts\activate
macOS / Linux
bash
Copy code
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
If you have a requirements.txt:

bash
Copy code
pip install -r requirements.txt
Otherwise, install manually:

bash
Copy code
pip install streamlit opencv-python numpy pandas matplotlib SpeechRecognition pipwin
Install PyAudio (for microphone input on Windows)
bash
Copy code
pip install pipwin
pipwin install pyaudio
If pipwin fails, download a matching PyAudio wheel from
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio and install with:

bash
Copy code
pip install PyAudio-<version>-cp<xy>-cp<xy>-win_amd64.whl
🚀 Running the App
From the project root (where app.py is located):

bash
Copy code
streamlit run app.py
Streamlit will open a browser window automatically.
If not, open the URL shown in the terminal (typically http://localhost:8501).

🎮 How to Use
Go to the “Live Practice” page (default).

Select a recording duration (30s, 1 min, 2 min, etc.).

Click “🔴 Start Live Recording”.

Look at your webcam and speak as if giving a real presentation.

After the timer ends, the app:

Processes your audio & video

Computes metrics

Stores the session in presentation_history.db

Shows:

Overall score

WPM

Filler word count

Eye contact score

Dominant emotion

Transcript (if audio recognized)

Visual dashboards & feedback

Go to “History & Progress” in the sidebar to see:

Average metrics

Progress plots

Session history table

Option to download your history as CSV.

“About” page explains:

Project overview

Features

Tech stack

Course information

Future work

📂 Project Structure
text
Copy code
presentation-coach/
├─ app.py                      # Main Streamlit app
├─ presentation_history.db     # SQLite DB (auto-created on first run)
├─ README.md                   # Project documentation
├─ .gitignore                  # Git ignore file (e.g. venv, cache, etc.)
└─ venv/                       # (Optional) Virtual environment – not committed to Git
The SQLite file presentation_history.db is created in the same directory as app.py and stores all your session metrics.

📊 Metrics Explained
Overall Score (0–100)
Combines filler words, eye contact, pace, and content length.

≥ 80 → Excellent

60–79 → Good

< 60 → Needs improvement

Words Per Minute (WPM)
Calculated from transcript and duration.

Ideal range: 120–180 WPM

Filler Count
Counts occurrences of:

um, uh, like, so, actually, basically, literally, you know

Pause Count
Roughly estimated from "..." and other pause indicators.

Eye Contact Score (%)
Percentage of frames where a face + eyes are detected.

Emotion Distribution
Simple heuristic-based emotions:

happy, confident, neutral

🧪 Known Limitations
Requires a working microphone and webcam.

Uses Google Speech API, so:

Needs an internet connection.

May occasionally miss words or fail on noisy input.

Emotion detection is basic (Haar cascades + smile detection), not a deep learning model.

Designed primarily for local usage, not static deployment:

Will not run on Netlify or other static hosts.

Needs a Python environment running streamlit.

🔧 Troubleshooting
❌ “Could not access webcam”
Make sure no other app is using the camera.

Check OS/browser camera permissions.

On some systems, an external webcam works more reliably.

❌ “No speech detected” / Empty transcript
Check that your microphone is:

Selected as default input device

Not muted

Reduce background noise.

Ensure internet is available (for Google Speech API).

❌ Import errors (e.g. No module named 'cv2')
Run inside your virtual environment:

bash
Copy code
pip install streamlit opencv-python numpy pandas matplotlib SpeechRecognition pipwin
pipwin install pyaudio
🔮 Future Enhancements
Deep-learning-based emotion recognition (e.g. FER/DeepFace).

More accurate gaze estimation (MediaPipe face mesh).

Gesture & posture analysis.

Offline speech recognition (e.g. Whisper).

Multi-language support.

Export detailed PDF reports.

Side-by-side comparison with previous sessions or expert talks.

👥 Contributors / Team
This project was created as part of DES646 – Mid-Term Assessment Project:

Surya Shukla (221109) – Point of Contact

Pushpendra Singh (220841)

Meet Pal Singh (220644)

Vasundhara Agarwal (221177)

Ayush (220259)

📄 License
This project is developed for academic and educational purposes as part of DES646.
Please check with the course team before using it commercially.
