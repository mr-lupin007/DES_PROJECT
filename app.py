import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import json
import time
from collections import Counter
import speech_recognition as sr
import threading
import queue
import requests
from PIL import Image
import io
import sys

# Force UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# Configure page
st.set_page_config(
    page_title="AI Presentation Coach Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPROVED CSS - Fixed text visibility with explicit colors
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3, .metric-card p, .metric-card li, .metric-card strong {
        color: white !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .score-excellent {
        color: #2ecc71;
        font-size: 4rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .score-good {
        color: #f39c12;
        font-size: 4rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .score-poor {
        color: #e74c3c;
        font-size: 4rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feedback-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50 !important;
    }
    .feedback-box p, .feedback-box li, .feedback-box strong, .feedback-box h3, .feedback-box ul {
        color: #2c3e50 !important;
    }
    .ai-insight {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50 !important;
    }
    .ai-insight p, .ai-insight h3, .ai-insight li, .ai-insight strong, .ai-insight h4 {
        color: #2c3e50 !important;
    }
    /* Global text visibility fix */
    .stMarkdown, .stMarkdown p, .stMarkdown h3, .stMarkdown li, .stMarkdown strong {
        color: #2c3e50 !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'face_cascade' not in st.session_state:
    st.session_state.face_cascade = None
if 'eye_cascade' not in st.session_state:
    st.session_state.eye_cascade = None

# IMPROVED: Load cascades with error handling
def load_cascades():
    """Load Haar Cascades with fallback options"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Verify cascades loaded correctly
        if face_cascade.empty() or eye_cascade.empty():
            st.error("⚠️ Error loading face detection models. Please reinstall OpenCV.")
            return None, None
            
        return face_cascade, eye_cascade
    except Exception as e:
        st.error(f"❌ Cascade loading error: {str(e)}")
        return None, None

# Load cascades on startup
if st.session_state.face_cascade is None:
    st.session_state.face_cascade, st.session_state.eye_cascade = load_cascades()

# Database setup
def init_db():
    conn = sqlite3.connect('presentation_history.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  duration REAL,
                  wpm REAL,
                  filler_count INTEGER,
                  eye_contact_score REAL,
                  emotion_scores TEXT,
                  pause_count INTEGER,
                  overall_score REAL,
                  transcript TEXT,
                  ai_feedback TEXT,
                  confidence_score REAL,
                  clarity_score REAL,
                  engagement_score REAL)''')
    
    cursor = c.execute('PRAGMA table_info(sessions)')
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'ai_feedback' not in columns:
        c.execute('ALTER TABLE sessions ADD COLUMN ai_feedback TEXT')
    if 'confidence_score' not in columns:
        c.execute('ALTER TABLE sessions ADD COLUMN confidence_score REAL')
    if 'clarity_score' not in columns:
        c.execute('ALTER TABLE sessions ADD COLUMN clarity_score REAL')
    if 'engagement_score' not in columns:
        c.execute('ALTER TABLE sessions ADD COLUMN engagement_score REAL')
    
    conn.commit()
    conn.close()

init_db()

# Gemini AI Integration
def get_gemini_feedback(transcript, metrics):
    """Get AI-powered feedback using Gemini API"""
    if not st.session_state.gemini_api_key:
        return generate_rule_based_feedback(metrics)
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.session_state.gemini_api_key}"
        
        prompt = f"""As an expert public speaking coach, analyze this presentation:

TRANSCRIPT: "{transcript}"

METRICS:
- Words per minute: {metrics['wpm']:.0f}
- Filler words count: {metrics['filler_count']}
- Eye contact score: {metrics['eye_contact_score']:.0f}%
- Duration: {metrics['duration']} seconds
- Word count: {metrics['word_count']}

Provide a detailed, professional analysis covering:
1. Content Quality & Structure (2-3 sentences)
2. Delivery & Pace Assessment (2-3 sentences)
3. Body Language & Engagement (2-3 sentences)
4. Top 3 Specific Improvement Actions
5. Top 3 Strengths to Maintain

Be encouraging but honest. Format with clear sections."""

        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return generate_rule_based_feedback(metrics)
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return generate_rule_based_feedback(metrics)

def generate_rule_based_feedback(metrics):
    """Generate detailed rule-based feedback"""
    feedback = "### 📊 Professional Analysis\n\n"
    
    # Content Quality
    feedback += "**🎯 Content & Structure:**\n"
    if metrics['word_count'] > 100:
        feedback += "Your presentation demonstrated substantial content depth. "
    elif metrics['word_count'] > 50:
        feedback += "Your presentation covered key points adequately. "
    else:
        feedback += "Consider expanding your content for more comprehensive coverage. "
    
    if metrics['filler_count'] < 3:
        feedback += "The message was delivered with clarity and precision.\n\n"
    elif metrics['filler_count'] < 8:
        feedback += "The message was generally clear with room for refinement.\n\n"
    else:
        feedback += "Focus on eliminating verbal fillers to enhance message clarity.\n\n"
    
    # Delivery
    feedback += "**🎤 Delivery & Pace:**\n"
    if 120 <= metrics['wpm'] <= 180:
        feedback += f"Excellent pacing at {metrics['wpm']:.0f} WPM - within the optimal range for audience comprehension. "
    elif metrics['wpm'] < 120:
        feedback += f"Your pace of {metrics['wpm']:.0f} WPM is slower than optimal. Consider increasing energy and tempo. "
    else:
        feedback += f"Your pace of {metrics['wpm']:.0f} WPM is quite fast. Slow down to ensure clarity and audience retention. "
    
    feedback += "Professional speakers aim for 120-180 words per minute.\n\n"
    
    # Engagement
    feedback += "**👁️ Body Language & Engagement:**\n"
    if metrics['eye_contact_score'] > 75:
        feedback += f"Outstanding eye contact at {metrics['eye_contact_score']:.0f}%. You maintained strong visual connection with your audience. "
    elif metrics['eye_contact_score'] > 60:
        feedback += f"Good eye contact at {metrics['eye_contact_score']:.0f}%. Work on maintaining this throughout. "
    else:
        feedback += f"Eye contact at {metrics['eye_contact_score']:.0f}% needs improvement. This is crucial for audience engagement. "
    
    feedback += "Strong eye contact builds trust and credibility.\n\n"
    
    # Strengths
    feedback += "**✅ Key Strengths:**\n"
    strengths = []
    if metrics['wpm'] >= 120 and metrics['wpm'] <= 180:
        strengths.append("Optimal speaking pace")
    if metrics['filler_count'] < 5:
        strengths.append("Minimal use of filler words")
    if metrics['eye_contact_score'] > 70:
        strengths.append("Strong audience engagement")
    if metrics['word_count'] > 80:
        strengths.append("Comprehensive content coverage")
    
    if not strengths:
        strengths = ["Willingness to practice", "Taking initiative to improve", "Self-awareness"]
    
    for i, strength in enumerate(strengths[:3], 1):
        feedback += f"{i}. {strength}\n"
    
    feedback += "\n**🎯 Priority Improvements:**\n"
    improvements = []
    if metrics['filler_count'] > 5:
        improvements.append(f"Reduce filler words - currently at {metrics['filler_count']}, target <5")
    if metrics['eye_contact_score'] < 70:
        improvements.append(f"Increase eye contact from {metrics['eye_contact_score']:.0f}% to 75%+")
    if metrics['wpm'] < 120:
        improvements.append(f"Increase speaking pace from {metrics['wpm']:.0f} to 120+ WPM")
    if metrics['wpm'] > 180:
        improvements.append(f"Decrease speaking pace from {metrics['wpm']:.0f} to 150-170 WPM")
    if metrics['word_count'] < 50:
        improvements.append("Develop more substantial content")
    
    if not improvements:
        improvements = ["Maintain current performance", "Add more vocal variety", "Incorporate strategic pauses"]
    
    for i, improvement in enumerate(improvements[:3], 1):
        feedback += f"{i}. {improvement}\n"
    
    return feedback

# Helper Functions
def save_session(data):
    conn = sqlite3.connect('presentation_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO sessions 
                 (date, duration, wpm, filler_count, eye_contact_score, 
                  emotion_scores, pause_count, overall_score, transcript, ai_feedback,
                  confidence_score, clarity_score, engagement_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (data['date'], data['duration'], data['wpm'], 
               data['filler_count'], data['eye_contact_score'],
               json.dumps(data['emotion_scores']), data['pause_count'],
               data['overall_score'], data.get('transcript', ''),
               data.get('ai_feedback', ''), data.get('confidence_score', 0),
               data.get('clarity_score', 0), data.get('engagement_score', 0)))
    conn.commit()
    conn.close()

def get_session_history():
    conn = sqlite3.connect('presentation_history.db')
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY date DESC", conn)
    conn.close()
    return df

def analyze_speech_text(text):
    """Advanced speech analysis"""
    if not text or len(text.strip()) == 0:
        return {'word_count': 0, 'filler_count': 0, 'pause_count': 0, 'unique_words': 0, 'avg_word_length': 0}
    
    words = text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    
    # Filler words detection
    filler_words = ['um', 'uh', 'like', 'so', 'actually', 'basically', 'literally', 'you know', 'i mean', 'kind of', 'sort of']
    filler_count = sum(text.lower().count(filler) for filler in filler_words)
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Pause detection
    pause_indicators = text.count('...') + text.count('..') + text.count(' - ')
    
    return {
        'word_count': word_count,
        'filler_count': filler_count,
        'pause_count': max(0, pause_indicators),
        'unique_words': unique_words,
        'avg_word_length': avg_word_length
    }

# IMPROVED: Robust face and eye detection
def detect_face_and_eyes(frame):
    """Enhanced face and eye detection with better error handling"""
    try:
        if st.session_state.face_cascade is None or st.session_state.eye_cascade is None:
            return False, False, 0
        
        # Convert to grayscale with error handling
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            return False, False, 0
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # IMPROVED: Better parameters for face detection
        faces = st.session_state.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # More sensitive
            minNeighbors=5,   # Balance between false positives and detection
            minSize=(60, 60), # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        eye_contact = False
        face_detected = False
        face_size = 0
        
        for (x, y, w, h) in faces:
            face_detected = True
            face_size = w * h
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Extract face ROI for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # IMPROVED: Better parameters for eye detection
            eyes = st.session_state.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=7,    # Reduced false positives
                minSize=(20, 20),
                maxSize=(80, 80)
            )
            
            if len(eyes) >= 2:
                eye_contact = True
                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes[:2]:  # Only first two eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            break  # Only process first face
        
        return eye_contact, face_detected, face_size
    
    except Exception as e:
        print(f"Face detection error: {str(e)}")
        return False, False, 0

def calculate_advanced_scores(metrics, transcript):
    """Calculate advanced presentation scores"""
    
    # Confidence Score
    confidence = 100
    if metrics['wpm'] < 100:
        confidence -= 20
    if metrics['filler_count'] > 10:
        confidence -= 30
    elif metrics['filler_count'] > 5:
        confidence -= 15
    
    confidence_score = max(0, min(100, confidence))
    
    # Clarity Score
    clarity = 100
    if metrics['filler_count'] > 8:
        clarity -= 25
    if metrics['wpm'] > 200:
        clarity -= 20
    if metrics.get('unique_words', 0) > 0 and metrics['word_count'] > 0:
        vocabulary_ratio = metrics['unique_words'] / metrics['word_count']
        if vocabulary_ratio < 0.3:
            clarity -= 10
    
    clarity_score = max(0, min(100, clarity))
    
    # Engagement Score
    engagement = metrics['eye_contact_score']
    if 120 <= metrics['wpm'] <= 180:
        engagement += 10
    engagement = min(100, engagement)
    
    engagement_score = max(0, min(100, engagement))
    
    return confidence_score, clarity_score, engagement_score

# IMPROVED: Speech recognition with better error handling
def record_audio_continuous(duration, result_queue, mic_index=None):
    """Enhanced audio recording with better noise handling"""
    recognizer = sr.Recognizer()
    
    # IMPROVED: Better recognition parameters
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 400  # Increased for better noise rejection
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    
    full_text = ""
    
    try:
        if mic_index is not None:
            mic = sr.Microphone(device_index=mic_index)
        else:
            mic = sr.Microphone()
        
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            start_time = time.time()
            chunks_processed = 0
            
            while time.time() - start_time < duration:
                try:
                    remaining = duration - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    
                    chunk_duration = min(5, remaining)
                    audio = recognizer.listen(source, timeout=chunk_duration, phrase_time_limit=chunk_duration)
                    
                    try:
                        text = recognizer.recognize_google(audio, language='en-US', show_all=False)
                        if text:
                            full_text += " " + text
                            chunks_processed += 1
                            print(f"Recognized chunk {chunks_processed}: {text[:50]}...")
                    except sr.UnknownValueError:
                        print(f"Could not understand chunk {chunks_processed}")
                        continue
                    except sr.RequestError as e:
                        print(f"API error: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Listen error: {e}")
                    continue
                    
    except Exception as e:
        result_queue.put(f"ERROR: {str(e)}")
        return
    
    final_text = full_text.strip()
    print(f"Total text recognized: {len(final_text)} characters")
    result_queue.put(final_text if final_text else "")

def calculate_overall_score(metrics):
    """Enhanced scoring algorithm"""
    score = 100
    
    # Filler words penalty (max -25)
    if metrics['filler_count'] > 15:
        score -= 25
    elif metrics['filler_count'] > 10:
        score -= 20
    elif metrics['filler_count'] > 5:
        score -= 10
    elif metrics['filler_count'] > 3:
        score -= 5
    
    # Eye contact penalty (max -25)
    if metrics['eye_contact_score'] < 40:
        score -= 25
    elif metrics['eye_contact_score'] < 60:
        score -= 15
    elif metrics['eye_contact_score'] < 75:
        score -= 8
    
    # Pacing penalty (max -20)
    if metrics['wpm'] < 100:
        score -= 20
    elif metrics['wpm'] < 120:
        score -= 10
    elif metrics['wpm'] > 200:
        score -= 20
    elif metrics['wpm'] > 180:
        score -= 10
    
    # Content penalty (max -15)
    if metrics['word_count'] < 20:
        score -= 15
    elif metrics['word_count'] < 40:
        score -= 8
    
    # Bonus for excellence
    if metrics['wpm'] >= 140 and metrics['wpm'] <= 160:
        score += 5
    if metrics['eye_contact_score'] > 85:
        score += 5
    
    return max(0, min(100, score))

def plot_advanced_metrics(metrics, history_df=None):
    """Create professional visualization dashboard"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    primary_color = '#667eea'
    secondary_color = '#764ba2'
    
    # 1. Overall Score Gauge
    ax1 = fig.add_subplot(gs[0, :2])
    score = metrics['overall_score']
    colors_map = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    color_idx = min(int(score / 20), 4)
    
    ax1.barh([0], [score], color=colors_map[color_idx], height=0.5, alpha=0.8)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Overall Score', fontsize=14, fontweight='bold')
    ax1.set_yticks([])
    ax1.text(score + 2, 0, f'{score:.1f}', va='center', fontsize=20, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_title('Overall Presentation Score', fontsize=16, fontweight='bold', pad=20)
    
    # 2. Multi-dimensional scores
    ax2 = fig.add_subplot(gs[0, 2])
    categories = ['Confidence', 'Clarity', 'Engagement']
    scores = [
        metrics.get('confidence_score', 70),
        metrics.get('clarity_score', 70),
        metrics.get('engagement_score', 70)
    ]
    colors = ['#667eea', '#764ba2', '#f093fb']
    ax2.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Skill Breakdown', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(scores):
        ax2.text(i, v + 3, f'{v:.0f}', ha='center', fontweight='bold')
    
    # 3. WPM Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    wpm = metrics['wpm']
    optimal_range = [120, 180]
    ax3.axhspan(optimal_range[0], optimal_range[1], alpha=0.3, color='green', label='Optimal Range')
    ax3.barh(['Your WPM'], [wpm], color=primary_color, height=0.4, alpha=0.8)
    ax3.set_xlim(0, max(250, wpm + 20))
    ax3.set_xlabel('Words Per Minute', fontsize=11)
    ax3.set_title('Speaking Pace Analysis', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.text(wpm + 5, 0, f'{wpm:.0f}', va='center', fontsize=14, fontweight='bold')
    
    # 4. Filler Words
    ax4 = fig.add_subplot(gs[1, 1])
    filler_data = [metrics['filler_count'], max(0, 15 - metrics['filler_count'])]
    colors_filler = ['#e74c3c', '#2ecc71']
    ax4.pie(filler_data, labels=['Fillers', 'Clean Speech'], autopct='%1.1f%%',
            colors=colors_filler, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title(f'Filler Words: {metrics["filler_count"]}', fontsize=12, fontweight='bold')
    
    # 5. Eye Contact
    ax5 = fig.add_subplot(gs[1, 2])
    eye_score = metrics['eye_contact_score']
    remaining = 100 - eye_score
    colors_eye = ['#667eea', '#e0e0e0']
    ax5.pie([eye_score, remaining], labels=['Eye Contact', 'Away'],
            autopct='%1.1f%%', colors=colors_eye, startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Eye Contact Distribution', fontsize=12, fontweight='bold')
    
    # 6. Historical Trend
    ax6 = fig.add_subplot(gs[2, :])
    if history_df is not None and len(history_df) > 1:
        recent = history_df.head(10).iloc[::-1]
        sessions = range(1, len(recent) + 1)
        
        ax6.plot(sessions, recent['overall_score'], marker='o', linewidth=3,
                color=primary_color, markersize=10, label='Overall Score')
        ax6.plot(sessions, recent['eye_contact_score'], marker='s', linewidth=2,
                color=secondary_color, markersize=8, alpha=0.7, label='Eye Contact')
        
        ax6.set_xlabel('Session Number', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax6.set_title('Performance Progression', fontsize=14, fontweight='bold')
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 105)
        
        z = np.polyfit(sessions, recent['overall_score'], 1)
        p = np.poly1d(z)
        ax6.plot(sessions, p(sessions), "--", alpha=0.5, color='gray', label='Trend')
    else:
        ax6.text(0.5, 0.5, 'Complete more sessions to see progression',
                ha='center', va='center', fontsize=14, transform=ax6.transAxes)
        ax6.set_title('Performance Progression', fontsize=14, fontweight='bold')
    
    plt.suptitle('🎯 Comprehensive Presentation Analytics', fontsize=18, fontweight='bold', y=0.995)
    
    return fig

def test_microphone():
    """Test microphone functionality"""
    recognizer = sr.Recognizer()
    try:
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            return False, "No microphones found", []
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            return True, f"Microphone working! Found {len(mic_list)} device(s)", mic_list
    except Exception as e:
        return False, f"Microphone error: {str(e)}", []

# Main Application
st.markdown('<h1 class="main-header">🎯 AI Presentation Coach Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Multimodal Analysis with AI-Powered Insights</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/microphone.png", width=80)
    st.title("🎛️ Control Panel")
    
    page = st.radio("", [
        "🎤 Live Practice",
        "🧪 System Check",
        "📊 Analytics Dashboard",
        "⚙️ AI Settings",
        "ℹ️ About Project"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    history_df = get_session_history()
    if len(history_df) > 0:
        st.metric("Total Sessions", len(history_df))
        st.metric("Avg Score", f"{history_df['overall_score'].mean():.1f}")
        st.metric("Best Score", f"{history_df['overall_score'].max():.1f}")
    else:
        st.info("No sessions yet")
    
    st.markdown("---")
    st.markdown("### 🎓 DES646 Project")
    st.caption("Team: Surya, Pushpendra, Meet, Vasundhara, Ayush")

# PAGE 1: LIVE PRACTICE
if "🎤 Live Practice" in page:
    st.header("🎥 Professional Practice Session")
    
    duration_options = {
        "30 seconds - Quick Practice": 30,
        "1 minute - Short Pitch": 60,
        "2 minutes - Standard Presentation": 120,
        "3 minutes - Detailed Presentation": 180,
        "5 minutes - Full Presentation": 300
    }
    
    if not st.session_state.recording and not st.session_state.analysis_complete:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="feedback-box">
            <h3 style="color: #2c3e50 !important;">📋 Session Guidelines</h3>
            <ul style="color: #2c3e50 !important;">
                <li><strong>Environment:</strong> Quiet space with good lighting</li>
                <li><strong>Position:</strong> Face camera directly, maintain eye contact</li>
                <li><strong>Delivery:</strong> Speak clearly at conversational pace</li>
                <li><strong>Content:</strong> Present as if to a real audience</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            selected_duration = st.selectbox(
                "🕒 Select Duration:",
                options=list(duration_options.keys()),
                index=2
            )
            
            st.session_state.selected_duration = selected_duration
            duration = duration_options[selected_duration]
            
            try:
                mic_list = sr.Microphone.list_microphone_names()
                if len(mic_list) > 1:
                    mic_choice = st.selectbox(
                        "🎤 Select Microphone:",
                        options=["Default (Recommended)"] + [f"Device {i}: {name[:30]}" for i, name in enumerate(mic_list)]
                    )
                    if "Device" in mic_choice:
                        st.session_state.selected_mic_index = int(mic_choice.split()[1].replace(":", ""))
                    else:
                        st.session_state.selected_mic_index = None
                else:
                    st.session_state.selected_mic_index = None
            except:
                st.session_state.selected_mic_index = None
            
            st.markdown("---")
            
            if st.button("🔴 START RECORDING", type="primary", use_container_width=True):
                success, _, _ = test_microphone()
                if not success:
                    st.error("❌ Microphone not detected! Please check System Check page.")
                else:
                    st.session_state.recording = True
                    st.rerun()
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px;">
            <h3 style="color: white !important; margin-bottom: 1rem;">🎯 Target Metrics</h3>
            <ul style="color: white !important; list-style: none; padding-left: 0;">
                <li style="color: white !important; margin-bottom: 0.5rem;"><strong style="color: white !important;">WPM:</strong> 120-180</li>
                <li style="color: white !important; margin-bottom: 0.5rem;"><strong style="color: white !important;">Fillers:</strong> < 5</li>
                <li style="color: white !important; margin-bottom: 0.5rem;"><strong style="color: white !important;">Eye Contact:</strong> > 75%</li>
                <li style="color: white !important;"><strong style="color: white !important;">Score:</strong> 80+</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; border: 2px solid #667eea;">
            <h3 style="color: #2c3e50 !important;">💡 Pro Tips</h3>
            <ul style="color: #2c3e50 !important;">
                <li>Breathe deeply first</li>
                <li>Smile naturally</li>
                <li>Use gestures</li>
                <li>Pause for emphasis</li>
                <li>Be authentic</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.recording:
        selected_duration = st.session_state.selected_duration
        duration = duration_options[selected_duration]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🔴 RECORDING IN PROGRESS</h2>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{selected_duration.split(' - ')[0]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_display = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera permissions.")
            st.session_state.recording = False
            st.stop()
        
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(
            target=record_audio_continuous,
            args=(duration, audio_queue, st.session_state.selected_mic_index)
        )
        audio_thread.start()
        
        start_time = time.time()
        frame_count = 0
        eye_contact_frames = 0
        total_frames = 0
        emotion_list = []
        face_sizes = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            eye_contact, face_detected, face_size = detect_face_and_eyes(frame)
            
            if face_detected:
                total_frames += 1
                face_sizes.append(face_size)
                if eye_contact:
                    eye_contact_frames += 1
                    emotion_list.append('confident')
                else:
                    emotion_list.append('neutral')
                
                # IMPROVED: UTF-8 safe text overlay
                cv2.putText(frame, "Face: OK", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if eye_contact:
                    cv2.putText(frame, "Eye Contact: GOOD", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Eye Contact: WEAK", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 140, 0), 2)
            else:
                cv2.putText(frame, "Face: NOT DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            cv2.putText(frame, f"Time: {int(remaining)}s", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            progress = min(elapsed / duration, 1.0)
            progress_bar.progress(progress)
            status_text.markdown(f"**Recording:** {int(elapsed)}s / {duration}s")
            
            if total_frames > 0:
                current_eye_contact = (eye_contact_frames / total_frames) * 100
                metrics_display.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; color: #2c3e50 !important;">
                <strong>Live Metrics:</strong> Eye Contact: {current_eye_contact:.0f}% | Frames: {total_frames}
                </div>
                """, unsafe_allow_html=True)
            
            frame_count += 1
            time.sleep(0.03)
        
        cap.release()
        st.session_state.recording = False
        
        status_text.markdown("**⏳ Processing audio... Please wait...**")
        audio_thread.join(timeout=10)
        
        try:
            transcribed_text = audio_queue.get_nowait()
            if transcribed_text.startswith("ERROR:"):
                transcribed_text = ""
        except:
            transcribed_text = ""
        
        speech_metrics = analyze_speech_text(transcribed_text)
        
        wpm = (speech_metrics['word_count'] / duration) * 60 if duration > 0 else 0
        eye_contact_score = (eye_contact_frames / total_frames) * 100 if total_frames > 0 else 0
        
        emotion_counter = Counter(emotion_list)
        emotion_scores = dict(emotion_counter)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
        
        metrics = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration': duration,
            'wpm': wpm,
            'word_count': speech_metrics['word_count'],
            'filler_count': speech_metrics['filler_count'],
            'pause_count': speech_metrics['pause_count'],
            'eye_contact_score': eye_contact_score,
            'emotion_scores': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'transcript': transcribed_text,
            'unique_words': speech_metrics.get('unique_words', 0),
            'avg_word_length': speech_metrics.get('avg_word_length', 0),
            'overall_score': 0
        }
        
        confidence_score, clarity_score, engagement_score = calculate_advanced_scores(metrics, transcribed_text)
        metrics['confidence_score'] = confidence_score
        metrics['clarity_score'] = clarity_score
        metrics['engagement_score'] = engagement_score
        
        metrics['overall_score'] = calculate_overall_score(metrics)
        
        status_text.markdown("**🤖 Generating AI insights...**")
        ai_feedback = get_gemini_feedback(transcribed_text, metrics)
        metrics['ai_feedback'] = ai_feedback
        
        save_session(metrics)
        
        st.session_state.session_data = metrics
        st.session_state.analysis_complete = True
        
        st.success("✅ Analysis Complete!")
        time.sleep(1)
        st.rerun()
    
    elif st.session_state.analysis_complete:
        metrics = st.session_state.session_data
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            score = metrics['overall_score']
            if score >= 85:
                score_class = "score-excellent"
                emoji = "🌟"
                level = "OUTSTANDING"
                message = "Exceptional performance! You're presentation-ready!"
            elif score >= 75:
                score_class = "score-excellent"
                emoji = "⭐"
                level = "EXCELLENT"
                message = "Great job! Minor refinements will make you perfect!"
            elif score >= 65:
                score_class = "score-good"
                emoji = "👍"
                level = "GOOD"
                message = "Solid performance with room for improvement!"
            elif score >= 50:
                score_class = "score-good"
                emoji = "📈"
                level = "DEVELOPING"
                message = "You're on the right track. Keep practicing!"
            else:
                score_class = "score-poor"
                emoji = "💪"
                level = "NEEDS WORK"
                message = "Focus on the feedback. Improvement is coming!"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 20px; border: 3px solid #667eea;">
                <div class="{score_class}">{emoji} {score:.1f}</div>
                <h2 style="margin: 0; color: #667eea;">{level}</h2>
                <p style="font-size: 1.1rem; color: #666; margin-top: 1rem;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            wpm = metrics['wpm']
            wpm_status = "🟢" if 120 <= wpm <= 180 else "🟡" if 100 <= wpm <= 200 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h3 style="color: white; margin: 0;">{wpm_status} {wpm:.0f}</h3>
                <p style="margin: 0.5rem 0 0 0; color: white;">Words/Minute</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fillers = metrics['filler_count']
            filler_status = "🟢" if fillers < 5 else "🟡" if fillers < 10 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h3 style="color: white; margin: 0;">{filler_status} {fillers}</h3>
                <p style="margin: 0.5rem 0 0 0; color: white;">Filler Words</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            eye_contact = metrics['eye_contact_score']
            eye_status = "🟢" if eye_contact > 75 else "🟡" if eye_contact > 60 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h3 style="color: white; margin: 0;">{eye_status} {eye_contact:.0f}%</h3>
                <p style="margin: 0.5rem 0 0 0; color: white;">Eye Contact</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            confidence = metrics.get('confidence_score', 70)
            conf_status = "🟢" if confidence > 75 else "🟡" if confidence > 60 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h3 style="color: white; margin: 0;">{conf_status} {confidence:.0f}%</h3>
                <p style="margin: 0.5rem 0 0 0; color: white;">Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if metrics.get('transcript') and len(metrics['transcript']) > 10:
            with st.expander("📝 Full Transcript", expanded=False):
                st.markdown(f"""
                <div class="feedback-box">
                <p style="font-size: 1.1rem; line-height: 1.8; color: #2c3e50 !important;">{metrics['transcript']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("""
            ⚠️ **Limited/No Speech Detected**
            
            **Possible reasons:**
            - Microphone was muted
            - Background noise too high
            - Speaking too softly
            - Internet connection issue
            
            **Solutions:**
            - Check microphone settings
            - Test in System Check page
            - Speak louder and clearer
            - Ensure stable internet
            """)
        
        st.markdown("### 📊 Comprehensive Analytics")
        fig = plot_advanced_metrics(metrics, get_session_history())
        st.pyplot(fig)
        
        st.markdown("### 🤖 AI-Powered Insights")
        st.markdown(f"""
        <div class="ai-insight">
        {metrics.get('ai_feedback', 'No AI feedback available')}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Practice Again", use_container_width=True, type="primary"):
                st.session_state.analysis_complete = False
                st.session_state.session_data = None
                st.rerun()
        
        with col2:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.session_data = None
                st.rerun()
        
        with col3:
            report_data = f"""
PRESENTATION ANALYSIS REPORT
Generated: {metrics['date']}
============================

OVERALL SCORE: {metrics['overall_score']:.1f}/100

KEY METRICS:
- Words Per Minute: {metrics['wpm']:.0f}
- Filler Words: {metrics['filler_count']}
- Eye Contact: {metrics['eye_contact_score']:.0f}%
- Confidence: {metrics.get('confidence_score', 0):.0f}%
- Clarity: {metrics.get('clarity_score', 0):.0f}%
- Engagement: {metrics.get('engagement_score', 0):.0f}%

TRANSCRIPT:
{metrics.get('transcript', 'No transcript available')}

AI FEEDBACK:
{metrics.get('ai_feedback', 'No AI feedback available')}
"""
            st.download_button(
                "📥 Export Report",
                report_data,
                file_name=f"presentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# PAGE 2: SYSTEM CHECK
elif "🧪 System Check" in page:
    st.header("🔧 System Diagnostics")
    
    st.markdown("""
    <div class="feedback-box">
    <h3 style="color: #2c3e50 !important;">System Requirements Check</h3>
    <p style="color: #2c3e50 !important;">Testing all components required for the AI Presentation Coach to function properly.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Camera Test")
        if st.button("Test Camera", use_container_width=True):
            with st.spinner("Testing camera..."):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Camera Preview", use_container_width=True)
                        st.success("✅ Camera is working properly!")
                    else:
                        st.error("❌ Camera detected but cannot capture frames")
                    cap.release()
                else:
                    st.error("❌ No camera detected. Please connect a webcam.")
    
    with col2:
        st.subheader("🎤 Microphone Test")
        if st.button("Test Microphone", use_container_width=True):
            with st.spinner("Testing microphone..."):
                success, message, mic_list = test_microphone()
                if success:
                    st.success(f"✅ {message}")
                    if mic_list:
                        with st.expander("Available Microphones"):
                            for i, mic in enumerate(mic_list):
                                st.write(f"{i}. {mic}")
                else:
                    st.error(f"❌ {message}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👁️ Face Detection Test")
        if st.button("Test Face Detection", use_container_width=True):
            with st.spinner("Testing face detection..."):
                if st.session_state.face_cascade is not None:
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.flip(frame, 1)
                            eye_contact, face_detected, face_size = detect_face_and_eyes(frame)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption="Face Detection Preview", use_container_width=True)
                            
                            if face_detected:
                                st.success("✅ Face detection working!")
                                st.info(f"Eye Contact: {'✅ Detected' if eye_contact else '❌ Not Detected'}")
                            else:
                                st.warning("⚠️ No face detected. Ensure good lighting and face the camera.")
                        cap.release()
                    else:
                        st.error("❌ Cannot access camera")
                else:
                    st.error("❌ Face detection models not loaded")
    
    with col2:
        st.subheader("🗣️ Speech Recognition Test")
        if st.button("Test Speech Recognition (5s)", use_container_width=True):
            st.info("Speak clearly for 5 seconds...")
            test_queue = queue.Queue()
            test_thread = threading.Thread(target=record_audio_continuous, args=(5, test_queue, None))
            test_thread.start()
            
            progress_bar = st.progress(0)
            for i in range(50):
                time.sleep(0.1)
                progress_bar.progress((i + 1) / 50)
            
            test_thread.join()
            
            try:
                result = test_queue.get_nowait()
                if result and not result.startswith("ERROR"):
                    st.success("✅ Speech recognition working!")
                    st.write(f"**Transcribed:** {result}")
                elif result.startswith("ERROR"):
                    st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ No speech detected. Speak louder or check microphone.")
            except:
                st.error("❌ Speech recognition failed")
    
    st.markdown("---")
    
    st.subheader("📦 Dependencies Check")
    
    dependencies = {
        "OpenCV": cv2,
        "NumPy": np,
        "Pandas": pd,
        "Matplotlib": plt,
        "Seaborn": sns,
        "Speech Recognition": sr,
        "Requests": requests,
        "PIL": Image
    }
    
    dep_cols = st.columns(4)
    for idx, (name, module) in enumerate(dependencies.items()):
        with dep_cols[idx % 4]:
            try:
                version = getattr(module, '__version__', 'Unknown')
                st.success(f"✅ {name}\n\nv{version}")
            except:
                st.success(f"✅ {name}\n\nInstalled")

# PAGE 3: ANALYTICS DASHBOARD
elif "📊 Analytics Dashboard" in page:
    st.header("📊 Performance Analytics Dashboard")
    
    history_df = get_session_history()
    
    if len(history_df) == 0:
        st.info("""
        ### No Data Available Yet
        
        Complete your first practice session to see analytics here!
        
        Go to **🎤 Live Practice** to get started.
        """)
    else:
        # Summary Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{len(history_df)}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Total Sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = history_df['overall_score'].mean()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{avg_score:.1f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Average Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_score = history_df['overall_score'].max()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{best_score:.1f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Best Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_wpm = history_df['wpm'].mean()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{avg_wpm:.0f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Avg WPM</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Progression Chart
        st.subheader("📈 Score Progression")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        recent = history_df.head(20).iloc[::-1]
        sessions = range(1, len(recent) + 1)
        
        ax.plot(sessions, recent['overall_score'], marker='o', linewidth=3,
                color='#667eea', markersize=10, label='Overall Score')
        ax.plot(sessions, recent['eye_contact_score'], marker='s', linewidth=2,
                color='#764ba2', markersize=8, alpha=0.7, label='Eye Contact')
        ax.plot(sessions, recent['wpm'] * 0.5, marker='^', linewidth=2,
                color='#f093fb', markersize=8, alpha=0.7, label='WPM (scaled)')
        
        ax.set_xlabel('Session Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance Over Time', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed Session Table
        st.subheader("📋 Session History")
        
        display_df = history_df[['date', 'overall_score', 'wpm', 'filler_count', 'eye_contact_score', 'duration']].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Date', 'Score', 'WPM', 'Fillers', 'Eye Contact %', 'Duration (s)']
        display_df = display_df.round(1)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full History (CSV)",
            csv,
            "presentation_history.csv",
            "text/csv",
            use_container_width=True
        )

# PAGE 4: AI SETTINGS
elif "⚙️ AI Settings" in page:
    st.header("⚙️ AI Configuration")
    
    st.markdown("""
    <div class="feedback-box">
    <h3 style="color: #2c3e50 !important;">🤖 Google Gemini AI Integration</h3>
    <p style="color: #2c3e50 !important;">
    Enable advanced AI-powered feedback by connecting your Google Gemini API key. 
    Without an API key, the system will use rule-based feedback (still effective).
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔑 API Key Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            placeholder="AIzaSy..."
        )
        
        if st.button("💾 Save API Key", use_container_width=True):
            st.session_state.gemini_api_key = api_key
            st.success("✅ API Key saved successfully!")
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border: 2px solid #667eea;">
        <h4 style="color: #2c3e50 !important; margin-top: 0;">How to Get API Key:</h4>
        <ol style="color: #2c3e50 !important; font-size: 0.9rem;">
            <li>Visit <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
            <li>Click "Get API Key"</li>
            <li>Create new key</li>
            <li>Copy and paste here</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🧪 Test AI Connection")
    
    if st.button("Test Gemini API", use_container_width=True):
        if not st.session_state.gemini_api_key:
            st.warning("⚠️ No API key configured. Using rule-based feedback.")
        else:
            with st.spinner("Testing API connection..."):
                test_metrics = {
                    'wpm': 150,
                    'filler_count': 3,
                    'eye_contact_score': 80,
                    'duration': 60,
                    'word_count': 150
                }
                
                feedback = get_gemini_feedback("This is a test presentation about public speaking skills.", test_metrics)
                
                if "error" in feedback.lower() or len(feedback) < 50:
                    st.error("❌ API test failed. Check your API key.")
                else:
                    st.success("✅ Gemini AI is connected and working!")
                    with st.expander("View Sample AI Feedback"):
                        st.markdown(f"""
                        <div class="ai-insight">
                        {feedback}
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feedback-box">
        <h4 style="color: #2c3e50 !important;">Feedback Mode</h4>
        <p style="color: #2c3e50 !important;">
        <strong>AI-Powered:</strong> Uses Gemini for personalized, context-aware feedback<br>
        <strong>Rule-Based:</strong> Uses predefined rules and thresholds
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feedback-box">
        <h4 style="color: #2c3e50 !important;">Current Status</h4>
        <p style="color: #2c3e50 !important;">
        <strong>Mode:</strong> {}<br>
        <strong>API Key:</strong> {}
        </p>
        </div>
        """.format(
            "AI-Powered ✅" if st.session_state.gemini_api_key else "Rule-Based 📏",
            "Configured ✅" if st.session_state.gemini_api_key else "Not Set ❌"
        ), unsafe_allow_html=True)

# PAGE 5: ABOUT PROJECT
elif "ℹ️ About Project" in page:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🎯 AI Presentation Coach Pro</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Advanced Multimodal Presentation Analysis System</p>
        <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Version 3.0 - Professional Edition with AI Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">🎓 Project Details</h3>
        <ul style="color: #2c3e50 !important;">
            <li><strong>Course:</strong> DES646 - Design Engineering and Analysis</li>
            <li><strong>Institution:</strong> Your University Name</li>
            <li><strong>Year:</strong> 2025</li>
            <li><strong>Category:</strong> AI/ML, Computer Vision, NLP</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">👥 Team Members</h3>
        <ul style="color: #2c3e50 !important;">
            <li>Surya</li>
            <li>Pushpendra</li>
            <li>Meet</li>
            <li>Vasundhara</li>
            <li>Ayush</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">🔧 Technologies Used</h3>
        <ul style="color: #2c3e50 !important;">
            <li><strong>Frontend:</strong> Streamlit</li>
            <li><strong>Computer Vision:</strong> OpenCV, Haar Cascades</li>
            <li><strong>Speech Recognition:</strong> Google Speech API</li>
            <li><strong>AI:</strong> Google Gemini Pro</li>
            <li><strong>Data Analysis:</strong> Pandas, NumPy</li>
            <li><strong>Visualization:</strong> Matplotlib, Seaborn</li>
            <li><strong>Database:</strong> SQLite</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("✨ Key Features")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">🎥 Video Analysis</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Real-time face detection</li>
            <li>Eye contact tracking</li>
            <li>Emotion recognition</li>
            <li>Body language analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">🗣️ Audio Analysis</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Speech-to-text transcription</li>
            <li>Speaking pace (WPM)</li>
            <li>Filler words detection</li>
            <li>Pause pattern analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col3:
        st.markdown("""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">🤖 AI Insights</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Personalized feedback</li>
            <li>Performance scoring</li>
            <li>Progress tracking</li>
            <li>Improvement suggestions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🎯 Project Objectives")
    
    st.markdown("""
    <div class="feedback-box">
    <ol style="color: #2c3e50 !important; line-height: 2;">
        <li><strong>Develop a comprehensive multimodal analysis system</strong> combining computer vision and natural language processing</li>
        <li><strong>Provide actionable feedback</strong> to help users improve their presentation skills</li>
        <li><strong>Integrate AI capabilities</strong> for intelligent, context-aware analysis</li>
        <li><strong>Create an intuitive interface</strong> that makes professional coaching accessible to everyone</li>
        <li><strong>Track progress over time</strong> with detailed analytics and visualizations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📚 Use Cases")
    
    use_cases_col1, use_cases_col2 = st.columns(2)
    
    with use_cases_col1:
        st.markdown("""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">🎓 Education</h4>
        <p style="color: #2c3e50 !important;">
        Students preparing for presentations, thesis defenses, or public speaking assignments
        </p>
        </div>
        
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">💼 Professional</h4>
        <p style="color: #2c3e50 !important;">
        Business professionals practicing for pitches, meetings, or conference talks
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with use_cases_col2:
        st.markdown("""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">🎤 Public Speaking</h4>
        <p style="color: #2c3e50 !important;">
        Aspiring speakers building confidence and refining their delivery style
        </p>
        </div>
        
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important;">👨‍🏫 Training</h4>
        <p style="color: #2c3e50 !important;">
        Trainers and coaches monitoring student progress and providing feedback
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 15px;">
        <p style="margin: 0; color: #2c3e50; font-size: 1.1rem;">
        Built with ❤️ for Better Public Speaking | © 2025 DES646 Team
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3 style="color: white; margin: 0;">🎯 AI Presentation Coach Pro</h3>
    <p style="margin: 0.5rem 0; opacity: 0.9; color: white;">Advanced Multimodal Presentation Analysis System</p>
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8; color: white;">DES646 Course Project © 2025 | Built with ❤️ for Better Public Speaking</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7; color: white;">Version 3.0 - Professional Edition with AI Integration</p>
</div>
""", unsafe_allow_html=True)
