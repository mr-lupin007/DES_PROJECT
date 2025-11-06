"""
AI PRESENTATION COACH PRO - ULTIMATE EDITION v4.0
=================================================
Complete Professional Presentation Analysis System
Team: Surya, Pushpendra, Meet, Vasundhara, Ayush
Course: DES646 - Design Engineering
"""

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
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set plotting style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AI Presentation Coach Pro - Ultimate Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    .metric-card h3, .metric-card p, .metric-card li, .metric-card strong {
        color: white !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    
    /* Score Displays */
    .score-excellent {
        color: #2ecc71;
        font-size: 5rem;
        font-weight: bold;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .score-good {
        color: #f39c12;
        font-size: 5rem;
        font-weight: bold;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.2);
    }
    
    .score-poor {
        color: #e74c3c;
        font-size: 5rem;
        font-weight: bold;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.2);
    }
    
    /* Feedback Boxes */
    .feedback-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        color: #2c3e50 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .feedback-box p, .feedback-box li, .feedback-box strong, .feedback-box h3, 
    .feedback-box h4, .feedback-box ul, .feedback-box ol, .feedback-box table {
        color: #2c3e50 !important;
    }
    
    /* AI Insight Box */
    .ai-insight {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #667eea;
        margin: 1.5rem 0;
        color: #2c3e50 !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.2);
    }
    
    .ai-insight p, .ai-insight h3, .ai-insight li, .ai-insight strong, 
    .ai-insight h4, .ai-insight table, .ai-insight ul, .ai-insight ol {
        color: #2c3e50 !important;
    }
    
    /* Advanced Badge */
    .advanced-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0 0.3rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Pro Feature Box */
    .pro-feature {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        margin: 1rem 0;
        box-shadow: 0 6px 16px rgba(79, 172, 254, 0.3);
    }
    
    .pro-feature h3, .pro-feature h4, .pro-feature p, .pro-feature li, .pro-feature ul {
        color: white !important;
    }
    
    /* Global Text Colors */
    .stMarkdown, .stMarkdown p, .stMarkdown h3, .stMarkdown li, .stMarkdown strong {
        color: #2c3e50 !important;
    }
    
    /* Tables */
    table {
        color: #2c3e50 !important;
        border-collapse: collapse;
        width: 100%;
    }
    
    table th {
        background: #667eea;
        color: white !important;
        padding: 1rem;
        font-weight: bold;
    }
    
    table td {
        color: #2c3e50 !important;
        padding: 0.8rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    table tr:hover {
        background: #f8f9fa;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea15 0%, #764ba215 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
session_defaults = {
    'recording': False,
    'analysis_complete': False,
    'session_data': None,
    'gemini_api_key': "",
    'face_cascade': None,
    'eye_cascade': None,
    'smile_cascade': None,
    'selected_duration': "2 minutes",
    'selected_mic_index': None,
    'benchmark_type': 'ted_speakers'
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================================
# HAAR CASCADE LOADING
# ============================================================================
def load_cascades():
    """Load OpenCV Haar Cascade classifiers for face and feature detection"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        if face_cascade.empty() or eye_cascade.empty():
            return None, None, None
            
        return face_cascade, eye_cascade, smile_cascade
    except Exception as e:
        print(f"Error loading cascades: {e}")
        return None, None, None

if st.session_state.face_cascade is None:
    st.session_state.face_cascade, st.session_state.eye_cascade, st.session_state.smile_cascade = load_cascades()

# ============================================================================
# DATABASE INITIALIZATION & MIGRATION
# ============================================================================
def init_db():
    """Initialize SQLite database with automatic schema migration"""
    conn = sqlite3.connect('presentation_history.db')
    c = conn.cursor()
    
    # Create table with all required columns
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT NOT NULL,
                  duration REAL NOT NULL,
                  wpm REAL NOT NULL,
                  filler_count INTEGER NOT NULL,
                  eye_contact_score REAL NOT NULL,
                  emotion_scores TEXT,
                  pause_count INTEGER,
                  overall_score REAL NOT NULL,
                  transcript TEXT,
                  ai_feedback TEXT,
                  confidence_score REAL DEFAULT 0,
                  clarity_score REAL DEFAULT 0,
                  engagement_score REAL DEFAULT 0,
                  fluency_score REAL DEFAULT 0,
                  vocabulary_richness REAL DEFAULT 0,
                  engagement_prediction REAL DEFAULT 0)''')
    
    # Check for existing columns and add missing ones
    cursor = c.execute('PRAGMA table_info(sessions)')
    existing_columns = [column[1] for column in cursor.fetchall()]
    
    required_columns = {
        'ai_feedback': 'TEXT',
        'confidence_score': 'REAL DEFAULT 0',
        'clarity_score': 'REAL DEFAULT 0',
        'engagement_score': 'REAL DEFAULT 0',
        'fluency_score': 'REAL DEFAULT 0',
        'vocabulary_richness': 'REAL DEFAULT 0',
        'engagement_prediction': 'REAL DEFAULT 0'
    }
    
    for column_name, column_def in required_columns.items():
        if column_name not in existing_columns:
            try:
                c.execute(f'ALTER TABLE sessions ADD COLUMN {column_name} {column_def}')
                print(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    print(f"⚠️ Column addition error for {column_name}: {e}")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

init_db()

# ============================================================================
# PROFESSIONAL BENCHMARKS
# ============================================================================
PROFESSIONAL_BENCHMARKS = {
    'ted_speakers': {
        'name': 'TED Talk Speakers',
        'description': 'World-class presenters with millions of views',
        'wpm': 165,
        'filler_per_min': 0.3,
        'eye_contact': 85,
        'vocabulary_richness': 65,
        'pause_frequency': 4.2
    },
    'business_presenters': {
        'name': 'Business Presenters',
        'description': 'Corporate executives and business leaders',
        'wpm': 145,
        'filler_per_min': 0.8,
        'eye_contact': 78,
        'vocabulary_richness': 55,
        'pause_frequency': 3.5
    },
    'academic_speakers': {
        'name': 'Academic Speakers',
        'description': 'University professors and researchers',
        'wpm': 135,
        'filler_per_min': 1.2,
        'eye_contact': 72,
        'vocabulary_richness': 70,
        'pause_frequency': 3.0
    }
}

# ============================================================================
# SPEECH ANALYSIS FUNCTIONS
# ============================================================================
def analyze_speech_advanced(text):
    """
    Advanced speech analysis with multiple metrics
    Returns: Dictionary with comprehensive speech statistics
    """
    if not text or len(text.strip()) == 0:
        return {
            'word_count': 0,
            'filler_count': 0,
            'pause_count': 0,
            'unique_words': 0,
            'avg_word_length': 0,
            'vocabulary_richness': 0,
            'avg_sentence_length': 0,
            'repetition_count': 0,
            'complex_sentences': 0,
            'sentence_count': 0
        }
    
    # Basic word analysis
    words = text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    vocabulary_richness = (unique_words / word_count * 100) if word_count > 0 else 0
    
    # Filler words detection (expanded list)
    filler_words = [
        'um', 'uh', 'like', 'so', 'actually', 'basically', 'literally', 
        'you know', 'i mean', 'kind of', 'sort of', 'right', 'okay', 
        'well', 'anyway', 'honestly', 'seriously', 'totally'
    ]
    filler_count = sum(text.lower().count(filler) for filler in filler_words)
    
    # Repetition detection (3-word phrases)
    repetitions = {}
    for i in range(len(words) - 2):
        phrase = ' '.join(words[i:i+3])
        if phrase in repetitions:
            repetitions[phrase] += 1
        else:
            repetitions[phrase] = 1
    
    repetition_count = sum(1 for count in repetitions.values() if count > 2)
    
    # Sentence analysis
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 5]
    sentence_count = len(sentences)
    
    if sentence_count > 0:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / sentence_count
    else:
        avg_sentence_length = 0
    
    # Complex sentence detection (using conjunctions)
    complex_conjunctions = [
        'because', 'although', 'however', 'therefore', 'moreover',
        'furthermore', 'nevertheless', 'whereas', 'unless', 'while'
    ]
    complex_sentences = sum(1 for s in sentences if any(conj in s.lower() for conj in complex_conjunctions))
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Pause indicators
    pause_count = text.count('...') + text.count('..') + text.count(' - ') + text.count(',')
    
    return {
        'word_count': word_count,
        'filler_count': filler_count,
        'pause_count': max(0, pause_count),
        'unique_words': unique_words,
        'avg_word_length': round(avg_word_length, 2),
        'vocabulary_richness': round(vocabulary_richness, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'repetition_count': repetition_count,
        'complex_sentences': complex_sentences,
        'sentence_count': sentence_count
    }

# ============================================================================
# FACE & EYE DETECTION
# ============================================================================
def detect_face_and_eyes_advanced(frame):
    """
    Advanced face and eye detection using Haar Cascades
    Returns: (eye_contact, face_detected, face_size, face_x, face_y)
    """
    try:
        if st.session_state.face_cascade is None:
            return False, False, 0, 0, 0
        
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = st.session_state.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        eye_contact = False
        face_detected = False
        face_size = 0
        face_x = 0
        face_y = 0
        
        for (x, y, w, h) in faces:
            face_detected = True
            face_size = w * h
            face_x = x + w // 2
            face_y = y + h // 2
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Extract face region for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes in face region
            if st.session_state.eye_cascade is not None:
                eyes = st.session_state.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.05,
                    minNeighbors=7,
                    minSize=(20, 20),
                    maxSize=(80, 80)
                )
                
                if len(eyes) >= 2:
                    eye_contact = True
                    # Draw eye rectangles
                    for (ex, ey, ew, eh) in eyes[:2]:  # Only mark first 2 eyes
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            break  # Only process first detected face
        
        return eye_contact, face_detected, face_size, face_x, face_y
        
    except Exception as e:
        print(f"Face detection error: {e}")
        return False, False, 0, 0, 0

# ============================================================================
# EMOTION DETECTION
# ============================================================================
def detect_emotions_advanced(frame):
    """
    Emotion detection using smile cascade and facial analysis
    Returns: (emotion, confidence, smiling)
    """
    try:
        if st.session_state.face_cascade is None:
            return 'neutral', 0, False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = st.session_state.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return 'neutral', 0, False
        
        emotion = 'neutral'
        confidence = 0
        smiling = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            # Try to detect smile
            if st.session_state.smile_cascade is not None:
                smiles = st.session_state.smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(25, 25)
                )
                
                if len(smiles) > 0:
                    emotion = 'happy'
                    confidence = min(len(smiles) * 30, 100)
                    smiling = True
                else:
                    # Check for confident expression (eyes visible)
                    if st.session_state.eye_cascade is not None:
                        eyes = st.session_state.eye_cascade.detectMultiScale(
                            roi_gray, scaleFactor=1.1, minNeighbors=5
                        )
                        
                        if len(eyes) >= 2:
                            emotion = 'confident'
                            confidence = 70
                        else:
                            emotion = 'neutral'
                            confidence = 50
            
            break  # Only process first face
        
        return emotion, confidence, smiling
        
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return 'neutral', 0, False

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================
def calculate_advanced_scores(metrics, transcript):
    """
    Calculate advanced presentation scores
    Returns: Dictionary with confidence, clarity, engagement, fluency scores
    """
    # Confidence Score (based on pacing and filler reduction)
    confidence = 100
    confidence -= min(metrics['filler_count'] * 2, 30)
    confidence -= max(0, (120 - metrics['wpm']) * 0.5)
    confidence -= max(0, (metrics['wpm'] - 180) * 0.3)
    confidence_score = max(0, min(100, confidence))
    
    # Clarity Score (based on fillers, pacing, vocabulary)
    clarity = 100
    clarity -= min(metrics['filler_count'] * 2.5, 30)
    clarity -= max(0, (metrics['wpm'] - 180) * 0.3)
    vocab = metrics.get('vocabulary_richness', 0)
    if vocab < 30:
        clarity -= 15
    elif vocab > 60:
        clarity += 5
    clarity_score = max(0, min(100, clarity))
    
    # Engagement Score (eye contact + pacing optimization)
    engagement = metrics['eye_contact_score']
    if 120 <= metrics['wpm'] <= 180:
        engagement += 10
    if metrics.get('vocabulary_richness', 0) > 60:
        engagement += 5
    engagement_score = max(0, min(100, engagement))
    
    # Fluency Score (inverse of filler words)
    fluency = 100 - (metrics['filler_count'] * 3)
    if metrics.get('repetition_count', 0) > 0:
        fluency -= metrics['repetition_count'] * 5
    fluency_score = max(0, min(100, fluency))
    
    # Engagement Prediction (weighted average)
    engagement_prediction = (
        engagement_score * 0.4 +
        confidence_score * 0.3 +
        clarity_score * 0.3
    )
    
    return {
        'confidence_score': round(confidence_score, 1),
        'clarity_score': round(clarity_score, 1),
        'engagement_score': round(engagement_score, 1),
        'fluency_score': round(fluency_score, 1),
        'engagement_prediction': round(engagement_prediction, 1)
    }

def calculate_overall_score_advanced(metrics):
    """
    Calculate comprehensive overall presentation score
    Range: 0-100
    """
    score = 100
    
    # Filler words penalty (max -25 points)
    if metrics['filler_count'] > 15:
        score -= 25
    elif metrics['filler_count'] > 10:
        score -= 20
    elif metrics['filler_count'] > 5:
        score -= 10
    elif metrics['filler_count'] > 3:
        score -= 5
    
    # Eye contact penalty (max -25 points)
    eye_score = metrics['eye_contact_score']
    if eye_score < 40:
        score -= 25
    elif eye_score < 60:
        score -= 15
    elif eye_score < 75:
        score -= 8
    
    # Speaking pace penalty (max -20 points)
    wpm = metrics['wpm']
    if wpm < 100 or wpm > 200:
        score -= 20
    elif wpm < 120 or wpm > 180:
        score -= 10
    
    # Content penalty (max -15 points)
    if metrics['word_count'] < 20:
        score -= 15
    elif metrics['word_count'] < 40:
        score -= 8
    
    # Vocabulary richness bonus/penalty (max ±10 points)
    vocab = metrics.get('vocabulary_richness', 0)
    if vocab > 60:
        score += 10
    elif vocab < 30:
        score -= 10
    
    # Excellence bonus (max +10 points)
    if (140 <= wpm <= 160 and 
        metrics['filler_count'] < 3 and 
        eye_score > 85):
        score += 10
    
    return max(0, min(100, round(score, 1)))

# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================
def compare_with_benchmark(user_metrics, benchmark_type='ted_speakers'):
    """
    Compare user performance with professional benchmarks
    Returns: Dictionary with detailed comparison
    """
    benchmark = PROFESSIONAL_BENCHMARKS[benchmark_type]
    duration_mins = user_metrics['duration'] / 60
    user_filler_per_min = user_metrics['filler_count'] / duration_mins if duration_mins > 0 else 0
    
    comparison = {
        'benchmark_name': benchmark['name'],
        'benchmark_description': benchmark['description'],
        'metrics': {
            'wpm': {
                'your_score': round(user_metrics['wpm'], 1),
                'professional_avg': benchmark['wpm'],
                'difference': round(user_metrics['wpm'] - benchmark['wpm'], 1),
                'difference_pct': round(((user_metrics['wpm'] - benchmark['wpm']) / benchmark['wpm']) * 100, 1),
                'status': '✅ Above' if user_metrics['wpm'] >= benchmark['wpm'] * 0.95 else '⚠️ Below'
            },
            'fillers': {
                'your_score': round(user_filler_per_min, 2),
                'professional_avg': benchmark['filler_per_min'],
                'difference': round(user_filler_per_min - benchmark['filler_per_min'], 2),
                'difference_pct': round(((user_filler_per_min - benchmark['filler_per_min']) / benchmark['filler_per_min']) * 100, 1) if benchmark['filler_per_min'] > 0 else 0,
                'status': '✅ Better' if user_filler_per_min < benchmark['filler_per_min'] * 1.2 else '⚠️ More'
            },
            'eye_contact': {
                'your_score': round(user_metrics['eye_contact_score'], 1),
                'professional_avg': benchmark['eye_contact'],
                'difference': round(user_metrics['eye_contact_score'] - benchmark['eye_contact'], 1),
                'difference_pct': round(((user_metrics['eye_contact_score'] - benchmark['eye_contact']) / benchmark['eye_contact']) * 100, 1),
                'status': '✅ Above' if user_metrics['eye_contact_score'] >= benchmark['eye_contact'] * 0.95 else '⚠️ Below'
            },
            'vocabulary': {
                'your_score': round(user_metrics.get('vocabulary_richness', 0), 1),
                'professional_avg': benchmark['vocabulary_richness'],
                'difference': round(user_metrics.get('vocabulary_richness', 0) - benchmark['vocabulary_richness'], 1),
                'difference_pct': round(((user_metrics.get('vocabulary_richness', 0) - benchmark['vocabulary_richness']) / benchmark['vocabulary_richness']) * 100, 1),
                'status': '✅ Above' if user_metrics.get('vocabulary_richness', 0) >= benchmark['vocabulary_richness'] * 0.9 else '⚠️ Below'
            }
        }
    }
    
    return comparison

# ============================================================================
# AI FEEDBACK GENERATION
# ============================================================================
def get_gemini_feedback(transcript, metrics):
    """
    Generate professional AI feedback based on presentation metrics
    Returns: Formatted feedback string
    """
    wpm = metrics['wpm']
    fillers = metrics['filler_count']
    eye_contact = metrics['eye_contact_score']
    overall = metrics['overall_score']
    vocab = metrics.get('vocabulary_richness', 0)
    
    # Pre-calculate conditional strings to avoid backslashes in f-strings
    if overall >= 85:
        assessment = "🌟 Outstanding performance! You're presenting at a professional level."
    elif overall >= 75:
        assessment = "⭐ Excellent work! You're on track to professional-level presentation."
    elif overall >= 65:
        assessment = "👍 Good foundation with clear improvement opportunities."
    else:
        assessment = "📈 Developing skills - keep practicing consistently!"
    
    if 140 <= wpm <= 170:
        pace_feedback = "excellent - right in the optimal range"
    elif 120 <= wpm <= 180:
        pace_feedback = "within acceptable range"
    elif wpm < 120:
        pace_feedback = "below optimal - try to increase energy"
    else:
        pace_feedback = "too fast - slow down for clarity"
    
    if fillers < 3:
        filler_feedback = "Outstanding! Professional speakers aim for fewer than 3."
    elif fillers < 5:
        filler_feedback = "Very good! Keep working to reduce further."
    elif fillers < 10:
        filler_feedback = "Work on eliminating these to sound more confident and polished."
    else:
        filler_feedback = "Focus significantly on reducing filler words - this is a key area for improvement."
    
    if eye_contact > 85:
        eye_feedback = "Exceptional! You maintained strong visual connection with your audience."
    elif eye_contact > 75:
        eye_feedback = "Very good eye contact maintained."
    elif eye_contact > 65:
        eye_feedback = "Good foundation, but aim for 80%+ eye contact."
    else:
        eye_feedback = "This is a critical area needing significant improvement."
    
    if vocab > 60:
        vocab_feedback = "Excellent variety in word choice!"
    elif vocab > 50:
        vocab_feedback = "Good vocabulary usage."
    elif vocab > 40:
        vocab_feedback = "Consider expanding vocabulary for more engaging content."
    else:
        vocab_feedback = "Work on using more diverse vocabulary."
    
    # Build strengths list
    strengths = []
    if 140 <= wpm <= 170:
        strengths.append("Outstanding pace control")
    if fillers < 5:
        strengths.append("Minimal filler words")
    if eye_contact > 80:
        strengths.append("Strong eye contact and engagement")
    if vocab > 60:
        strengths.append("Rich vocabulary usage")
    if overall >= 80:
        strengths.append("Clear and confident delivery")
    
    if not strengths:
        strengths = ["Taking initiative to improve through practice"]
    
    strengths_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(strengths[:5])])
    
    # Build improvements list
    improvements = []
    if fillers >= 5:
        improvements.append(f"Reduce filler words from {fillers} to <3")
    if eye_contact < 75:
        improvements.append(f"Improve eye contact to 80%+ (current: {eye_contact:.0f}%)")
    if wpm < 140 or wpm > 170:
        improvements.append("Adjust pace to 140-170 WPM")
    if vocab < 50:
        improvements.append("Expand vocabulary variety")
    
    if not improvements or overall >= 85:
        improvements = [
            "Maintain your excellent performance",
            "Add more vocal variety and emphasis", 
            "Incorporate strategic pauses for effect"
        ]
    
    improvements_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(improvements[:3])])
    
    # Benchmark comparison
    if overall >= 85:
        benchmark_text = "exceeding"
    elif overall >= 75:
        benchmark_text = "meeting"
    elif overall >= 65:
        benchmark_text = "approaching"
    else:
        benchmark_text = "working toward"
    
    feedback = f"""### 🎯 Professional Analysis Report

**Overall Assessment:**
{assessment}

**Score Breakdown:**
- Overall Score: {overall:.1f}/100
- Confidence: {metrics.get('confidence_score', 0):.1f}%
- Clarity: {metrics.get('clarity_score', 0):.1f}%
- Engagement: {metrics.get('engagement_score', 0):.1f}%
- Fluency: {metrics.get('fluency_score', 0):.1f}%

**🎤 Delivery & Pace Analysis:**
Your speaking pace of {wpm:.0f} WPM is {pace_feedback}. Professional speakers typically maintain 140-170 WPM for maximum audience comprehension and engagement.

You used {fillers} filler words during your presentation. {filler_feedback}

**👁️ Body Language & Engagement:**
Eye contact score: {eye_contact:.0f}%. {eye_feedback}

**📚 Content & Vocabulary:**
Vocabulary richness: {vocab:.1f}%. {vocab_feedback}

**✅ Key Strengths:**
{strengths_text}

**🎯 Priority Improvements:**
{improvements_text}

**💡 Actionable Next Steps:**
1. Record yourself and count filler words - awareness is the first step
2. Practice the "pause instead of um" technique
3. Use the "triangle technique" for eye contact (left side, center, right side)
4. Read aloud at your target pace to build muscle memory
5. Present to friends/family for real-world practice

**🏆 Professional Benchmark:**
You're {benchmark_text} professional speaker standards. Keep up the excellent work!
"""
    
    return feedback

# ============================================================================
# AUDIO RECORDING
# ============================================================================
def record_audio_advanced(duration, result_queue, mic_index=None):
    """
    Record and transcribe audio using Google Speech Recognition
    Args:
        duration: Recording duration in seconds
        result_queue: Queue to store transcribed text
        mic_index: Microphone device index (None for default)
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 400
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    
    full_text = ""
    chunks_processed = 0
    
    try:
        # Initialize microphone
        if mic_index is not None:
            mic = sr.Microphone(device_index=mic_index)
        else:
            mic = sr.Microphone()
        
        with mic as source:
            print("🎤 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            
            start_time = time.time()
            
            while time.time() - start_time < duration:
                try:
                    remaining = duration - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    
                    # Listen for audio chunk
                    chunk_duration = min(5, remaining)
                    audio = recognizer.listen(
                        source,
                        timeout=chunk_duration,
                        phrase_time_limit=chunk_duration
                    )
                    
                    # Transcribe chunk
                    try:
                        text = recognizer.recognize_google(audio, language='en-US')
                        if text:
                            full_text += " " + text
                            chunks_processed += 1
                            print(f"✅ Chunk {chunks_processed}: {text[:50]}...")
                    except sr.UnknownValueError:
                        print(f"⚠️ Could not understand chunk {chunks_processed + 1}")
                        continue
                    except sr.RequestError as e:
                        print(f"❌ API error: {e}")
                        continue
                        
                except sr.WaitTimeoutError:
                    print("⚠️ Listening timeout")
                    continue
                except Exception as e:
                    print(f"❌ Listen error: {e}")
                    continue
                    
    except Exception as e:
        print(f"❌ Recording error: {e}")
        result_queue.put(f"ERROR: {str(e)}")
        return
    
    final_text = full_text.strip()
    print(f"📝 Total transcribed: {len(final_text)} characters, {chunks_processed} chunks")
    result_queue.put(final_text if final_text else "")

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_metrics(metrics, history_df):
    """
    Create comprehensive visualization dashboard
    Returns: Matplotlib figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # Color scheme
    primary_color = '#667eea'
    secondary_color = '#764ba2'
    
    # 1. Overall Score Gauge
    ax1 = fig.add_subplot(gs[0, :2])
    score = metrics['overall_score']
    colors_map = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2ecc71']
    color_idx = min(int(score / 20), 4)
    
    ax1.barh([0], [score], color=colors_map[color_idx], height=0.5, alpha=0.9)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_yticks([])
    ax1.text(score + 2, 0, f'{score:.1f}', va='center', fontsize=22, fontweight='bold')
    ax1.set_xlabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Overall Presentation Score', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add benchmark line
    ax1.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Professional Standard')
    ax1.legend()
    
    # 2. Skills Breakdown
    ax2 = fig.add_subplot(gs[0, 2])
    skills = ['Confidence', 'Clarity', 'Engagement', 'Fluency']
    scores = [
        metrics.get('confidence_score', 70),
        metrics.get('clarity_score', 70),
        metrics.get('engagement_score', 70),
        metrics.get('fluency_score', 70)
    ]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    bars = ax2.barh(skills, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Skills Breakdown', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax2.text(score + 2, bar.get_y() + bar.get_height()/2, 
                f'{score:.0f}', va='center', fontweight='bold', fontsize=10)
    
    # 3. WPM Analysis with Optimal Range
    ax3 = fig.add_subplot(gs[1, 0])
    wpm = metrics['wpm']
    optimal_range = [120, 180]
    
    ax3.axhspan(optimal_range[0], optimal_range[1], alpha=0.3, color='green', label='Optimal Range')
    ax3.axhspan(140, 170, alpha=0.2, color='darkgreen', label='Ideal Range')
    ax3.barh(['Your WPM'], [wpm], color=primary_color, height=0.4, alpha=0.9)
    ax3.set_xlim(0, max(250, wpm + 20))
    ax3.set_xlabel('Words Per Minute', fontsize=11, fontweight='bold')
    ax3.set_title('Speaking Pace Analysis', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='lower right', fontsize=8)
    ax3.text(wpm + 5, 0, f'{wpm:.0f}', va='center', fontsize=15, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Filler Words Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    fillers = metrics['filler_count']
    clean_words = max(metrics['word_count'] - fillers, 0)
    
    sizes = [fillers, clean_words]
    colors_pie = ['#e74c3c', '#2ecc71']
    explode = (0.1, 0)  # Explode filler slice
    
    ax4.pie(sizes, labels=['Filler Words', 'Clean Speech'], autopct='%1.1f%%',
            colors=colors_pie, startangle=90, explode=explode,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax4.set_title(f'Filler Words: {fillers}', fontsize=13, fontweight='bold', pad=10)
    
    # 5. Eye Contact Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    eye_score = metrics['eye_contact_score']
    away_score = 100 - eye_score
    
    sizes_eye = [eye_score, away_score]
    colors_eye = ['#667eea', '#e0e0e0']
    explode_eye = (0.05, 0)
    
    ax5.pie(sizes_eye, labels=['Eye Contact', 'Looking Away'], autopct='%1.1f%%',
            colors=colors_eye, startangle=90, explode=explode_eye,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Eye Contact Distribution', fontsize=13, fontweight='bold', pad=10)
    
    # 6. Vocabulary Richness
    ax6 = fig.add_subplot(gs[2, 0])
    vocab = metrics.get('vocabulary_richness', 0)
    
    ax6.bar(['Vocabulary\nRichness'], [vocab], color='#f093fb', alpha=0.8, width=0.5)
    ax6.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Professional Level')
    ax6.set_ylim(0, 100)
    ax6.set_ylabel('Percentage', fontsize=11, fontweight='bold')
    ax6.set_title('Vocabulary Richness', fontsize=13, fontweight='bold', pad=10)
    ax6.text(0, vocab + 3, f'{vocab:.1f}%', ha='center', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Content Metrics
    ax7 = fig.add_subplot(gs[2, 1])
    metrics_names = ['Word\nCount', 'Unique\nWords', 'Sentences']
    metrics_values = [
        min(metrics['word_count'], 500),  # Cap for visualization
        metrics.get('unique_words', 0),
        metrics.get('sentence_count', 0) * 10  # Scale for visibility
    ]
    
    bars = ax7.bar(metrics_names, metrics_values, color=['#4facfe', '#00f2fe', '#764ba2'], alpha=0.8)
    ax7.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax7.set_title('Content Metrics', fontsize=13, fontweight='bold', pad=10)
    ax7.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [metrics['word_count'], metrics.get('unique_words', 0), metrics.get('sentence_count', 0)]):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 8. Engagement Prediction
    ax8 = fig.add_subplot(gs[2, 2])
    engagement_pred = metrics.get('engagement_prediction', 0)
    
    # Create gauge-style visualization
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    
    # Color segments
    colors_gauge = ['#e74c3c', '#f39c12', '#2ecc71']
    segments = [33.3, 66.6, 100]
    
    for i, (start, end, color) in enumerate(zip([0, 33.3, 66.6], segments, colors_gauge)):
        mask = (theta >= np.pi * start/100) & (theta <= np.pi * end/100)
        ax8.fill_between(theta[mask], 0, r[mask], color=color, alpha=0.3)
    
    # Needle
    needle_angle = np.pi * (100 - engagement_pred) / 100
    ax8.plot([needle_angle, needle_angle], [0, 1], 'k-', linewidth=3)
    ax8.plot(needle_angle, 1, 'ko', markersize=10)
    
    ax8.set_ylim(0, 1.2)
    ax8.set_xlim(0, np.pi)
    ax8.axis('off')
    ax8.text(np.pi/2, 0.5, f'{engagement_pred:.0f}%', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax8.set_title('Predicted Audience\nEngagement', fontsize=13, fontweight='bold', pad=10)
    
    # 9. Historical Performance
    ax9 = fig.add_subplot(gs[3, :])
    
    if history_df is not None and len(history_df) > 1:
        recent = history_df.head(15).iloc[::-1]
        sessions = range(1, len(recent) + 1)
        
        # Plot multiple metrics
        ax9.plot(sessions, recent['overall_score'], 
                marker='o', linewidth=3, markersize=10, 
                color=primary_color, label='Overall Score', alpha=0.8)
        
        ax9.plot(sessions, recent['eye_contact_score'], 
                marker='s', linewidth=2, markersize=8, 
                color=secondary_color, label='Eye Contact', alpha=0.7)
        
        # Calculate and plot trend line
        if len(sessions) > 2:
            z = np.polyfit(list(sessions), recent['overall_score'].values, 1)
            p = np.poly1d(z)
            ax9.plot(sessions, p(sessions), "--", 
                    color='gray', linewidth=2, alpha=0.6, label='Trend')
        
        ax9.set_xlabel('Session Number', fontsize=13, fontweight='bold')
        ax9.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax9.set_title('Performance Progression Over Time', fontsize=15, fontweight='bold', pad=15)
        ax9.legend(loc='best', fontsize=11, framealpha=0.9)
        ax9.grid(True, alpha=0.3, linestyle='--')
        ax9.set_ylim(0, 105)
        
        # Highlight improvement
        if len(recent) >= 2:
            improvement = recent['overall_score'].iloc[-1] - recent['overall_score'].iloc[0]
            color = 'green' if improvement > 0 else 'red'
            ax9.text(0.02, 0.98, f'Overall Change: {improvement:+.1f} points',
                    transform=ax9.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    else:
        ax9.text(0.5, 0.5, 'Complete more sessions to see your progress!\n\nConsistency is key to improvement.',
                ha='center', va='center', fontsize=14, transform=ax9.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
        ax9.set_title('Performance Progression', fontsize=15, fontweight='bold', pad=15)
        ax9.axis('off')
    
    plt.suptitle('📊 Comprehensive Presentation Analytics Dashboard', 
                 fontsize=20, fontweight='bold', y=0.998)
    
    return fig

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def save_session(data):
    """Save presentation session to database"""
    conn = sqlite3.connect('presentation_history.db')
    c = conn.cursor()
    
    try:
        c.execute('''INSERT INTO sessions 
                     (date, duration, wpm, filler_count, eye_contact_score, emotion_scores,
                      pause_count, overall_score, transcript, ai_feedback, confidence_score,
                      clarity_score, engagement_score, fluency_score, vocabulary_richness,
                      engagement_prediction)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                   data.get('duration', 0),
                   data.get('wpm', 0),
                   data.get('filler_count', 0),
                   data.get('eye_contact_score', 0),
                   json.dumps(data.get('emotion_scores', {})),
                   data.get('pause_count', 0),
                   data.get('overall_score', 0),
                   data.get('transcript', ''),
                   data.get('ai_feedback', ''),
                   data.get('confidence_score', 0),
                   data.get('clarity_score', 0),
                   data.get('engagement_score', 0),
                   data.get('fluency_score', 0),
                   data.get('vocabulary_richness', 0),
                   data.get('engagement_prediction', 0)))
        
        conn.commit()
        print("✅ Session saved successfully")
    except Exception as e:
        print(f"❌ Error saving session: {e}")
    finally:
        conn.close()

def get_session_history():
    """Retrieve all session history from database"""
    try:
        conn = sqlite3.connect('presentation_history.db')
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY date DESC", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading history: {e}")
        return pd.DataFrame()

def test_microphone():
    """Test microphone availability and functionality"""
    recognizer = sr.Recognizer()
    try:
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            return False, "No microphones found on this system", []
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            return True, f"Microphone working! Found {len(mic_list)} device(s)", mic_list
    except Exception as e:
        return False, f"Microphone error: {str(e)}", []

# ============================================================================
# MAIN APPLICATION UI
# ============================================================================

st.markdown('<h1 class="main-header">🎯 AI Presentation Coach Pro</h1>', unsafe_allow_html=True)
st.markdown('''
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.4rem; color: #667eea; font-weight: bold; margin: 0.5rem 0;">
        Advanced Multimodal Analysis with Professional Benchmarking
    </p>
    <span class="advanced-badge">ULTIMATE EDITION v4.0</span>
    <span class="advanced-badge">🤖 AI-POWERED</span>
    <span class="advanced-badge">📊 PROFESSIONAL METRICS</span>
</div>
''', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/microphone.png", width=80)
    st.title("🎛️ Control Panel")
    
    page = st.radio("Navigation", [
        "🎤 Live Practice",
        "🧪 System Check",
        "📊 Analytics Dashboard",
        "⚙️ Settings",
        "ℹ️ About Project"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    
    history_df = get_session_history()
    if len(history_df) > 0:
        st.metric("Total Sessions", len(history_df))
        st.metric("Average Score", f"{history_df['overall_score'].mean():.1f}")
        st.metric("Best Score", f"{history_df['overall_score'].max():.1f}")
        
        # Show improvement trend
        if len(history_df) >= 2:
            recent_avg = history_df.head(3)['overall_score'].mean()
            older_avg = history_df.tail(3)['overall_score'].mean()
            improvement = recent_avg - older_avg
            
            st.metric("Recent Improvement", 
                     f"{improvement:+.1f}", 
                     delta=f"{improvement:+.1f} points")
    else:
        st.info("📝 No sessions yet\n\nStart practicing to see your stats!")
    
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.caption("**Course:** DES646")
    st.caption("**Version:** 4.0 Ultimate")
    st.caption("**Year:** 2025")
    
    st.markdown("---")
    st.caption("**Team Members:**")
    st.caption("• Surya")
    st.caption("• Pushpendra")
    st.caption("• Meet")
    st.caption("• Vasundhara")
    st.caption("• Ayush")

# ============================================================================
# PAGE 1: LIVE PRACTICE
# ============================================================================
if "🎤 Live Practice" in page:
    st.header("🎥 Live Practice Session")
    
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
                <li><strong>Position:</strong> Face camera directly at eye level</li>
                <li><strong>Delivery:</strong> Speak at 140-160 WPM (conversational pace)</li>
                <li><strong>Content:</strong> Present naturally as if to a real audience</li>
                <li><strong>Practice:</strong> Treat this like an actual presentation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            selected_duration = st.selectbox(
                "🕒 Select Practice Duration:",
                options=list(duration_options.keys()),
                index=2  # Default to 2 minutes
            )
            
            st.session_state.selected_duration = selected_duration
            duration = duration_options[selected_duration]
            
            # Benchmark selection
            benchmark_options = {
                "TED Talk Speakers": "ted_speakers",
                "Business Presenters": "business_presenters",
                "Academic Speakers": "academic_speakers"
            }
            
            selected_benchmark = st.selectbox(
                "🏆 Compare Performance With:",
                options=list(benchmark_options.keys()),
                index=0
            )
            
            st.session_state.benchmark_type = benchmark_options[selected_benchmark]
            
            # Microphone selection
            try:
                mic_list = sr.Microphone.list_microphone_names()
                if len(mic_list) > 1:
                    mic_choice = st.selectbox(
                        "🎤 Select Microphone:",
                        options=["Default (Recommended)"] + 
                               [f"Device {i}: {name[:40]}" for i, name in enumerate(mic_list)]
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
                success, message, _ = test_microphone()
                if success:
                    st.session_state.recording = True
                    st.rerun()
                else:
                    st.error(f"❌ {message}\n\nPlease check the System Check page for troubleshooting.")
        
        with col2:
            st.markdown("""
            <div class="pro-feature">
            <h3 style="color: white !important; margin-bottom: 1rem;">🎯 Professional Targets</h3>
            <ul style="color: white !important; list-style: none; padding-left: 0;">
                <li style="color: white !important; margin-bottom: 0.6rem;">
                    <strong style="color: white !important;">Speaking Pace:</strong><br>140-170 WPM
                </li>
                <li style="color: white !important; margin-bottom: 0.6rem;">
                    <strong style="color: white !important;">Filler Words:</strong><br>< 3 total
                </li>
                <li style="color: white !important; margin-bottom: 0.6rem;">
                    <strong style="color: white !important;">Eye Contact:</strong><br>> 80%
                </li>
                <li style="color: white !important; margin-bottom: 0.6rem;">
                    <strong style="color: white !important;">Vocabulary:</strong><br>> 60%
                </li>
                <li style="color: white !important;">
                    <strong style="color: white !important;">Overall Score:</strong><br>85+
                </li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; border: 2px solid #667eea;">
            <h3 style="color: #2c3e50 !important; margin-bottom: 1rem;">💡 Pro Tips</h3>
            <ul style="color: #2c3e50 !important; font-size: 0.9rem;">
                <li style="margin-bottom: 0.5rem;">Take deep breaths before starting</li>
                <li style="margin-bottom: 0.5rem;">Smile naturally - it improves tone</li>
                <li style="margin-bottom: 0.5rem;">Use hand gestures for emphasis</li>
                <li style="margin-bottom: 0.5rem;">Pause strategically for effect</li>
                <li style="margin-bottom: 0.5rem;">Be authentic and passionate</li>
                <li style="margin-bottom: 0.5rem;">Imagine your ideal audience</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.recording:
        # RECORDING MODE - Full implementation continues...
        # (This section contains the complete recording logic - continuing from previous code)
        
        selected_duration = st.session_state.selected_duration
        duration = duration_options[selected_duration]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
             padding: 2.5rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
             box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4);">
            <h2 style="color: white; margin: 0; font-size: 2.5rem;">🔴 RECORDING IN PROGRESS</h2>
            <p style="color: white; margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.9;">
                {selected_duration.split(' - ')[0]}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_display = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check camera permissions and try again.")
            st.session_state.recording = False
            st.stop()
        
        # Start audio recording in separate thread
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(
            target=record_audio_advanced,
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
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Detect face and eyes
            eye_contact, face_detected, face_size, face_x, face_y = detect_face_and_eyes_advanced(frame)
            
            # Detect emotions
            emotion, emotion_conf, smiling = detect_emotions_advanced(frame)
            
            if face_detected:
                total_frames += 1
                face_sizes.append(face_size)
                
                if eye_contact:
                    eye_contact_frames += 1
                
                emotion_list.append(emotion)
                
                # Display status on frame
                cv2.putText(frame, "Face: DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if eye_contact:
                    cv2.putText(frame, "Eye Contact: EXCELLENT", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Eye Contact: IMPROVE", (10, 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                cv2.putText(frame, f"Emotion: {emotion.upper()}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Face: NOT DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Please face the camera", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Time remaining
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            cv2.putText(frame, f"Time Remaining: {int(remaining)}s", 
                       (frame.shape[1] - 300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update progress
            progress = min(elapsed / duration, 1.0)
            progress_bar.progress(progress)
            status_text.markdown(f"**⏱️ Recording:** {int(elapsed)}s / {duration}s")
            
            # Display live metrics
            if total_frames > 0:
                current_eye_contact = (eye_contact_frames / total_frames) * 100
                emotion_counts = Counter(emotion_list)
                dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_list else 'neutral'
                
                metrics_display.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                     padding: 1rem; border-radius: 10px; border: 2px solid #667eea;">
                    <strong style="color: #2c3e50;">📊 Live Metrics:</strong><br>
                    <span style="color: #2c3e50;">Eye Contact: {current_eye_contact:.0f}% | 
                    Frames: {total_frames} | 
                    Emotion: {dominant_emotion.title()}</span>
                </div>
                """, unsafe_allow_html=True)
            
            frame_count += 1
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        st.session_state.recording = False
        
        status_text.markdown("**⏳ Processing audio transcription... Please wait...**")
        audio_thread.join(timeout=15)
        
        # Get transcribed text
        try:
            transcribed_text = audio_queue.get_nowait()
            if isinstance(transcribed_text, str) and transcribed_text.startswith("ERROR:"):
                st.warning(f"⚠️ {transcribed_text}")
                transcribed_text = ""
        except:
            transcribed_text = ""
        
        # Analyze speech
        speech_metrics = analyze_speech_advanced(transcribed_text)
        
        # Calculate metrics
        wpm = (speech_metrics['word_count'] / duration) * 60 if duration > 0 else 0
        eye_contact_score = (eye_contact_frames / total_frames) * 100 if total_frames > 0 else 0
        
        emotion_counter = Counter(emotion_list)
        emotion_scores = dict(emotion_counter)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
        emotion_variety = len(set(emotion_list))
        
        # Compile all metrics
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
            'unique_words': speech_metrics['unique_words'],
            'vocabulary_richness': speech_metrics['vocabulary_richness'],
            'avg_sentence_length': speech_metrics['avg_sentence_length'],
            'repetition_count': speech_metrics['repetition_count'],
            'complex_sentences': speech_metrics['complex_sentences'],
            'sentence_count': speech_metrics['sentence_count'],
            'overall_score': 0  # Will be calculated
        }
        
        # Calculate advanced scores
        advanced_scores = calculate_advanced_scores(metrics, transcribed_text)
        metrics.update(advanced_scores)
        
        # Calculate overall score
        metrics['overall_score'] = calculate_overall_score_advanced(metrics)
        
        # Generate AI feedback
        status_text.markdown("**🤖 Generating professional AI feedback...**")
        ai_feedback = get_gemini_feedback(transcribed_text, metrics)
        metrics['ai_feedback'] = ai_feedback
        
        # Save to database
        save_session(metrics)
        
        st.session_state.session_data = metrics
        st.session_state.analysis_complete = True
        
        st.success("✅ Analysis Complete! Preparing your detailed report...")
        time.sleep(1.5)
        st.rerun()
    
    elif st.session_state.analysis_complete:
        # RESULTS DISPLAY MODE
        metrics = st.session_state.session_data
        benchmark_comparison = compare_with_benchmark(metrics, st.session_state.benchmark_type)
        
        # Score Display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            score = metrics['overall_score']
            
            if score >= 90:
                score_class, emoji, level, message = "score-excellent", "🌟", "EXCEPTIONAL", "World-class performance! You're ready for any stage!"
            elif score >= 85:
                score_class, emoji, level, message = "score-excellent", "⭐", "OUTSTANDING", "Excellent! You're presenting at a professional level!"
            elif score >= 75:
                score_class, emoji, level, message = "score-excellent", "💫", "EXCELLENT", "Great work! Just a few tweaks away from perfection!"
            elif score >= 65:
                score_class, emoji, level, message = "score-good", "👍", "GOOD", "Solid performance with clear room for growth!"
            elif score >= 50:
                score_class, emoji, level, message = "score-good", "📈", "DEVELOPING", "You're on the right track. Keep practicing!"
            else:
                score_class, emoji, level, message = "score-poor", "💪", "NEEDS WORK", "Focus on the feedback. Improvement is coming!"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; 
                 background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                 border-radius: 25px; border: 4px solid #667eea;
                 box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
                <div class="{score_class}">{emoji} {score:.1f}</div>
                <h2 style="margin: 1rem 0 0 0; color: #667eea; font-size: 2rem;">{level}</h2>
                <p style="font-size: 1.2rem; color: #666; margin-top: 1rem; font-weight: 500;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            wpm = metrics['wpm']
            wpm_status = "🟢" if 140 <= wpm <= 170 else "🟡" if 120 <= wpm <= 180 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                 box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
                <h3 style="color: white; margin: 0; font-size: 2rem;">{wpm_status} {wpm:.0f}</h3>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">Words/Min</p>
                <p style="margin: 0.3rem 0 0 0; color: white; font-size: 0.75rem; opacity: 0.8;">
                    Target: 140-170
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fillers = metrics['filler_count']
            filler_status = "🟢" if fillers < 3 else "🟡" if fillers < 5 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                 box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);">
                <h3 style="color: white; margin: 0; font-size: 2rem;">{filler_status} {fillers}</h3>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">Filler Words</p>
                <p style="margin: 0.3rem 0 0 0; color: white; font-size: 0.75rem; opacity: 0.8;">
                    Target: < 3
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            eye_contact = metrics['eye_contact_score']
            eye_status = "🟢" if eye_contact > 80 else "🟡" if eye_contact > 65 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                 box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);">
                <h3 style="color: white; margin: 0; font-size: 2rem;">{eye_status} {eye_contact:.0f}%</h3>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">Eye Contact</p>
                <p style="margin: 0.3rem 0 0 0; color: white; font-size: 0.75rem; opacity: 0.8;">
                    Target: > 80%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vocab = metrics.get('vocabulary_richness', 0)
            vocab_status = "🟢" if vocab > 60 else "🟡" if vocab > 45 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                 box-shadow: 0 4px 12px rgba(250, 112, 154, 0.3);">
                <h3 style="color: white; margin: 0; font-size: 2rem;">{vocab_status} {vocab:.0f}%</h3>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">Vocabulary</p>
                <p style="margin: 0.3rem 0 0 0; color: white; font-size: 0.75rem; opacity: 0.8;">
                    Target: > 60%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            engagement_pred = metrics.get('engagement_prediction', 0)
            engage_status = "🟢" if engagement_pred > 75 else "🟡" if engagement_pred > 60 else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                 box-shadow: 0 4px 12px rgba(168, 237, 234, 0.3);">
                <h3 style="color: white; margin: 0; font-size: 2rem;">{engage_status} {engagement_pred:.0f}%</h3>
                <p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">Engagement</p>
                <p style="margin: 0.3rem 0 0 0; color: white; font-size: 0.75rem; opacity: 0.8;">
                    Predicted
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Transcript Section
        if metrics.get('transcript') and len(metrics['transcript']) > 10:
            with st.expander("📝 Full Transcript", expanded=False):
                st.markdown(f"""
                <div class="feedback-box">
                <p style="font-size: 1.1rem; line-height: 1.8; color: #2c3e50 !important;">
                    {metrics['transcript']}
                </p>
                <hr>
                <p style="color: #2c3e50 !important; margin-top: 1rem;">
                    <strong>Word Count:</strong> {metrics['word_count']} | 
                    <strong>Unique Words:</strong> {metrics.get('unique_words', 0)} | 
                    <strong>Sentences:</strong> {metrics.get('sentence_count', 0)}
                </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("""
            ⚠️ **Limited or No Speech Detected**
            
            **Possible Issues:**
            - Microphone was muted or not properly connected
            - Background noise interfered with recording
            - Speaking volume was too low
            - Internet connection issue affecting speech recognition
            
            **Solutions:**
            1. Check microphone settings in System Check page
            2. Test your microphone before starting
            3. Speak loudly and clearly
            4. Ensure stable internet connection
            5. Reduce background noise
            """)
        
        st.markdown("---")
        
        # Visualization
        st.markdown("### 📊 Comprehensive Analytics Dashboard")
        fig = plot_metrics(metrics, get_session_history())
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Benchmark Comparison
        st.markdown("### 🏆 Professional Benchmark Comparison")
        
        st.markdown(f"""
        <div class="ai-insight">
        <h4 style="color: #2c3e50 !important; margin-bottom: 1rem;">
            Comparing with {benchmark_comparison['benchmark_name']}
        </h4>
        <p style="color: #666; font-style: italic; margin-bottom: 1.5rem;">
            {benchmark_comparison['benchmark_description']}
        </p>
        
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="border-bottom: 3px solid #667eea;">
                    <th style="padding: 1rem; text-align: left; color: #2c3e50 !important; font-size: 1.1rem;">
                        Metric
                    </th>
                    <th style="padding: 1rem; text-align: center; color: #2c3e50 !important; font-size: 1.1rem;">
                        Your Score
                    </th>
                    <th style="padding: 1rem; text-align: center; color: #2c3e50 !important; font-size: 1.1rem;">
                        Professional Avg
                    </th>
                    <th style="padding: 1rem; text-align: center; color: #2c3e50 !important; font-size: 1.1rem;">
                        Difference
                    </th>
                    <th style="padding: 1rem; text-align: center; color: #2c3e50 !important; font-size: 1.1rem;">
                        Status
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 0.8rem; color: #2c3e50 !important; font-weight: 500;">
                        Speaking Pace (WPM)
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['wpm']['your_score']:.1f}
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['wpm']['professional_avg']}
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['wpm']['difference']:+.1f} ({benchmark_comparison['metrics']['wpm']['difference_pct']:+.1f}%)
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['wpm']['status']}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 0.8rem; color: #2c3e50 !important; font-weight: 500;">
                        Fillers per Minute
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['fillers']['your_score']:.2f}
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['fillers']['professional_avg']:.2f}
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['fillers']['difference']:+.2f} ({benchmark_comparison['metrics']['fillers']['difference_pct']:+.1f}%)
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['fillers']['status']}
                    </td>
                </tr>
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 0.8rem; color: #2c3e50 !important; font-weight: 500;">
                        Eye Contact
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['eye_contact']['your_score']:.1f}%
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['eye_contact']['professional_avg']}%
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['eye_contact']['difference']:+.1f}% ({benchmark_comparison['metrics']['eye_contact']['difference_pct']:+.1f}%)
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['eye_contact']['status']}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 0.8rem; color: #2c3e50 !important; font-weight: 500;">
                        Vocabulary Richness
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['vocabulary']['your_score']:.1f}%
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['vocabulary']['professional_avg']}%
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important;">
                        {benchmark_comparison['metrics']['vocabulary']['difference']:+.1f}% ({benchmark_comparison['metrics']['vocabulary']['difference_pct']:+.1f}%)
                    </td>
                    <td style="padding: 0.8rem; text-align: center; color: #2c3e50 !important; font-weight: bold;">
                        {benchmark_comparison['metrics']['vocabulary']['status']}
                    </td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # AI Feedback
        st.markdown("### 🤖 Professional AI-Powered Feedback")
        st.markdown(f"""
        <div class="ai-insight">
        {metrics.get('ai_feedback', 'No AI feedback available. Consider adding a Gemini API key in Settings.')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Practice Again", use_container_width=True, type="primary"):
                st.session_state.analysis_complete = False
                st.session_state.session_data = None
                st.rerun()
        
        with col2:
            if st.button("📊 View Dashboard", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.session_data = None
                st.rerun()
        
        with col3:
            # Generate comprehensive report
            report_data = f"""
╔═══════════════════════════════════════════════════════════╗
║        PRESENTATION ANALYSIS REPORT - DETAILED            ║
╚═══════════════════════════════════════════════════════════╝

Generated: {metrics['date']}
Session Duration: {metrics['duration']} seconds

═══════════════════════════════════════════════════════════
                    OVERALL PERFORMANCE
═══════════════════════════════════════════════════════════
Overall Score: {metrics['overall_score']:.1f}/100
Performance Level: {level}

═══════════════════════════════════════════════════════════
                    CORE METRICS
═══════════════════════════════════════════════════════════
Speaking Pace: {metrics['wpm']:.1f} WPM (Target: 140-170)
Filler Words: {metrics['filler_count']} (Target: < 3)
Eye Contact: {metrics['eye_contact_score']:.1f}% (Target: > 80%)
Vocabulary Richness: {metrics.get('vocabulary_richness', 0):.1f}% (Target: > 60%)

═══════════════════════════════════════════════════════════
                    SKILL SCORES
═══════════════════════════════════════════════════════════
Confidence: {metrics.get('confidence_score', 0):.1f}%
Clarity: {metrics.get('clarity_score', 0):.1f}%
Engagement: {metrics.get('engagement_score', 0):.1f}%
Fluency: {metrics.get('fluency_score', 0):.1f}%

═══════════════════════════════════════════════════════════
                    CONTENT ANALYSIS
═══════════════════════════════════════════════════════════
Total Words: {metrics['word_count']}
Unique Words: {metrics.get('unique_words', 0)}
Sentences: {metrics.get('sentence_count', 0)}
Avg Sentence Length: {metrics.get('avg_sentence_length', 0):.1f} words
Complex Sentences: {metrics.get('complex_sentences', 0)}
Repetitions: {metrics.get('repetition_count', 0)}

═══════════════════════════════════════════════════════════
                    BENCHMARK COMPARISON
═══════════════════════════════════════════════════════════
Benchmark: {benchmark_comparison['benchmark_name']}

WPM: {benchmark_comparison['metrics']['wpm']['your_score']:.1f} vs {benchmark_comparison['metrics']['wpm']['professional_avg']} 
     ({benchmark_comparison['metrics']['wpm']['difference']:+.1f}, {benchmark_comparison['metrics']['wpm']['status']})

Fillers: {benchmark_comparison['metrics']['fillers']['your_score']:.2f} vs {benchmark_comparison['metrics']['fillers']['professional_avg']:.2f} per min
        ({benchmark_comparison['metrics']['fillers']['status']})

Eye Contact: {benchmark_comparison['metrics']['eye_contact']['your_score']:.1f}% vs {benchmark_comparison['metrics']['eye_contact']['professional_avg']}%
            ({benchmark_comparison['metrics']['eye_contact']['difference']:+.1f}%, {benchmark_comparison['metrics']['eye_contact']['status']})

═══════════════════════════════════════════════════════════
                    FULL TRANSCRIPT
═══════════════════════════════════════════════════════════
{metrics.get('transcript', 'No transcript available')}

═══════════════════════════════════════════════════════════
                    AI FEEDBACK SUMMARY
═══════════════════════════════════════════════════════════
{metrics.get('ai_feedback', 'No AI feedback available')}

═══════════════════════════════════════════════════════════
Report Generated by AI Presentation Coach Pro v4.0
DES646 Project - 2025
═══════════════════════════════════════════════════════════
"""
            
            st.download_button(
                "📥 Download Full Report",
                report_data,
                file_name=f"presentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ============================================================================
# PAGE 2: SYSTEM CHECK
# ============================================================================
elif "🧪 System Check" in page:
    st.header("🔧 System Diagnostics & Testing")
    
    st.markdown("""
    <div class="feedback-box">
    <h3 style="color: #2c3e50 !important;">System Requirements Check</h3>
    <p style="color: #2c3e50 !important;">
    Testing all hardware and software components required for optimal performance.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Camera Test")
        if st.button("🎥 Test Camera", use_container_width=True):
            with st.spinner("Testing camera..."):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="📸 Camera Preview", use_container_width=True)
                        st.success("✅ Camera is working perfectly!")
                        
                        # Test face detection on preview
                        eye_contact, face_detected, _, _, _ = detect_face_and_eyes_advanced(frame)
                        if face_detected:
                            st.info(f"👤 Face Detection: ✅ Working\n\n👁️ Eye Detection: {'✅ Working' if eye_contact else '⚠️ Adjust position'}")
                        else:
                            st.warning("⚠️ Face not detected in preview. Make sure you're facing the camera.")
                    else:
                        st.error("❌ Camera detected but cannot capture frames")
                    cap.release()
                else:
                    st.error("❌ No camera detected. Please check:\n1. Camera permissions\n2. Camera is not in use by another app\n3. Camera drivers are installed")
    
    with col2:
        st.subheader("🎤 Microphone Test")
        if st.button("🎙️ Test Microphone", use_container_width=True):
            with st.spinner("Testing microphone..."):
                success, message, mic_list = test_microphone()
                if success:
                    st.success(f"✅ {message}")
                    if mic_list:
                        with st.expander("📋 Available Microphones"):
                            for i, mic_name in enumerate(mic_list):
                                st.write(f"**Device {i}:** {mic_name}")
                    
                    # Quick recording test
                    st.info("🎤 Speak now for 3 seconds to test recording...")
                    test_queue = queue.Queue()
                    test_thread = threading.Thread(
                        target=record_audio_advanced,
                        args=(3, test_queue, None)
                    )
                    test_thread.start()
                    
                    progress = st.progress(0)
                    for i in range(30):
                        time.sleep(0.1)
                        progress.progress((i + 1) / 30)
                    
                    test_thread.join()
                    
                    try:
                        result = test_queue.get_nowait()
                        if result and not result.startswith("ERROR"):
                            st.success(f"✅ Recording successful!\n\n**Transcribed:** {result}")
                        elif result.startswith("ERROR"):
                            st.error(f"❌ {result}")
                        else:
                            st.warning("⚠️ No speech detected. Please speak louder.")
                    except:
                        st.error("❌ Recording test failed")
                else:
                    st.error(f"❌ {message}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👁️ Face Detection Test")
        if st.button("🔍 Test Face Detection", use_container_width=True):
            with st.spinner("Testing face detection algorithms..."):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        eye_contact, face_detected, face_size, _, _ = detect_face_and_eyes_advanced(frame)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Face Detection Test", use_container_width=True)
                        
                        if face_detected:
                            st.success("✅ Face detection working!")
                            st.info(f"""
                            **Detection Results:**
                            - Face: ✅ Detected
                            - Eye Contact: {'✅ Detected' if eye_contact else '❌ Not Detected'}
                            - Face Size: {face_size} pixels²
                            - Quality: {'Excellent' if face_size > 10000 else 'Good' if face_size > 5000 else 'Fair'}
                            """)
                        else:
                            st.warning("⚠️ No face detected. Please:\n1. Face the camera directly\n2. Ensure good lighting\n3. Remove obstructions\n4. Adjust camera angle")
                    cap.release()
                else:
                    st.error("❌ Cannot access camera")
    
    with col2:
        st.subheader("😊 Emotion Detection Test")
        if st.button("😄 Test Emotion Detection", use_container_width=True):
            with st.spinner("Testing emotion recognition..."):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        emotion, confidence, smiling = detect_emotions_advanced(frame)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Emotion Detection Test", use_container_width=True)
                        
                        st.success(f"""
                        ✅ Emotion detection working!
                        
                        **Detected Emotion:** {emotion.title()}
                        **Confidence:** {confidence:.0f}%
                        **Smiling:** {'Yes 😊' if smiling else 'No'}
                        """)
                    cap.release()
                else:
                    st.error("❌ Cannot access camera")
    
    st.markdown("---")
    
    st.subheader("📦 Dependencies Check")
    
    dependencies = {
        "Core Libraries": {
            "Streamlit": st,
            "OpenCV": cv2,
            "NumPy": np,
            "Pandas": pd,
            "Matplotlib": plt,
            "Seaborn": sns
        },
        "Audio Processing": {
            "Speech Recognition": sr,
            "Requests": requests
        },
        "Utilities": {
            "PIL": Image,
            "SQLite3": sqlite3,
            "JSON": json
        }
    }
    
    for category, libs in dependencies.items():
        st.markdown(f"**{category}:**")
        cols = st.columns(len(libs))
        for idx, (name, module) in enumerate(libs.items()):
            with cols[idx]:
                try:
                    version = getattr(module, '__version__', 'Unknown')
                    st.success(f"✅ **{name}**\n\nv{version}")
                except:
                    st.error(f"❌ **{name}**\n\nMissing")
    
    st.markdown("---")
    
    st.subheader("💾 Database Status")
    
    try:
        conn = sqlite3.connect('presentation_history.db')
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM sessions")
        count = c.fetchone()[0]
        conn.close()
        
        st.success(f"✅ Database operational\n\n**Total sessions:** {count}")
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")

# ============================================================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================================================
elif "📊 Analytics" in page:
    st.header("📊 Performance Analytics Dashboard")
    
    history_df = get_session_history()
    
    if len(history_df) == 0:
        st.info("""
        ### 📝 No Data Available Yet
        
        Complete your first practice session to see detailed analytics!
        
        **What you'll get:**
        - Performance trends over time
        - Skill improvement tracking
        - Benchmark comparisons
        - Detailed session history
        - Downloadable reports
        """)
    else:
        # Summary Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{len(history_df)}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Total Sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = history_df['overall_score'].mean()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{avg_score:.1f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Average Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_score = history_df['overall_score'].max()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{best_score:.1f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Best Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_wpm = history_df['wpm'].mean()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{avg_wpm:.0f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Avg WPM</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            improvement = history_df.head(3)['overall_score'].mean() - history_df.tail(3)['overall_score'].mean() if len(history_df) >= 3 else 0
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                 padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                <h2 style="color: white; margin: 0;">{improvement:+.1f}</h2>
                <p style="margin: 0.5rem 0 0 0; color: white;">Improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Progress Chart
        st.subheader("📈 Performance Progression")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        recent = history_df.head(20).iloc[::-1]
        sessions = range(1, len(recent) + 1)
        
        ax.plot(sessions, recent['overall_score'], marker='o', linewidth=3,
                color='#667eea', markersize=10, label='Overall Score', alpha=0.8)
        ax.plot(sessions, recent['eye_contact_score'], marker='s', linewidth=2,
                color='#764ba2', markersize=8, alpha=0.7, label='Eye Contact')
        ax.plot(sessions, recent['wpm'] * 0.5, marker='^', linewidth=2,
                color='#f093fb', markersize=8, alpha=0.7, label='WPM (scaled)')
        
        ax.set_xlabel('Session Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance Over Time', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
        
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed History Table
        st.subheader("📋 Session History")
        
        display_columns = ['date', 'overall_score', 'wpm', 'filler_count', 
                          'eye_contact_score', 'duration']
        
        if 'vocabulary_richness' in history_df.columns:
            display_columns.insert(5, 'vocabulary_richness')
        
        display_df = history_df[display_columns].copy()
        display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
        
        column_names = ['Date', 'Score', 'WPM', 'Fillers', 'Eye %']
        if 'vocabulary_richness' in display_columns:
            column_names.append('Vocab %')
        column_names.append('Duration (s)')
        
        display_df.columns = column_names
        display_df = display_df.round(1)
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download History
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full History (CSV)",
            csv,
            file_name=f"presentation_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# PAGE 4: SETTINGS
# ============================================================================
elif "⚙️ Settings" in page:
    st.header("⚙️ Application Settings")
    
    st.markdown("""
    <div class="feedback-box">
    <h3 style="color: #2c3e50 !important;">🤖 Google Gemini AI Integration</h3>
    <p style="color: #2c3e50 !important;">
    Enable advanced AI-powered feedback by connecting your Google Gemini API key.
    This is optional - the system works without it using rule-based feedback.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input(
            "🔑 Enter your Gemini API Key:",
            type="password",
            value=st.session_state.gemini_api_key,
            placeholder="AIzaSy..."
        )
        
        if st.button("💾 Save API Key", use_container_width=True):
            st.session_state.gemini_api_key = api_key
            st.success("✅ API Key saved successfully!")
            st.info("The AI feedback will be used in your next presentation session.")
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
             border: 2px solid #667eea;">
        <h4 style="color: #2c3e50 !important; margin-top: 0;">📚 How to Get API Key:</h4>
        <ol style="color: #2c3e50 !important; font-size: 0.9rem; padding-left: 1.2rem;">
            <li>Visit <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
            <li>Sign in with Google</li>
            <li>Click "Get API Key"</li>
            <li>Create new key</li>
            <li>Copy and paste here</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🧪 Test AI Connection")
    
    if st.button("🔍 Test Gemini API", use_container_width=True):
        if not st.session_state.gemini_api_key:
            st.warning("⚠️ No API key configured. Using rule-based feedback (which works great!).")
        else:
            with st.spinner("Testing API connection..."):
                test_metrics = {
                    'wpm': 155,
                    'filler_count': 2,
                    'eye_contact_score': 85,
                    'duration': 120,
                    'word_count': 310,
                    'overall_score': 88,
                    'vocabulary_richness': 62,
                    'confidence_score': 85,
                    'clarity_score': 90,
                    'engagement_score': 87,
                    'fluency_score': 92
                }
                
                feedback = get_gemini_feedback("This is a test presentation to verify the AI feedback system.", test_metrics)
                
                if "error" in feedback.lower() or len(feedback) < 50:
                    st.error("❌ API test failed. Please check your API key.")
                else:
                    st.success("✅ Gemini AI connected successfully!")
                    with st.expander("📄 View Sample AI Feedback"):
                        st.markdown(f"""
                        <div class="ai-insight">
                        {feedback}
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🗑️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.warning("⚠️ This will delete all session history!")
            if st.button("⚠️ Confirm Reset", use_container_width=True):
                try:
                    conn = sqlite3.connect('presentation_history.db')
                    c = conn.cursor()
                    c.execute('DELETE FROM sessions')
                    conn.commit()
                    conn.close()
                    st.success("✅ All statistics reset successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.info("""
        **📊 Current Stats:**
        - Sessions: {len(get_session_history())}
        - Database: presentation_history.db
        """)

# ============================================================================
# PAGE 5: ABOUT PROJECT
# ============================================================================
elif "ℹ️ About" in page:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem; 
         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         border-radius: 25px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🎯 AI Presentation Coach Pro</h1>
        <p style="color: white; margin: 1rem 0; font-size: 1.3rem; opacity: 0.95;">
            Advanced Multimodal Presentation Analysis System
        </p>
        <p style="color: white; margin: 0.5rem 0; font-size: 1rem; opacity: 0.85;">
            Version 4.0 - Ultimate Edition with Professional Benchmarking
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">🎓 Project Details</h3>
        <ul style="color: #2c3e50 !important; font-size: 1rem;">
            <li><strong>Course:</strong> DES646 - Design Engineering</li>
            <li><strong>Institution:</strong> Engineering College</li>
            <li><strong>Year:</strong> 2025</li>
            <li><strong>Version:</strong> 4.0 Ultimate Edition</li>
            <li><strong>Category:</strong> AI/ML, Computer Vision, NLP</li>
            <li><strong>Status:</strong> Production Ready</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">👥 Development Team</h3>
        <ul style="color: #2c3e50 !important; font-size: 1rem;">
            <li><strong>Surya</strong> - Lead Developer</li>
            <li><strong>Pushpendra</strong> - AI/ML Engineer</li>
            <li><strong>Meet</strong> - Computer Vision</li>
            <li><strong>Vasundhara</strong> - Data Analysis</li>
            <li><strong>Ayush</strong> - UI/UX Design</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feedback-box">
        <h3 style="color: #2c3e50 !important;">🔧 Technologies Used</h3>
        <ul style="color: #2c3e50 !important; font-size: 0.95rem;">
            <li><strong>Frontend:</strong> Streamlit</li>
            <li><strong>Computer Vision:</strong> OpenCV, Haar Cascades</li>
            <li><strong>Speech Processing:</strong> Google Speech API</li>
            <li><strong>AI Integration:</strong> Google Gemini Pro</li>
            <li><strong>Data Analysis:</strong> Pandas, NumPy</li>
            <li><strong>Visualization:</strong> Matplotlib, Seaborn</li>
            <li><strong>Database:</strong> SQLite3</li>
            <li><strong>Language:</strong> Python 3.8+</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("✨ Key Features")
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        <div class="pro-feature">
        <h4 style="color: white !important;">🎥 Video Analysis</h4>
        <ul style="color: white !important; font-size: 0.9rem;">
            <li>Real-time face detection</li>
            <li>Eye contact tracking</li>
            <li>Emotion recognition</li>
            <li>Body language analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class="pro-feature">
        <h4 style="color: white !important;">🎤 Speech Analysis</h4>
        <ul style="color: white !important; font-size: 0.9rem;">
            <li>Speech-to-text transcription</li>
            <li>Speaking pace (WPM)</li>
            <li>Filler word detection</li>
            <li>Vocabulary richness</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col3:
        st.markdown("""
        <div class="pro-feature">
        <h4 style="color: white !important;">📊 Analytics</h4>
        <ul style="color: white !important; font-size: 0.9rem;">
            <li>Professional benchmarking</li>
            <li>Progress tracking</li>
            <li>Detailed visualizations</li>
            <li>AI-powered feedback</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📚 Documentation")
    
    st.markdown("""
    <div class="feedback-box">
    <h4 style="color: #2c3e50 !important;">📖 User Guide</h4>
    <ol style="color: #2c3e50 !important;">
        <li><strong>Setup:</strong> Ensure camera and microphone are connected</li>
        <li><strong>Practice:</strong> Select duration and start recording</li>
        <li><strong>Review:</strong> Analyze detailed feedback and metrics</li>
        <li><strong>Improve:</strong> Track progress over multiple sessions</li>
        <li><strong>Compare:</strong> Benchmark against professional speakers</li>
    </ol>
    
    <h4 style="color: #2c3e50 !important; margin-top: 1.5rem;">🎯 Best Practices</h4>
    <ul style="color: #2c3e50 !important;">
        <li>Practice in a quiet, well-lit environment</li>
        <li>Position yourself at eye level with camera</li>
        <li>Speak naturally as if presenting to an audience</li>
        <li>Review feedback after each session</li>
        <li>Focus on one improvement area at a time</li>
        <li>Practice regularly for best results</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; 
     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
     border-radius: 20px; color: white; margin-top: 3rem;
     box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);">
    <h3 style="color: white; margin: 0; font-size: 1.8rem;">🎯 AI Presentation Coach Pro</h3>
    <p style="margin: 0.8rem 0; color: white; font-size: 1rem;">
        DES646 Engineering Project © 2025
    </p>
    <p style="margin: 0; font-size: 0.85rem; opacity: 0.9; color: white;">
        Version 4.0 - Ultimate Edition | Built with ❤️ by Team DES646
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.75rem; opacity: 0.8; color: white;">
        Powered by Streamlit • OpenCV • Google AI
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# END OF APPLICATION
# ============================================================================

