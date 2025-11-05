import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3
import json
import time
from collections import Counter
import speech_recognition as sr
import threading
import queue

# Configure page
st.set_page_config(
    page_title="AI Presentation Coach",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'recorded_text' not in st.session_state:
    st.session_state.recorded_text = ""
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []
if 'emotion_data' not in st.session_state:
    st.session_state.emotion_data = []
if 'mic_checked' not in st.session_state:
    st.session_state.mic_checked = False

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
                  transcript TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Microphone testing function
def test_microphone():
    """Test if microphone is working"""
    recognizer = sr.Recognizer()
    try:
        # List all microphones
        mic_list = sr.Microphone.list_microphone_names()
        
        if not mic_list:
            return False, "No microphones found", []
        
        # Try to access default microphone
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            return True, f"Microphone working! Found {len(mic_list)} device(s)", mic_list
    except Exception as e:
        return False, f"Microphone error: {str(e)}", []

# Helper Functions
def save_session(data):
    conn = sqlite3.connect('presentation_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO sessions 
                 (date, duration, wpm, filler_count, eye_contact_score, 
                  emotion_scores, pause_count, overall_score, transcript)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (data['date'], data['duration'], data['wpm'], 
               data['filler_count'], data['eye_contact_score'],
               json.dumps(data['emotion_scores']), data['pause_count'],
               data['overall_score'], data.get('transcript', '')))
    conn.commit()
    conn.close()

def get_session_history():
    conn = sqlite3.connect('presentation_history.db')
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY date DESC", conn)
    conn.close()
    return df

def analyze_speech_text(text):
    """Analyze transcribed speech for fillers, pace, and word count"""
    if not text or len(text.strip()) == 0:
        return {'word_count': 0, 'filler_count': 0, 'pause_count': 0}
    
    words = text.lower().split()
    word_count = len(words)
    
    # Filler words detection
    filler_words = ['um', 'uh', 'like', 'so', 'actually', 'basically', 'literally', 'you know']
    filler_count = 0
    for word in words:
        if word in filler_words:
            filler_count += 1
    
    # Count "you know" as phrase
    text_lower = text.lower()
    filler_count += text_lower.count('you know')
    
    # Pause detection
    pause_indicators = text.count('...') + text.count('..') + len([w for w in text.split() if len(w) == 0])
    
    return {
        'word_count': word_count,
        'filler_count': filler_count,
        'pause_count': max(0, pause_indicators)
    }

def detect_face_and_eyes(frame):
    """Detect face and eyes for eye contact estimation"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        eye_contact = False
        face_detected = False
        
        for (x, y, w, h) in faces:
            face_detected = True
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            if len(eyes) >= 2:
                eye_contact = True
                break
        
        return eye_contact, face_detected
    except:
        return False, False

def simple_emotion_detection(frame):
    """Simple emotion detection based on facial features"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return 'neutral'
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            
            if len(smiles) > 0:
                return 'happy'
            else:
                return 'confident'
        
        return 'neutral'
    except:
        return 'neutral'

def record_audio_continuous(duration, result_queue, mic_index=None):
    """Record audio continuously and convert to text"""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 300  # Lower threshold for better detection
    
    full_text = ""
    
    try:
        # Use specific microphone or default
        if mic_index is not None:
            mic = sr.Microphone(device_index=mic_index)
        else:
            mic = sr.Microphone()
        
        with mic as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            start_time = time.time()
            
            # Record in chunks
            while time.time() - start_time < duration:
                try:
                    remaining = duration - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    
                    chunk_duration = min(4, remaining)
                    audio = recognizer.listen(source, timeout=chunk_duration, phrase_time_limit=chunk_duration)
                    
                    try:
                        text = recognizer.recognize_google(audio)
                        if text:
                            full_text += " " + text
                            print(f"Recognized: {text}")  # Debug output
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                        continue
                    except sr.RequestError as e:
                        print(f"API error: {e}")
                        continue
                        
                except sr.WaitTimeoutError:
                    print("Listening timeout, continuing...")
                    continue
                except Exception as e:
                    print(f"Chunk error: {e}")
                    continue
                    
    except Exception as e:
        print(f"Audio recording error: {e}")
        result_queue.put(f"ERROR: {str(e)}")
        return
    
    result_queue.put(full_text.strip())

def calculate_overall_score(metrics):
    """Calculate overall presentation score"""
    score = 100
    
    # Deduct for fillers (max -20)
    filler_penalty = min(metrics['filler_count'] * 2, 20)
    score -= filler_penalty
    
    # Deduct for poor eye contact (max -25)
    eye_contact_penalty = (100 - metrics['eye_contact_score']) * 0.25
    score -= eye_contact_penalty
    
    # Deduct for poor pacing (max -15)
    if metrics['wpm'] < 120 or metrics['wpm'] > 180:
        pace_penalty = min(abs(150 - metrics['wpm']) * 0.3, 15)
        score -= pace_penalty
    
    # Deduct for low word count (max -10)
    if metrics['word_count'] < 20:
        score -= 10
    
    return max(0, min(100, score))

def generate_feedback(metrics):
    """Generate personalized feedback based on metrics"""
    feedback = []
    
    # WPM feedback
    if metrics['wpm'] < 120:
        feedback.append(f"🐢 **Pacing**: Your speech pace is slow ({int(metrics['wpm'])} wpm). Try to speak a bit faster for better engagement. Target: 120-180 wpm")
    elif metrics['wpm'] > 180:
        feedback.append(f"🏃 **Pacing**: You're speaking quite fast ({int(metrics['wpm'])} wpm). Slow down to ensure clarity. Target: 120-180 wpm")
    else:
        feedback.append(f"✅ **Pacing**: Excellent pace ({int(metrics['wpm'])} wpm)! You're in the ideal range (120-180 wpm).")
    
    # Filler words feedback
    if metrics['filler_count'] > 10:
        feedback.append(f"⚠️ **Filler Words**: High filler word usage ({metrics['filler_count']} detected). Practice pausing instead of using fillers like 'um', 'uh', 'like'.")
    elif metrics['filler_count'] > 5:
        feedback.append(f"⚡ **Filler Words**: Moderate filler usage ({metrics['filler_count']}). Try to reduce them further with conscious pauses.")
    else:
        feedback.append(f"✅ **Filler Words**: Great job! Minimal filler words detected ({metrics['filler_count']}).")
    
    # Eye contact feedback
    if metrics['eye_contact_score'] < 50:
        feedback.append(f"👀 **Eye Contact**: Low eye contact ({int(metrics['eye_contact_score'])}%). Look more directly at the camera. Aim for 70%+")
    elif metrics['eye_contact_score'] < 75:
        feedback.append(f"👁️ **Eye Contact**: Good eye contact ({int(metrics['eye_contact_score'])}%). Keep improving to reach 75%+!")
    else:
        feedback.append(f"✅ **Eye Contact**: Excellent engagement ({int(metrics['eye_contact_score'])}%)! Well done.")
    
    # Emotion feedback
    if metrics['dominant_emotion'] == 'neutral':
        feedback.append("😐 **Expression**: Your expression appeared mostly neutral. Try to show more enthusiasm and energy!")
    elif metrics['dominant_emotion'] == 'happy':
        feedback.append("😊 **Expression**: Great positive energy! Your expressions were engaging.")
    else:
        feedback.append(f"✅ **Expression**: You projected {metrics['dominant_emotion']} demeanor. Keep it natural!")
    
    # Word count feedback
    if metrics['word_count'] < 30:
        feedback.append(f"📝 **Content**: Only {metrics['word_count']} words detected. Try speaking more to get better analysis.")
    
    return feedback

def plot_metrics(metrics, history_df=None):
    """Create visualization of metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Presentation Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Current session metrics
    categories = ['Overall\nScore', 'Eye Contact', 'Pace\nScore', 'Fluency\nScore']
    pace_score = 100 if 120 <= metrics['wpm'] <= 180 else max(0, 100 - abs(150 - metrics['wpm']) * 0.5)
    fluency_score = max(0, 100 - metrics['filler_count'] * 3)
    scores = [metrics['overall_score'], metrics['eye_contact_score'], pace_score, fluency_score]
    
    colors = ['#2ecc71' if s >= 75 else '#f39c12' if s >= 50 else '#e74c3c' for s in scores]
    axes[0, 0].bar(categories, scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_ylabel('Score (%)', fontsize=11)
    axes[0, 0].set_title('Current Session Scores', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Metrics breakdown
    metric_names = ['Words', 'Fillers', 'Pauses', 'Duration(s)']
    metric_values = [metrics['word_count'], metrics['filler_count'], 
                     metrics['pause_count'], int(metrics['duration'])]
    axes[0, 1].barh(metric_names, metric_values, color='#3498db', edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Count', fontsize=11)
    axes[0, 1].set_title('Session Metrics', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Historical trend
    if history_df is not None and len(history_df) > 1:
        recent = history_df.head(10).iloc[::-1]
        axes[1, 0].plot(range(len(recent)), recent['overall_score'], 
                       marker='o', color='#9b59b6', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Session Number (Recent)', fontsize=11)
        axes[1, 0].set_ylabel('Overall Score', fontsize=11)
        axes[1, 0].set_title('Progress Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
    else:
        axes[1, 0].text(0.5, 0.5, 'Complete more sessions\nto see progress', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Progress Over Time', fontsize=12, fontweight='bold')
    
    # Emotion distribution
    if 'emotion_scores' in metrics and metrics['emotion_scores']:
        emotions = list(metrics['emotion_scores'].keys())
        counts = list(metrics['emotion_scores'].values())
        if sum(counts) > 0:
            axes[1, 1].pie(counts, labels=emotions, autopct='%1.1f%%', 
                          startangle=90, textprops={'fontsize': 10})
            axes[1, 1].set_title('Emotion Distribution', fontsize=12, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No emotion data', ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Emotion Distribution', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No emotion data', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Emotion Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main UI
st.title("🎤 AI Presentation Coach - Live Recording")
st.markdown("### Practice your presentations with real-time analysis")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard")
    page = st.radio("Navigation", ["Live Practice", "Microphone Test", "History & Progress", "About"])

if page == "Microphone Test":
    st.header("🎤 Microphone Diagnostics")
    
    st.markdown("""
    ### Test your microphone before recording
    This will help identify and fix any audio issues.
    """)
    
    if st.button("🔍 Test Microphone", type="primary"):
        with st.spinner("Testing microphone..."):
            success, message, mic_list = test_microphone()
            
            if success:
                st.success(f"✅ {message}")
                
                st.subheader("Available Microphones:")
                for i, mic_name in enumerate(mic_list):
                    st.write(f"{i}. {mic_name}")
                
                st.info("💡 Your default microphone is working! You can now use Live Practice.")
                
            else:
                st.error(f"❌ {message}")
                
                st.markdown("""
                ### Troubleshooting Steps:
                
                #### Windows:
                1. **Check Privacy Settings:**
                   - Go to Settings > Privacy > Microphone
                   - Enable "Allow apps to access your microphone"
                   - Scroll down and enable for Python/VS Code
                
                2. **Check Device Manager:**
                   - Right-click Start > Device Manager
                   - Expand "Audio inputs and outputs"
                   - Make sure microphone is enabled
                
                3. **Test in Sound Settings:**
                   - Right-click speaker icon > Sounds
                   - Go to Recording tab
                   - Speak and check if the bar moves
                
                #### Mac:
                1. **Check System Preferences:**
                   - Go to System Preferences > Security & Privacy
                   - Click Microphone tab
                   - Enable Terminal/VS Code/iTerm
                
                2. **Test microphone:**
                   - Open QuickTime Player
                   - File > New Audio Recording
                   - Check if it detects sound
                
                #### Linux:
                ```bash
                # Check microphone
                arecord -l
                
                # Test recording
                arecord -d 5 test.wav
                aplay test.wav
                ```
                
                #### Common Issues:
                - **No microphone found**: Check if microphone is plugged in
                - **Permission denied**: Grant microphone permissions to Terminal/Python
                - **Device in use**: Close other applications using microphone
                - **Wrong default device**: Set correct default in system settings
                """)
    
    st.markdown("---")
    st.subheader("🎙️ Quick Microphone Test")
    
    recognizer = sr.Recognizer()
    
    if st.button("🔴 Record 3 Seconds"):
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                
                st.success("✅ Audio captured! Processing...")
                
                try:
                    text = recognizer.recognize_google(audio)
                    st.success(f"**Recognized:** {text}")
                except sr.UnknownValueError:
                    st.warning("⚠️ Could not understand audio. Speak louder and clearer.")
                except sr.RequestError:
                    st.error("❌ Could not request results. Check internet connection.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure microphone permissions are granted!")

elif page == "Live Practice":
    st.header("🎥 Live Recording Session")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Select recording duration
        2. Click **Start Live Recording**
        3. Speak naturally while looking at camera
        4. Analysis starts automatically when done
        """)
        
        # Duration selection
        duration_options = {
            "30 seconds": 30,
            "1 minute": 60,
            "2 minutes": 120,
            "3 minutes": 180,
            "5 minutes": 300
        }
        
        selected_duration = st.selectbox(
            "Select Recording Duration:",
            options=list(duration_options.keys()),
            index=1
        )
        
        duration = duration_options[selected_duration]
        
        # Microphone selection (optional)
        try:
            mic_list = sr.Microphone.list_microphone_names()
            if len(mic_list) > 1:
                mic_choice = st.selectbox(
                    "Select Microphone (optional):",
                    options=["Default"] + [f"{i}: {name}" for i, name in enumerate(mic_list)]
                )
                if mic_choice != "Default":
                    selected_mic_index = int(mic_choice.split(":")[0])
                else:
                    selected_mic_index = None
            else:
                selected_mic_index = None
        except:
            selected_mic_index = None
        
        # Start recording button
        if not st.session_state.recording:
            if st.button("🔴 Start Live Recording", type="primary", use_container_width=True):
                # Test microphone first
                success, _, _ = test_microphone()
                if not success:
                    st.error("❌ Microphone not detected! Please check 'Microphone Test' page.")
                else:
                    st.session_state.recording = True
                    st.session_state.recorded_text = ""
                    st.session_state.video_frames = []
                    st.session_state.emotion_data = []
                    st.rerun()
        
        # Recording in progress
        if st.session_state.recording:
            st.warning(f"🔴 **RECORDING IN PROGRESS** - {selected_duration}")
            st.info("💡 **Speak clearly and naturally. Look at the camera!**")
            
            # Placeholders
            video_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            audio_status = st.empty()
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Please check your camera permissions.")
                st.session_state.recording = False
                st.stop()
            
            # Start audio recording in separate thread
            audio_queue = queue.Queue()
            audio_thread = threading.Thread(
                target=record_audio_continuous, 
                args=(duration, audio_queue, selected_mic_index if 'selected_mic_index' in locals() else None)
            )
            audio_thread.start()
            audio_status.info("🎤 Audio recording started...")
            
            # Record video
            start_time = time.time()
            frame_count = 0
            eye_contact_frames = 0
            total_frames = 0
            emotion_list = []
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Analyze frame
                eye_contact, face_detected = detect_face_and_eyes(frame)
                emotion = simple_emotion_detection(frame)
                
                if face_detected:
                    total_frames += 1
                    if eye_contact:
                        eye_contact_frames += 1
                    emotion_list.append(emotion)
                
                # Draw info on frame
                if face_detected:
                    cv2.putText(frame, f"Face: {emotion.title()}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if eye_contact:
                        cv2.putText(frame, "Good Eye Contact!", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face Detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update progress
                elapsed = time.time() - start_time
                progress = min(elapsed / duration, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Recording: {int(elapsed)}s / {duration}s")
                
                frame_count += 1
                time.sleep(0.03)  # ~30 FPS
            
            # Stop recording
            cap.release()
            st.session_state.recording = False
            audio_status.info("⏳ Processing audio... Please wait...")
            
            # Wait for audio thread
            audio_thread.join(timeout=10)
            
            # Get transcribed text
            try:
                transcribed_text = audio_queue.get_nowait()
                if transcribed_text.startswith("ERROR:"):
                    st.error(transcribed_text)
                    transcribed_text = ""
            except:
                transcribed_text = ""
            
            # Calculate metrics
            speech_metrics = analyze_speech_text(transcribed_text)
            
            # Calculate WPM
            if duration > 0 and speech_metrics['word_count'] > 0:
                wpm = (speech_metrics['word_count'] / duration) * 60
            else:
                wpm = 0
            
            # Calculate eye contact score
            if total_frames > 0:
                eye_contact_score = (eye_contact_frames / total_frames) * 100
            else:
                eye_contact_score = 0
            
            # Calculate emotion distribution
            emotion_counter = Counter(emotion_list)
            emotion_scores = dict(emotion_counter)
            if emotion_scores:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            else:
                dominant_emotion = 'neutral'
            
            # Compile metrics
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
                'overall_score': 0
            }
            
            metrics['overall_score'] = calculate_overall_score(metrics)
            
            # Save to database
            save_session(metrics)
            
            st.session_state.session_data = metrics
            st.session_state.analysis_complete = True
            
            st.success("✅ Recording complete! Analyzing...")
            time.sleep(1)
            st.rerun()
    
    with col2:
        st.subheader("💡 Tips")
        st.info("""
        **For Best Results:**
        - Good lighting on face
        - Look at camera often
        - Speak clearly & loudly
        - Minimize background noise
        - Check microphone first
        """)
        
        st.subheader("🎯 Goals")
        st.markdown("""
        - **WPM**: 120-180
        - **Eye Contact**: >70%
        - **Fillers**: <5
        - **Expression**: Positive
        """)
        
        # Microphone status
        if st.button("🔍 Check Mic"):
            success, msg, _ = test_microphone()
            if success:
                st.success("✅ Mic OK")
            else:
                st.error("❌ Mic Issue")
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.session_data:
        st.success("✅ Analysis Complete!")
        
        metrics = st.session_state.session_data
        
        # Overall score
        st.markdown("---")
        score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
        with score_col2:
            score = metrics['overall_score']
            if score >= 80:
                color = "#2ecc71"
                emoji = "🌟"
                level = "Excellent"
            elif score >= 60:
                color = "#f39c12"
                emoji = "👍"
                level = "Good"
            else:
                color = "#e74c3c"
                emoji = "📈"
                level = "Needs Improvement"
            
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji} {score:.1f}/100</h1>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>{level}</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Transcript
        if metrics.get('transcript') and len(metrics['transcript']) > 0:
            with st.expander("📝 View Transcript", expanded=True):
                st.write(metrics['transcript'])
        else:
            st.warning("⚠️ **No speech detected!** Check:")
            st.markdown("""
            - Microphone is unmuted
            - Microphone permissions granted
            - Speaking loud enough
            - Internet connection active
            - Try 'Microphone Test' page
            """)
        
        # Detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            wpm_delta = "Optimal" if 120 <= metrics['wpm'] <= 180 else "Adjust"
            st.metric("Words Per Minute", f"{metrics['wpm']:.0f}", delta=wpm_delta)
        
        with col2:
            filler_delta = "Good" if metrics['filler_count'] < 5 else "Reduce"
            st.metric("Filler Words", metrics['filler_count'], delta=filler_delta)
        
        with col3:
            eye_delta = "Great" if metrics['eye_contact_score'] > 70 else "Improve"
            st.metric("Eye Contact", f"{metrics['eye_contact_score']:.0f}%", delta=eye_delta)
        
        with col4:
            st.metric("Dominant Emotion", metrics['dominant_emotion'].title())
        
        # Visualizations
        st.subheader("📊 Detailed Analysis")
        history_df = get_session_history()
        fig = plot_metrics(metrics, history_df)
        st.pyplot(fig)
        
        # Feedback
        st.subheader("💬 Personalized Feedback")
        feedback = generate_feedback(metrics)
        for item in feedback:
            st.markdown(item)
        
        # Improvement suggestions
        st.subheader("🎯 Action Items for Next Session")
        suggestions = []
        if metrics['filler_count'] > 5:
            suggestions.append("Practice pausing for 1-2 seconds instead of using filler words")
        if metrics['eye_contact_score'] < 70:
            suggestions.append("Place a mark near your camera and focus on it while speaking")
        if metrics['wpm'] < 120:
            suggestions.append("Read aloud daily to naturally increase your speaking pace")
        if metrics['wpm'] > 180:
            suggestions.append("Practice with a timer - aim for 150 words per minute")
        if metrics['word_count'] < 30:
            suggestions.append("Speak more during recording to get comprehensive feedback")
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        else:
            st.success("Great job! Keep practicing to maintain your performance level.")
        
        # Reset button
        if st.button("🔄 Start New Session", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.session_data = None
            st.rerun()

elif page == "History & Progress":
    st.header("📈 Your Progress Journey")
    
    history_df = get_session_history()
    
    if len(history_df) > 0:
        st.subheader(f"Total Sessions: {len(history_df)}")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = history_df['overall_score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}/100")
        
        with col2:
            avg_wpm = history_df['wpm'].mean()
            st.metric("Average WPM", f"{avg_wpm:.0f}")
        
        with col3:
            avg_fillers = history_df['filler_count'].mean()
            st.metric("Avg Filler Words", f"{avg_fillers:.1f}")
        
        with col4:
            avg_eye = history_df['eye_contact_score'].mean()
            st.metric("Avg Eye Contact", f"{avg_eye:.0f}%")
        
        # Progress charts
        st.subheader("📊 Progress Over Time")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall score trend
        axes[0, 0].plot(range(len(history_df)), history_df['overall_score'][::-1], 
                       marker='o', linewidth=2, color='#9b59b6', markersize=8)
        axes[0, 0].set_xlabel('Session Number', fontsize=11)
        axes[0, 0].set_ylabel('Overall Score', fontsize=11)
        axes[0, 0].set_title('Overall Score Progression', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 105)
        
        # WPM trend
        axes[0, 1].plot(range(len(history_df)), history_df['wpm'][::-1], 
                       marker='s', linewidth=2, color='#3498db', markersize=8)
        axes[0, 1].axhline(y=120, color='g', linestyle='--', alpha=0.5, label='Min optimal')
        axes[0, 1].axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Max optimal')
        axes[0, 1].set_xlabel('Session Number', fontsize=11)
        axes[0, 1].set_ylabel('Words Per Minute', fontsize=11)
        axes[0, 1].set_title('Speaking Pace Progression', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Filler words trend
        axes[1, 0].plot(range(len(history_df)), history_df['filler_count'][::-1], 
                       marker='^', linewidth=2, color='#e74c3c', markersize=8)
        axes[1, 0].set_xlabel('Session Number', fontsize=11)
        axes[1, 0].set_ylabel('Filler Word Count', fontsize=11)
        axes[1, 0].set_title('Filler Words Reduction', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Eye contact trend
        axes[1, 1].plot(range(len(history_df)), history_df['eye_contact_score'][::-1], 
                       marker='D', linewidth=2, color='#2ecc71', markersize=8)
        axes[1, 1].set_xlabel('Session Number', fontsize=11)
        axes[1, 1].set_ylabel('Eye Contact Score (%)', fontsize=11)
        axes[1, 1].set_title('Eye Contact Improvement', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 105)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Session history table
        st.subheader("📋 Session History")
        display_df = history_df[['date', 'overall_score', 'wpm', 'filler_count', 
                                'eye_contact_score']].copy()
        display_df.columns = ['Date', 'Overall Score', 'WPM', 'Filler Words', 'Eye Contact %']
        display_df['Overall Score'] = display_df['Overall Score'].round(1)
        display_df['WPM'] = display_df['WPM'].round(0)
        display_df['Eye Contact %'] = display_df['Eye Contact %'].round(0)
        st.dataframe(display_df, use_container_width=True)
        
        # Download data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"presentation_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("No practice sessions yet. Start your first session to see your progress!")
        st.markdown("### 🚀 Get Started")
        st.markdown("Complete live recording sessions to track your improvement over time.")

else:  # About page
    st.header("About AI Presentation Coach")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    The **Emotion-Aware AI Presentation Coach** is a multimodal feedback system designed to help 
    students and professionals improve their public speaking confidence and presentation skills.
    
    ### ✨ Key Features
    
    - **Live Video Recording**: Records from webcam with real-time face detection
    - **Speech Recognition**: Converts speech to text using Google Speech API
    - **Microphone Diagnostics**: Built-in testing to troubleshoot audio issues
    - **Speech Analysis**: Detects pacing (WPM), filler words, and pauses
    - **Emotion Recognition**: Analyzes facial expressions during presentations
    - **Eye Contact Tracking**: Estimates gaze direction and engagement
    - **Personalized Feedback**: Generates actionable improvement suggestions
    - **Progress Tracking**: Monitors improvement across multiple sessions
    - **Visual Dashboard**: Comprehensive analytics and trend visualization
    
    ### 🛠️ Technology Stack
    
    - **Framework**: Streamlit
    - **Computer Vision**: OpenCV with Haar Cascades
    - **Speech Recognition**: SpeechRecognition (Google API)
    - **Data Storage**: SQLite
    - **Visualization**: Matplotlib, Pandas
    
    ### 👥 Development Team
    
    - Surya Shukla (221109) - Point of Contact
    - Pushpendra Singh (220841)
    - Meet Pal Singh (220644)
    - Vasundhara Agarwal (221177)
    - Ayush (220259)
    
    ### 📚 Research Direction
    
    This project aims for publication at venues including:
    - ACM IUI 2026 (Intelligent User Interfaces)
    - India HCI 2026
    - ACII (Affective Computing and Intelligent Interaction)
    
    ### 🎓 Course Information
    
    **Course**: DES646 - Mid-Term Assessment Project  
    **Project**: Emotion-Aware AI Presentation Coach
    
    ---
    
    ### 📖 How to Use
    
    1. **Test Microphone**: Visit 'Microphone Test' page to ensure audio is working
    2. **Select Duration**: Choose recording length (30s to 5min)
    3. **Start Recording**: Click the button and begin presenting
    4. **Speak Naturally**: Look at camera and deliver your content clearly
    5. **Get Feedback**: Receive detailed analysis with improvement tips
    6. **Track Progress**: Review history to see your improvement
    
    ### 💻 System Requirements
    
    - **Webcam**: Required for video recording
    - **Microphone**: Required for speech recognition (with proper permissions)
    - **Internet**: Required for Google Speech API
    - **Python**: 3.8+ with required libraries
    
    ### 📦 Installation
    
    ```bash
    pip install streamlit opencv-python speechrecognition pyaudio pandas matplotlib
    ```
    
    ### 🚀 Running the Application
    
    ```bash
    streamlit run app.py
    ```
    
    ### 🔧 Troubleshooting Microphone Issues
    
    **Windows:**
    - Settings > Privacy > Microphone > Allow apps to access microphone
    - Enable for Python/Terminal
    
    **Mac:**
    - System Preferences > Security & Privacy > Microphone
    - Enable Terminal/VS Code/iTerm
    
    **Linux:**
    ```bash
    # Check microphone
    arecord -l
    # Test recording
    arecord -d 5 test.wav && aplay test.wav
    ```
    
    **Common Fixes:**
    - Unmute microphone in system settings
    - Close other apps using microphone (Zoom, Skype, Discord)
    - Select correct microphone in app settings
    - Grant permissions to Terminal/Python
    - Restart Terminal/VS Code after granting permissions
    
    ### 🔮 Future Enhancements
    
    - Advanced emotion recognition with DeepFace/FER
    - MediaPipe face mesh for precise eye tracking
    - Gesture analysis and body language feedback
    - Offline speech recognition with Whisper
    - Multi-language support
    - Export reports as PDF
    - Comparison with expert presentations
    
    ### 📊 Metrics Explained
    
    **Overall Score (0-100):**
    - Combines all metrics into single performance indicator
    - 80+ = Excellent, 60-79 = Good, <60 = Needs Improvement
    
    **Words Per Minute (WPM):**
    - Optimal range: 120-180 WPM
    - Below 120: Too slow, may lose audience
    - Above 180: Too fast, may reduce clarity
    
    **Filler Words:**
    - Counts: um, uh, like, so, actually, basically, literally
    - Target: Less than 5 per session
    
    **Eye Contact Score:**
    - Percentage of time face and eyes detected
    - Target: 70%+ for good engagement
    
    **Emotions:**
    - Tracks: Happy, Confident, Neutral
    - Shows engagement and confidence level
    
    ### 📞 Contact & Support
    
    For questions, feedback, or technical support:
    - Contact: Development Team
    - Course: DES646
    
    ### 📄 License
    
    This project is created for academic purposes as part of DES646 coursework.
    
    ---
    
    **Version 2.1 - Enhanced Microphone Detection & Diagnostics**
    
    *Built with ❤️ for better public speaking*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>DES646 Project - AI Presentation Coach © 2025 | Live Recording with Audio Diagnostics</div>",
    unsafe_allow_html=True
)
