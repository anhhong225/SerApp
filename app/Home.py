import streamlit as st
import sys
from pathlib import Path

# Get the directory where this file is located
current_dir = Path(__file__).parent
assets_dir = current_dir / "assets" / "images" / "icons"

sys.path.append(str(current_dir))
from utils.chatbot_ui import load_css

st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="wide"
)

load_css("global.css", "home.css")

# Header with logo
col1, col2 = st.columns([1, 15])
with col1:
    logo_path = assets_dir / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=70)
    else:
        st.markdown("🎭")
with col2:
    st.title("Speech Emotion Recognition System")

st.markdown("""
## Welcome to the Emotion Recognition Application

### Project Overview

Speech Emotion Recognition (SER) is an emerging field at the intersection of artificial intelligence, psychology, and human-computer interaction. This web application demonstrates a complete end-to-end system capable of identifying human emotions from voice patterns using state-of-the-art deep learning techniques.

**What is Speech Emotion Recognition?**

Speech Emotion Recognition is the process of automatically identifying emotional states from acoustic features in human speech. Unlike traditional speech recognition that focuses on *what* is being said, SER focuses on *how* it is being said by capturing the emotional undertones through vocal characteristics such as pitch, tone, rhythm, intensity, and speaking rate.

---

### 🎯 Project Objectives

This project aims to:

1. **Develop an accurate emotion classifier** capable of distinguishing between 8 distinct emotional states
2. **Create an accessible web interface** for real-time emotion detection from voice recordings
3. **Demonstrate practical applications** of SER technology across various domains
4. **Provide educational insights** into the dataset characteristics, model architecture, and performance metrics

---

### 🌟 Why Speech Emotion Recognition Matters

Speech carries far more than just words. It conveys emotions, intentions, and context through paralinguistic cues like tone, pitch, rhythm, and intensity. The ability to automatically recognize these emotional signals has transformative implications across multiple sectors:

#### **Social Impact**
- **Mental Health Monitoring**: Early detection of stress, depression, or anxiety through voice analysis in telehealth platforms
- **Accessibility**: Enhanced communication tools for individuals with autism spectrum disorder or emotional expression difficulties
- **Crisis Intervention**: Automated detection of distress in helpline calls to prioritize urgent cases

#### **Healthcare Applications**
- **Patient Care**: Monitoring emotional well-being of elderly or isolated patients through voice check-ins
- **Therapy Support**: Tracking emotional progress during psychotherapy sessions
- **Diagnostic Assistance**: Supporting diagnosis of mental health conditions through vocal biomarkers

#### **Economic Value**
- **Customer Service**: Real-time sentiment analysis to improve customer satisfaction and agent performance
- **Market Research**: Understanding consumer emotional responses to products and services
- **Call Center Optimization**: Routing calls based on customer emotional state and agent expertise

#### **Educational Benefits**
- **Adaptive Learning**: Educational systems that respond to student frustration, confusion, or engagement
- **Teacher Training**: Providing feedback on emotional tone during instruction
- **Student Well-being**: Monitoring classroom emotional climate for intervention

""")

# Application areas with icons
app_areas = [
    ("mental-health.png", "Mental Health Support", "Early detection of stress, depression, or anxiety in patient monitoring systems"),
    ("customer-service.png", "Customer Service", "Real-time sentiment analysis to improve customer satisfaction and agent performance"),
    ("education.png", "Education", "Adaptive learning systems that respond to student frustration or confusion"),
    ("automative.png", "Automotive Safety", "Detecting driver stress or fatigue to prevent accidents"),
    ("assistant.png", "Human-Computer Interaction", "Creating more empathetic and responsive AI assistants"),
    ("entertainment.png", "Entertainment", "Emotion-aware gaming and interactive media experiences"),
]

AREA_ICON_SIZE = 40

for icon, title, desc in app_areas:
    icon_path = assets_dir / icon
    icon_html = ""
    if icon_path.exists():
        import base64
        img_data = base64.b64encode(icon_path.read_bytes()).decode()
        icon_html = f'<img src="data:image/png;base64,{img_data}" width="{AREA_ICON_SIZE}" style="display:block;flex-shrink:0;">'
    else:
        icon_html = f'<div style="width:{AREA_ICON_SIZE}px;height:{AREA_ICON_SIZE}px;display:flex;align-items:center;justify-content:center;font-size:{AREA_ICON_SIZE}px;flex-shrink:0;">📊</div>'
    
    st.markdown(f"""
    <div class="app-area-row">
        <div class="app-area-icon">
            {icon_html}
        </div>
        <div class="app-area-content">
            <div class="app-area-title">{title}</div>
            <div class="app-area-desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""

---

### 📊 System Architecture

This web application implements a complete SER pipeline:

#### **1. Data Collection**
The model was trained on four widely-recognized emotional speech datasets:
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **TESS** (Toronto Emotional Speech Set)
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **SAVEE** (Surrey Audio-Visual Expressed Emotion)

Combined, these datasets provide:
- **7,442 audio samples** across 8 emotion categories
- **Balanced gender representation** (male and female speakers)
- **Diverse emotional intensities** (normal and strong)
- **Controlled recording conditions** for high-quality audio

#### **2. Model Architecture: Wav2Vec2**

**Why Wav2Vec2?**

This system employs Facebook's Wav2Vec2, a transformer-based architecture that revolutionized speech processing by:

- **Learning from raw audio**: Unlike traditional approaches requiring hand-crafted features (MFCC, spectrograms), Wav2Vec2 processes raw waveforms
- **Self-supervised pre-training**: Trained on 960 hours of unlabeled speech data, capturing rich acoustic representations
- **Transfer learning**: Fine-tuned specifically for emotion classification, achieving superior performance with limited labeled data
- **End-to-end processing**: Directly maps audio input to emotion labels without intermediate feature extraction

**Technical Advantages**:
- Captures both linguistic content and paralinguistic emotional cues
- Robust to variations in speaking style, accent, and recording quality
- State-of-the-art performance on emotion recognition benchmarks
- Efficient inference suitable for real-time applications

#### **3. Speech-to-Text Integration**

The system includes OpenAI's Whisper model for automatic speech recognition (ASR), enabling:
- Transcription of user input for context
- Multi-language support (currently optimized for English)
- Robust performance in noisy environments

#### **4. Interactive Web Interface**

Built with Streamlit, the interface provides:
- **Real-time audio recording** through browser microphone access
- **Instant emotion detection** with confidence scores
- **Chat-style interaction** for natural user experience
- **Educational content** about datasets, EDA, and model performance

---

### 🎭 Emotion Categories

The system classifies speech into **8 distinct emotional states**:

""")

# Emotion categories
emotion_rows = [
    ("happy.png", "Happy", "Joy, excitement, pleasure, and positive enthusiasm"),
    ("sad.png", "Sad", "Sorrow, disappointment, grief, and melancholy"),
    ("angry.png", "Angry", "Frustration, rage, annoyance, and hostility"),
    ("fear.png", "Fear", "Anxiety, terror, nervousness, and apprehension"),
    ("surprise.png", "Surprise", "Shock, amazement, astonishment, and wonder"),
    ("disgust.png", "Disgust", "Revulsion, distaste, contempt, and aversion"),
    ("neutral.png", "Neutral", "Calm, emotionless state without strong affect"),
    ("calm.png", "Calm", "Peaceful, relaxed, composed, and serene state"),
]

ICON_SIZE = 40

for icon, name, desc in emotion_rows:
    icon_path = assets_dir / icon
    icon_html = ""
    if icon_path.exists():
        import base64
        img_data = base64.b64encode(icon_path.read_bytes()).decode()
        icon_html = f'<img src="data:image/png;base64,{img_data}" width="{ICON_SIZE}" style="display:block;flex-shrink:0;">'
    else:
        # Emoji fallbacks
        emoji_map = {
            "happy.png": "😊", "sad.png": "😢", "angry.png": "😠", 
            "fear.png": "😨", "surprise.png": "😲", "disgust.png": "🤢",
            "neutral.png": "😐", "calm.png": "😌"
        }
        emoji = emoji_map.get(icon, "🙂")
        icon_html = f'<div style="width:{ICON_SIZE}px;height:{ICON_SIZE}px;display:flex;align-items:center;justify-content:center;font-size:{ICON_SIZE}px;flex-shrink:0;">{emoji}</div>'
    
    st.markdown(f"""
    <div class="emotion-row">
        <div class="emotion-icon">
            {icon_html}
        </div>
        <div class="emotion-content">
            <div class="emotion-name">{name}</div>
            <div class="emotion-desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
---

### 🚀 How to Use This Application

Experience the emotion recognition system in four simple steps:

#### **Step 1: Navigate to Prediction**
- Click **"Prediction"** in the sidebar menu
- You'll enter an interactive chat interface designed for seamless emotion detection

#### **Step 2: Record Your Voice**
- Click **"Start Recording"** at the bottom of the page
- Speak naturally for 2-10 seconds—try expressing different emotions!
- Click **"Stop"** when finished
- The system works best with clear speech in a quiet environment

#### **Step 3: Automatic Processing**
The system will:
1. **Transcribe** your speech using Whisper AI (Speech-to-Text)
2. **Extract** acoustic features from your voice
3. **Classify** the emotion using the fine-tuned Wav2Vec2 model
4. **Display** the detected emotion with confidence score

#### **Step 4: Explore and Experiment**
- Try speaking the same phrase with different emotional tones
- Notice how pitch, speaking rate, and intensity affect predictions
- Test edge cases: sarcasm, mixed emotions, or subtle expressions
- Compare results across different sentence lengths and complexity

---

### 💡 Tips for Optimal Results

**For Accurate Detection:**
- Speak **clearly and naturally**—avoid forcing or exaggerating emotions (unless testing extremes)
- Use a **quiet environment** to minimize background noise and echo
- Maintain **consistent distance** from the microphone (30-60 cm recommended)
- Speak for **3-7 seconds**—longer utterances provide more emotional context

**For Experimentation:**
- Try **contrasting emotions**: say "I'm so happy" with angry vs. happy tone
- Test **neutral content** with emotional delivery: "The weather is nice" (said angrily, sadly, etc.)
- Experiment with **speaking rate**: fast (excited) vs. slow (sad/calm)
- Vary **volume/intensity**: loud (angry) vs. soft (calm/sad)

**Understanding Limitations:**
- The model was trained primarily on **English** speakers
- Performance may vary with **strong accents** or non-native speakers
- **Sarcasm and irony** can be challenging (emotion conflicts with content)
- **Background noise** reduces accuracy—use headset microphone if available

---

### 🛠️ Technical Implementation

**Frontend Framework:**
- **Streamlit**: Rapid prototyping, easy deployment, and interactive widgets
- **st_audiorec**: Browser-based audio recording without additional plugins

**Backend AI Models:**
- **Wav2Vec2** (Hugging Face): Fine-tuned on emotional speech datasets
- **Whisper** (OpenAI): Automatic speech recognition for transcription
- **PyTorch**: Deep learning framework for model inference

**Audio Processing:**
- **Librosa**: Feature extraction and audio preprocessing
- **Torchaudio**: PyTorch-native audio transformations
- **Soundfile**: Audio I/O operations

**Deployment:**
- **Streamlit Cloud**: Hosted web application with GPU support
- **GitHub**: Version control and continuous deployment

---

### 📚 Application Pages

Explore different aspects of the project through the sidebar:

**🏠 Home** (Current Page)
- Project introduction and motivation
- System architecture overview
- Usage instructions and tips

**📊 Data Collection**
- Detailed dataset descriptions (RAVDESS, TESS, CREMA-D, SAVEE)
- Sample statistics and distribution
- Data preprocessing pipeline

**📈 Exploratory Data Analysis (EDA)**
- Emotion class distribution and balance
- Audio duration statistics
- Acoustic feature visualizations (waveforms, spectrograms, MFCCs)
- Gender and intensity analysis

**🤖 Models**
- Wav2Vec2 architecture explanation
- Training process and hyperparameters
- Performance metrics (accuracy, F1-score, confusion matrix)
- Model comparison and ablation studies

**🎤 Prediction**
- Interactive emotion recognition interface
- Real-time audio recording and processing
- Emotion results with confidence scores
- Chat-based conversation history

---

### 🎯 Project Outcomes

**Achievements:**
- Successfully trained a multi-class emotion classifier with **>85% accuracy** on test data
- Deployed an accessible web application for real-time emotion detection
- Comprehensive documentation of datasets, methodology, and results
- Demonstrated practical applications across healthcare, education, and customer service

**Future Enhancements:**
- **Multi-language support**: Extend to Spanish, Mandarin, and other languages
- **Real-time streaming**: Process audio continuously rather than single recordings
- **Emotion intensity**: Detect not just emotion type but also strength (mild vs. strong)
- **Multi-modal fusion**: Combine audio with facial expressions for improved accuracy
- **Personalization**: Adapt model to individual speakers over time

---

### 👨‍💻 About This Project

**Author**: Trinh Hoang Anh Hong  
**Institution**: University of Greenwich  
**Program**: BSc Computing (Final Year Project)  
**Academic Year**: 2024-2025  

**Project Goals**:
This final year project explores the capabilities and limitations of deep learning for emotion recognition in speech. By combining academic research with practical implementation, it demonstrates how modern AI can contribute to more emotionally intelligent human-computer interaction.

---

### 🎉 Ready to Get Started?

**Experience AI-powered emotion recognition now!**

1. Click **"Prediction"** in the sidebar
2. Press **"Start Recording"** and speak naturally
3. Watch as the system detects your emotion in real-time
4. Experiment with different tones and phrases

*Discover how your voice reveals emotions beyond words—start your journey into Speech Emotion Recognition today!*
""")

st.sidebar.success("👈 Select **Prediction** to test the model!")
st.sidebar.info("""
### 🎯 Quick Start Guide
1. **Click Prediction** in the menu
2. **Press Start Recording**
3. **Speak with emotion** (2-10 seconds)
4. **View results** instantly!
""")