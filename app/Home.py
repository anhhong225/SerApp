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

Speech Emotion Recognition (SER) is more than just a technical innovation — it has meaningful social, healthcare, economic, and educational impact. By enabling machines to understand human emotions, SER contributes to a future where technology is empathetic, inclusive, and supportive.

---

### 🌟 Why Speech Emotion Recognition Matters

Speech carries far more than just words—it conveys emotions, intentions, and context through tone, pitch, rhythm, and intensity. Recognizing these emotional cues enables:
""")

# Application areas with icons (using HTML for better mobile responsiveness)
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

### Project Overview

This application demonstrates a complete end-to-end Speech Emotion Recognition system built with state-of-the-art deep learning techniques.

#### **Dataset**
- Trained on diverse emotional speech datasets including RAVDESS, TESS, CREMA-D, and SAVEE
- Contains thousands of audio samples across multiple emotions
- Balanced representation of male and female speakers
- Various emotional intensities and speaking styles

#### **Model Architecture: Wav2Vec2**
- **Base Model**: Facebook's Wav2Vec2, a transformer-based architecture pre-trained on 960 hours of speech
- **Fine-tuning**: Specialized for emotion classification tasks
- **Advantages**: 
  - Learns directly from raw audio waveforms
  - Captures both linguistic and acoustic features
  - State-of-the-art performance on emotion recognition benchmarks

#### **Input & Output**

**Input**: 
- Raw audio recording (WAV format): English language
- Your voice speaking any phrase or sentence
- Duration: 2-10 seconds recommended

**Output**: 
- Detected emotion from **8 emotion categories**:
""")

# Emotion categories (using HTML for better mobile responsiveness)
emotion_rows = [
    ("happy.png", "Happy", "Joy, excitement, pleasure"),
    ("sad.png", "Sad", "Sorrow, disappointment, grief"),
    ("angry.png", "Angry", "Frustration, rage, annoyance"),
    ("fear.png", "Fear", "Anxiety, terror, nervousness"),
    ("surprise.png", "Surprise", "Shock, amazement, astonishment"),
    ("disgust.png", "Disgust", "Revulsion, distaste, contempt"),
    ("neutral.png", "Neutral", "Calm, emotionless state"),
    ("calm.png", "Calm", "Peaceful, relaxed, composed"),
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

### How to Use This Application

Follow these simple steps to test the emotion recognition model:

#### **Step 1: Navigate to Prediction**
- Click on **"Prediction"** in the sidebar menu
- This will take you to the interactive chatbot interface

#### **Step 2: Record Your Voice**
- Click the **"Start Recording"** button at the bottom of the page
- Speak naturally—try expressing different emotions!
- Click **"Stop"** when finished (2-5 seconds is ideal)

#### **Step 3: View Results**
- The system will automatically:
  1. Convert your speech to text using Whisper AI
  2. Analyze the emotional content of your voice
  3. Display the detected emotion with confidence

#### **Step 4: Experiment**
- Try different emotions: speak angrily, happily, sadly
- Notice how tone, pitch, and intensity affect the prediction
- Test with different phrases and speaking styles

---

### 💡 Tips for Best Results

- **Speak clearly** and naturally—don't force emotions
- **Use a quiet environment** to minimize background noise
- **Vary your tone** consciously to test different emotions
- **Longer phrases** (3-7 words) work better than single words
- **Exaggerate slightly** if you want clearer emotion detection

---

### 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Emotion Model** | Wav2Vec2 (Hugging Face Transformers) |
| **Speech-to-Text** | OpenAI Whisper |
| **Deep Learning** | PyTorch |
| **Audio Processing** | Librosa, Torchaudio |
| **Deployment** | Streamlit Cloud |

---

### 📚 Project Structure

This application is organized into several sections:

- **Home** (Current Page): Project introduction and instructions
- **Prediction**: Interactive emotion recognition chatbot

---

### 🎯 Getting Started

**Ready to try it out?**

1. Click **"Prediction"** in the sidebar
2. Follow the on-screen instructions
3. Start recording and have fun!

---

### 👨‍💻 About

**Author**: Trinh Hoang Anh Hong  
**Institution**: University of Greenwich  
**Program**: Computing (Final Year Project)  
**Date**: November 2025

---

*Experience the power of AI-driven emotion recognition—start your journey now by clicking Prediction in the sidebar!*
""")

st.sidebar.success("Select **Prediction** to test the model!")
st.sidebar.info("""
### Quick Start
1. Click **Prediction**
2. Press **Start Recording**
3. Speak with emotion
4. See the results!
""")