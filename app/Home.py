import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.chatbot_ui import load_css

st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="wide"
)

load_css("global.css","home.css")

st.title("🎭 Emotion Recognition from Speech")

st.markdown("""
## Welcome to the Emotion Recognition System

This project aims to detect emotions from speech audio using deep learning techniques.

### Project Overview

This application demonstrates an end-to-end emotion recognition system that:
- Collects and processes audio data
- Performs exploratory data analysis on speech features
- Trains machine learning models for emotion classification
- Provides real-time emotion prediction through voice input

### How It Works

1. **Data Collection**: Gather audio samples with labeled emotions
2. **Exploratory Data Analysis**: Analyze patterns and features in the audio data
3. **Model Training**: Build and train deep learning models (Wav2Vec2)
4. **Prediction**: Use the trained model to predict emotions from new audio inputs

### Technologies Used

- **Streamlit**: Web application framework
- **Transformers (Hugging Face)**: Pre-trained models for audio classification
- **Whisper**: Speech-to-text conversion
- **PyTorch**: Deep learning framework
- **Librosa**: Audio processing

### Navigate Through the Pages

Use the sidebar to explore different sections of this project:
- 📊 **Data Collection**: Learn about the dataset
- 📈 **EDA**: Explore data analysis and visualizations
- 🤖 **Models**: View model architecture and training details
- 🎤 **Prediction**: Try the emotion chatbot yourself!

---

**Author**: Trinh Hoang Anh Hong  
**Date**: November 2025
""")

st.sidebar.success("Select a page above to get started.")