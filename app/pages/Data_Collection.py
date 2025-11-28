import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.chatbot_ui import load_css

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

load_css("global.css", "data_collection.css")

st.header("📊 Data Collection")
# Quick summary at the top
st.info("""📌 **Quick Summary**: This page presents the dataset used in this project.""")

st.markdown("---")
st.write("""
        ### RAVDESS Dataset Overview

        The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset is a widely used dataset for emotion recognition research. It contains a variety of emotional speech samples recorded from professional actors.

        **Key Features**:
        - **Total Samples**: 1,440
        - **Emotion Categories**: 8 (Calm, Happy, Sad, Angry, Fearful, Disgust, Surprise, and Neutral)
        - **Audio Characteristics**:
            - Sample Rate: 48 kHz
            - Bit Depth: 24-bit
            - Channels: Stereo (2 channels)
            - Format: WAV (uncompressed)
        - **Download Link**: [RAVDESS Dataset](https://zenodo.org/record/1188976)
        """)
st.markdown("---")
st.write("""
        ### TESS Dataset Overview
        The TESS (Toronto Emotional Speech Set) dataset is another popular dataset for emotion recognition tasks. It consists of speech samples recorded by two actresses portraying different emotions.
        **Key Features**:
        - **Total Samples**: 2,800
        - **Emotion Categories**: 7 (Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise)
        - **Audio Characteristics**:
            - Sample Rate: 44.1 kHz
            - Bit Depth: 16-bit
            - Channels: Mono (1 channel)
            - Format: WAV (uncompressed)
        - **Download Link**: [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)
        """)
st.markdown("---")
st.write("""
        ### CREMA-D Dataset Overview
        The CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) is a comprehensive dataset that includes audio-visual recordings of actors expressing various emotions.
        **Key Features**:
        - **Total Samples**: 7,442
        - **Emotion Categories**: 6 (Anger, Disgust, Fear, Happiness, Neutral, Sadness)
        - **Audio Characteristics**:
            - Sample Rate: 48 kHz
            - Bit Depth: 16-bit
            - Channels: Mono (1 channel)
            - Format: WAV (uncompressed)
        - **Download Link**: [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
        """)
st.markdown("---")
st.write("""
        ### SAVED Dataset Overview
        The SAVED (Speech Audio Video Emotion Database) dataset is a multimodal dataset that includes audio and video recordings of actors expressing different emotions.
        **Key Features**:
        - **Total Samples**: 480
        - **Emotion Categories**: 6 (Anger, Happiness, Sadness, Neutral, Fear, Disgust)
        - **Audio Characteristics**:
            - Sample Rate: 44.1 kHz
            - Bit Depth: 16-bit
            - Channels: Mono (1 channel)
            - Format: WAV (uncompressed)
        - **Download Link**: [SAVED Dataset](https://zenodo.org/record/1188974)
        """)
st.markdown("---")
st.write("""
        ### Overall Dataset Summary

        The combined dataset used in this project consists of samples from RAVDESS, TESS, and CREMA-D datasets. This diverse collection of emotional speech samples provides a robust foundation for training and evaluating emotion recognition models.

        **Combined Dataset Features**:
        - **Total Samples**: 11,682
        - **Emotion Categories**: 8 (Calm, Happy, Sad, Angry, Fearful, Disgust, Surprise, Neutral)
        - **Audio Formats**: WAV (uncompressed)
        - **Sample Rates**: 44.1 kHz and 48 kHz
        - **Bit Depths**: 16-bit and 24-bit
        - **Channels**: Mono and Stereo

        This dataset is well-suited for developing machine learning models capable of accurately recognizing emotions from speech audio.
        """)