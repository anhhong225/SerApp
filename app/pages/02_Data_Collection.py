import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.chatbot_ui import load_css

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

load_css("global.css", "data_collection.css")

st.title("Data Collection")

# Quick summary banner
st.info("""
**Project Dataset Overview**: This Speech Emotion Recognition system was trained on four carefully selected, 
publicly available emotional speech datasets, providing a diverse and comprehensive foundation for emotion classification.
""")

st.markdown("---")

# Introduction
st.markdown("""
## Dataset Selection Rationale

The success of any machine learning model heavily depends on the quality and diversity of training data. 
For Speech Emotion Recognition (SER), this means collecting audio samples that:

- **Represent diverse emotional expressions** across multiple speakers
- **Include balanced gender representation** (male and female voices)
- **Contain controlled recording conditions** to minimize noise and artifacts
- **Cover various emotional intensities** (mild to strong expressions)
- **Provide sufficient samples per emotion** to avoid class imbalance

This project combines **four widely-recognized emotional speech datasets** to create a robust training corpus:

1. **RAVDESS** - Professional actors with high-quality recordings
2. **TESS** - Controlled female speech with clear emotional expressions
3. **CREMA-D** - Crowd-sourced diverse speakers with natural variations
4. **SAVEE** - Male speakers with distinct emotional portrayals

---
""")

# RAVDESS Dataset
st.markdown("""
### RAVDESS Dataset
**Ryerson Audio-Visual Database of Emotional Speech and Song**

The RAVDESS dataset is one of the most widely-used benchmarks in emotion recognition research. 
It features professional actors delivering scripted emotional speech with high production quality.

#### Key Characteristics

**Dataset Composition**:
- **Total Audio Samples**: 1,440 speech files
- **Speakers**: 24 professional actors (12 male, 12 female)
- **Age Range**: Young adults (20-35 years)
- **Recording Environment**: Professional studio with acoustic treatment

**Emotion Categories** (8 classes):
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

**Audio Specifications**:
- **Sample Rate**: 48 kHz (high-fidelity)
- **Bit Depth**: 24-bit (professional quality)
- **Channels**: Stereo (2 channels)
- **Format**: WAV (uncompressed, lossless)
- **Duration**: Approximately 3-5 seconds per clip

**Emotional Intensity**:
- Two levels: Normal and Strong
- Allows training models to detect subtle vs. pronounced emotions

**Lexical Content**:
- Two statements per speaker: "Kids are talking by the door" and "Dogs are sitting by the door"
- Reduces linguistic bias, focusing on emotional tone rather than semantic content

**Why RAVDESS?**
- High recording quality minimizes noise artifacts
- Balanced gender and emotion representation
- Professional actors ensure clear emotional expressions
- Widely cited in academic research for benchmarking

**Download**: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
---
""")

# TESS Dataset
st.markdown("""
### TESS Dataset
**Toronto Emotional Speech Set**

TESS provides a focused collection of emotional speech from female speakers, 
offering high-quality data for gender-specific emotion analysis.

#### Key Characteristics

**Dataset Composition**:
- **Total Audio Samples**: 2,800 speech files
- **Speakers**: 2 female actresses (one younger, one older)
- **Age Range**: 26 and 64 years old
- **Recording Environment**: Anechoic chamber (minimal echo/reverberation)

**Emotion Categories** (7 classes):
- Angry, Disgust, Fear, Happy, Pleasant Surprise, Sad, Neutral

**Audio Specifications**:
- **Sample Rate**: 24,414 Hz (originally 44.1 kHz downsampled)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Format**: WAV (uncompressed)
- **Duration**: Approximately 1-2 seconds per word

**Lexical Content**:
- 200 target words embedded in the carrier phrase "Say the word ___"
- Example: "Say the word happy" (spoken with various emotions)
- Controls for linguistic content while varying emotional tone

**Why TESS?**
- Age diversity (young vs. older speaker) captures age-related vocal variations
- Anechoic recording ensures clean audio without room acoustics
- Large vocabulary (200 words) provides lexical diversity
- Clear emotional contrasts make it ideal for model training

**Download**: [TESS on University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487)

---
""")

# CREMA-D Dataset
st.markdown("""
### CREMA-D Dataset
**Crowd-sourced Emotional Multimodal Actors Dataset**

CREMA-D is the largest dataset in this collection, featuring diverse speakers with varying 
ages, ethnicities, and emotional expression styles.

#### Key Characteristics

**Dataset Composition**:
- **Total Audio Samples**: 7,442 speech files
- **Speakers**: 91 actors (48 male, 43 female)
- **Age Range**: 20-74 years (includes older adults)
- **Ethnic Diversity**: African American, Asian, Caucasian, Hispanic, and Unspecified
- **Recording Environment**: Controlled studio setting

**Emotion Categories** (6 classes):
- Angry, Disgust, Fear, Happy, Sad, Neutral

**Audio Specifications**:
- **Sample Rate**: 16 kHz (standard for speech processing)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Format**: WAV (uncompressed)
- **Duration**: Approximately 2-3 seconds per clip

**Emotional Intensity**:
- Four levels: Low, Medium, High, and Unspecified
- Enables fine-grained emotion intensity modeling

**Lexical Content**:
- 12 sentences designed to be emotionally neutral
- Examples: "It's eleven o'clock", "That is exactly what happened"
- Avoids emotionally-charged words to isolate vocal tone

**Annotation Quality**:
- Crowd-sourced validation: Multiple human raters assessed emotional accuracy
- Includes perceived emotion labels (how listeners interpreted the emotion)
- Validation scores indicate how convincingly emotions were portrayed

**Why CREMA-D?**
- Largest speaker pool (91 actors) provides demographic diversity
- Age and ethnic variety improve model generalization
- Intensity levels support nuanced emotion detection
- Crowd-validation ensures perceptual accuracy

**Download**: [CREMA-D on GitHub](https://github.com/CheyneyComputerScience/CREMA-D)

---
""")

# SAVEE Dataset
st.markdown("""
### SAVEE Dataset
**Surrey Audio-Visual Expressed Emotion**

SAVEE complements the collection by providing male-only speech with clear emotional distinctions, 
recorded in a controlled laboratory environment.

#### Key Characteristics

**Dataset Composition**:
- **Total Audio Samples**: 480 speech files
- **Speakers**: 4 male native English speakers
- **Age Range**: Young adults (university students)
- **Recording Environment**: Visual media lab with noise-controlled conditions

**Emotion Categories** (7 classes):
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

**Audio Specifications**:
- **Sample Rate**: 44.1 kHz (CD quality)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Format**: WAV (uncompressed)
- **Duration**: Approximately 2-4 seconds per utterance

**Lexical Content**:
- 15 phonetically-balanced TIMIT sentences
- 3 emotion-loaded common phrases (e.g., "I'm so angry")
- 30 emotion-loaded words (e.g., "terrible", "wonderful")
- Total: 120 utterances × 4 speakers = 480 samples

**Recording Details**:
- Multi-angle video recordings (audio-visual dataset, audio used here)
- Speakers were native British English speakers (RP accent)
- Professional audio equipment for consistent quality

**Why SAVEE?**
- Male-focused complement to TESS (female-focused)
- Phonetically balanced sentences ensure acoustic diversity
- British English accent adds accent variation to dataset
- High-quality controlled recordings minimize noise

**Download**: [SAVEE on Kaggle](https://www.kaggle.com/datasets/barelydedicated/savee-database)

---
""")

# Combined Dataset Summary
st.markdown("""
## Combined Dataset Summary

By merging these four datasets, we create a comprehensive training corpus that balances:

| **Characteristic** | **Combined Dataset** |
|--------------------|---------------------|
| **Total Samples** | **13,174 audio files** |
| **Unique Speakers** | **121 actors** (65 male, 56 female) |
| **Emotion Classes** | **8 emotions** (merged taxonomy) |
| **Age Diversity** | 20-74 years (young adults to seniors) |
| **Ethnic Diversity** | Multiple ethnicities (primarily from CREMA-D) |
| **Accent Variation** | North American (RAVDESS, TESS, CREMA-D) + British (SAVEE) |
| **Sample Rates** | 16 kHz to 48 kHz (resampled to 16 kHz for consistency) |
| **Channels** | Mixed (converted to Mono for uniformity) |

### Emotion Taxonomy Harmonization

Since the datasets use slightly different emotion labels, we standardized them to **8 core emotions**:

| **Final Label** | **Source Datasets** |
|----------------|-------------------|
| **Neutral** | All datasets |
| **Calm** | RAVDESS only (merged with Neutral in some experiments) |
| **Happy** | All datasets |
| **Sad** | All datasets |
| **Angry** | All datasets |
| **Fear** | All datasets |
| **Disgust** | All datasets |
| **Surprise** | RAVDESS, TESS, SAVEE |

**Note**: "Calm" from RAVDESS is sometimes treated as a variant of "Neutral" or kept separate depending on the research focus.

---

## Dataset Strengths

This combined approach offers several advantages:

- **Demographic Diversity**: Wide age range (20-74), multiple ethnicities, balanced gender
- **Recording Quality**: Professional studios (RAVDESS, TESS) + controlled labs (CREMA-D, SAVEE)
- **Emotion Variety**: 8 distinct emotions with varying intensities
- **Speaker Pool**: 121 unique speakers reduces overfitting to individual voices
- **Accent Variation**: North American + British English accents
- **Lexical Diversity**: Scripted sentences, carrier phrases, emotional words
- **Intensity Levels**: From subtle (normal) to exaggerated (strong) emotional expressions

## Dataset Limitations

- **Language**: English-only (may not generalize to other languages)
- **Acted Emotions**: Professional actors may exaggerate emotions vs. natural speech
- **Cultural Bias**: Primarily Western emotional expressions (North American/British)
- **Accent Skew**: Limited representation of non-native speakers or diverse accents
- **Class Imbalance**: Some emotions (e.g., Disgust) may have fewer samples than others

---

## Data Availability and Licensing

All datasets used in this project are **publicly available** for research purposes:

- **RAVDESS**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
- **TESS**: Open access via University of Toronto repository
- **CREMA-D**: Available upon request from GitHub repository
- **SAVEE**: Freely available on Kaggle for research

**Ethical Considerations**:
- All participants provided informed consent for data collection
- No personally identifiable information (PII) is included
- Used solely for non-commercial academic research

---

## Download Links

| Dataset | Link | Size |
|---------|------|------|
| **RAVDESS** | [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | ~500 MB |
| **TESS** | [U of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487) | ~350 MB |
| **CREMA-D** | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) | ~2 GB |
| **SAVEE** | [Kaggle](https://www.kaggle.com/datasets/barelydedicated/savee-database) | ~200 MB |

**Total Combined Size**: Approximately **3 GB** (uncompressed WAV files)
            
""")
