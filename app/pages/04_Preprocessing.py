import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.chatbot_ui import load_css

st.set_page_config(page_title="Preprocessing & Features", page_icon="⚙️", layout="wide")
load_css("global.css", "preprocessing.css")

st.title("Preprocessing, Batch Processing, and Feature Extraction")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Preprocessing",
    "Batch Preprocessing & Organization",
    "Feature Extraction & Analysis"
])

# ==================== TAB 1: PREPROCESSING ====================
with tab1:
    st.header("Preprocessing")
    st.write("""
    Standardize raw audio to ensure consistent inputs for training and analysis.
    """)

    st.subheader("Pipeline")
    st.code("""
    Raw Audio
      → Resample (16 kHz)
      → Mono conversion
      → Silence trimming (top_db=40)
      → Duration normalization (target ~4s)
      → Amplitude normalization (RMS)
      → Save preprocessed waveform
    """, language="text")

    st.subheader("Key Settings")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Rate", "16 kHz")
    col2.metric("Channels", "Mono")
    col3.metric("Target Duration", "≈3.5 s")
    col4.metric("Trim Threshold", "-40 dB")

    st.image("assets/images/preprocess/silence_trimming.png",
             caption="Example: Waveform before and after silence trimming",
             use_container_width=True)

# ==================== TAB 2: BATCH PREPROCESSING & ORGANIZATION ====================
with tab2:
    st.header("Batch Preprocessing and Dataset Organization")

    st.subheader("Batch Processing")
    st.write("""
    Apply preprocessing to all files across datasets with consistent parameters and logging.
    """)
    st.code("""
    for file in all_audio_files:
        audio, sr = load(file)
        audio = resample(audio, sr, target_sr=16000)
        audio = to_mono(audio)
        audio = trim_silence(audio, top_db=40)
        audio = normalize_duration(audio, target_seconds=3.5, sr=16000)
        audio = rms_normalize(audio, target_rms=0.1)
        save(audio, metadata)
    """, language="python")

    st.subheader("Dataset Organization")
    st.write("Organize preprocessed data by split and emotion.")
    st.code("""
    preprocessed_data/
    ├── train/
    │   ├── neutral/
    │   ├── happy/
    │   ├── sad/
    │   ├── angry/
    │   ├── fearful/
    │   ├── disgust/
    │   ├── surprised/
    │   └── calm/
    ├── val/
    │   └── (same structure)
    ├── test/
    │   └── (same structure)
    └── metadata/
        ├── train_labels.csv
        ├── val_labels.csv
        └── test_labels.csv
    """, language="text")

    st.subheader("Split Strategy")
    st.write("""
    Stratified by emotion and, where available, split by speaker to avoid leakage across splits.
    """)

# ==================== TAB 3: FEATURE EXTRACTION & ANALYSIS ====================
with tab3:
    st.header("Feature Extraction and Analysis")

    st.subheader("Feature Types")
    st.write("""
    - Mel-Spectrograms (log-mel) for analysis/visualization
    - MFCCs (optional baseline features)
    - Raw waveform features via Wav2Vec2 for model training
    """)

    st.subheader("Mel-Spectrogram Configuration")
    st.write("""
    Focus on the low–mid frequency range identified in EDA as most discriminative.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.code("""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=0,
            fmax=8000
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        """, language="python")
    with col2:
        st.info("""
        - n_mels: 128
        - fmax: 8000 Hz
        - Window: Hann
        - Emphasis: 200–3700 Hz region
        """)

    st.image("assets/images/preprocess/melspectrogram_examples.png",
             caption="Log-mel spectrogram examples across emotions",
             use_container_width=True)

    st.subheader("Discriminative Frequency Ranges (from EDA)")
    st.write("""
    - 200–800 Hz: strongest (F0, F1)
    - 1500–3700 Hz: important (F2, F3)
    - 0–3740 Hz: most significant overall
    - 300–2800 Hz: consistent across all datasets
    - >8000 Hz: diminishing returns, mostly noise
    """)

    st.subheader("Model Feature Path (Wav2Vec2)")
    st.write("""
    The model ingests raw waveforms and learns hierarchical features (low-level spectral → high-level contextual),
    eliminating manual feature engineering while aligning with the EDA findings on useful frequency ranges.
    """)

st.markdown("---")
st.caption("Preprocessing, Batch Processing, and Feature Extraction | Based on project Introduction.docx")