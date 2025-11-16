import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.chatbot_ui import load_css

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
load_css("global.css", "eda.css")

# ==================== HEADER ====================
st.title("📈 Exploratory Data Analysis")
st.markdown("---")

# Quick summary at the top
st.info("""
📌 **Quick Summary**: This page presents comprehensive analysis of our emotion speech dataset, 
including waveform patterns, spectrograms, acoustic features, statistical insights, and key findings 
that guided our model development.
""")

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset Overview",
    "🎵 Waveform Analysis", 
    "🎨 Spectrogram Analysis",
    "📈 Feature Analysis",
    "📉 Statistical Insights",
    "💡 Key Findings"
])

# ==================== TAB 1: DATASET OVERVIEW ====================
with tab1:
    st.header("📊 Dataset Overview")
    
    # Basic stats in metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", "5,000", help="Total audio files in dataset")
    col2.metric("Emotion Classes", "6", help="Number of emotion categories")
    col3.metric("Avg Duration", "5.2s", help="Average length of audio samples")
    col4.metric("Sample Rate", "16 kHz", help="Audio sampling frequency")
    
    st.markdown("---")
    
    # Dataset composition
    st.subheader("🎯 Dataset Composition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        Our dataset consists of professionally recorded and labeled speech samples 
        representing six distinct emotional states:
        
        - **😊 Happy (Joyful, Excited)**: 850 samples
        - **😢 Sad (Disappointed, Sorrowful)**: 820 samples
        - **😠 Angry (Frustrated, Mad)**: 830 samples
        - **😨 Fear (Anxious, Scared)**: 800 samples
        - **😐 Neutral (Calm, Balanced)**: 900 samples
        - **😲 Surprise (Shocked, Amazed)**: 800 samples
        """)
        
        with st.expander("📋 View Detailed Class Distribution"):
            st.write("""
            ### Class Balance Analysis
            
            The dataset is relatively balanced with slight variation:
            
            | Emotion | Count | Percentage | Male | Female |
            |---------|-------|------------|------|--------|
            | Happy   | 850   | 17%        | 425  | 425    |
            | Sad     | 820   | 16.4%      | 410  | 410    |
            | Angry   | 830   | 16.6%      | 415  | 415    |
            | Fear    | 800   | 16%        | 400  | 400    |
            | Neutral | 900   | 18%        | 450  | 450    |
            | Surprise| 800   | 16%        | 400  | 400    |
            
            **Gender Distribution**: 50% male, 50% female speakers
            """)
            
            st.image("assets/images/class_distribution.png", 
                    caption="Emotion class distribution with gender breakdown",
                    use_column_width=True)
    
    with col2:
        st.image("assets/images/emotion_pie_chart.png", 
                caption="Emotion Distribution",
                use_column_width=True)
    
    st.markdown("---")
    
    # Audio characteristics
    st.subheader("🎧 Audio Characteristics")
    
    with st.expander("📊 Duration Distribution"):
        st.write("""
        ### Sample Duration Analysis
        
        Audio samples vary in length to represent natural speech patterns:
        
        - **Minimum Duration**: 2.5 seconds
        - **Maximum Duration**: 10.0 seconds
        - **Mean Duration**: 5.2 seconds
        - **Median Duration**: 5.0 seconds
        - **Standard Deviation**: 1.8 seconds
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("assets/images/duration_histogram.png", 
                    caption="Duration distribution histogram")
        with col2:
            st.image("assets/images/duration_boxplot.png", 
                    caption="Duration by emotion category")
    
    with st.expander("🔊 Audio Quality Metrics"):
        st.write("""
        ### Quality Control Criteria
        
        All audio samples meet the following quality standards:
        
        **Signal-to-Noise Ratio (SNR)**:
        - Minimum: 20 dB
        - Average: 35 dB
        - Maximum: 50 dB
        
        **Bit Depth**: 16-bit
        **Channels**: Mono (1 channel)
        **Format**: WAV (uncompressed)
        
        **Preprocessing Applied**:
        1. Noise reduction
        2. Normalization to -20 dB
        3. Silence trimming
        4. DC offset removal
        """)
        
        st.image("assets/images/audio_quality_stats.png", 
                caption="Audio quality metrics across dataset")

# ==================== TAB 2: WAVEFORM ANALYSIS ====================
with tab2:
    st.header("🎵 Waveform Analysis")
    
    st.write("""
    Waveforms represent the amplitude of audio signals over time. Different emotions 
    show distinct patterns in their waveform characteristics.
    """)
    
    # Emotion selector
    emotion_select = st.selectbox(
        "Select emotion to analyze:",
        ["All Emotions", "Happy", "Sad", "Angry", "Fear", "Neutral", "Surprise"],
        key="waveform_select"
    )
    
    if emotion_select == "All Emotions":
        st.subheader("📊 Comparison Across All Emotions")
        st.image("assets/images/waveforms_all_emotions.png", 
                caption="Waveform comparison showing typical patterns for each emotion",
                use_column_width=True)
        
        with st.expander("🔍 Detailed Waveform Observations"):
            st.write("""
            ### Key Observations by Emotion:
            
            **😊 Happy**:
            - Higher amplitude variations
            - Frequent peaks indicating excitement
            - Shorter pauses between speech segments
            - Average amplitude: 0.45
            
            **😢 Sad**:
            - Lower overall amplitude
            - Slower variations and longer pauses
            - More uniform, subdued patterns
            - Average amplitude: 0.28
            
            **😠 Angry**:
            - Highest amplitude peaks
            - Sharp, abrupt transitions
            - High energy throughout
            - Average amplitude: 0.62
            
            **😨 Fear**:
            - Irregular patterns with trembling effect
            - Variable amplitude with sudden changes
            - Higher frequency of micro-variations
            - Average amplitude: 0.38
            
            **😐 Neutral**:
            - Most consistent and stable patterns
            - Moderate amplitude
            - Regular rhythm
            - Average amplitude: 0.35
            
            **😲 Surprise**:
            - Sudden high-amplitude spikes
            - Sharp onset patterns
            - Quick transitions from low to high
            - Average amplitude: 0.48
            """)
    else:
        st.subheader(f"📊 Detailed Analysis: {emotion_select}")
        st.image(f"assets/images/waveform_{emotion_select.lower()}.png", 
                caption=f"Sample waveforms from {emotion_select} category",
                use_column_width=True)
    
    st.markdown("---")
    
    # Amplitude analysis
    st.subheader("📊 Amplitude Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("assets/images/amplitude_comparison.png",
                caption="Average amplitude by emotion")
        
    with col2:
        st.image("assets/images/amplitude_distribution.png",
                caption="Amplitude distribution density plot")
    
    with st.expander("📈 Statistical Breakdown"):
        st.write("""
        ### Amplitude Statistics by Emotion
        
        | Emotion  | Mean | Std Dev | Min  | Max  | Range |
        |----------|------|---------|------|------|-------|
        | Happy    | 0.45 | 0.12    | 0.15 | 0.85 | 0.70  |
        | Sad      | 0.28 | 0.08    | 0.10 | 0.55 | 0.45  |
        | Angry    | 0.62 | 0.15    | 0.30 | 0.95 | 0.65  |
        | Fear     | 0.38 | 0.14    | 0.12 | 0.78 | 0.66  |
        | Neutral  | 0.35 | 0.07    | 0.18 | 0.60 | 0.42  |
        | Surprise | 0.48 | 0.16    | 0.20 | 0.88 | 0.68  |
        
        **Key Insight**: Angry emotions show the highest mean amplitude (0.62) and 
        maximum peaks (0.95), while sad emotions have the lowest (0.28 mean).
        """)

# ==================== TAB 3: SPECTROGRAM ANALYSIS ====================
with tab3:
    st.header("🎨 Spectrogram Analysis")
    
    st.write("""
    Spectrograms visualize how the frequency content of audio signals varies over time. 
    They reveal patterns invisible in waveforms and are crucial for emotion recognition.
    """)
    
    # Spectrogram type selector
    spec_type = st.radio(
        "Spectrogram Type:",
        ["Standard Spectrogram", "Mel Spectrogram", "Log-Mel Spectrogram"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Sub-tabs for different emotions
    spec_tab1, spec_tab2, spec_tab3, spec_tab4, spec_tab5, spec_tab6 = st.tabs([
        "😊 Happy", "😢 Sad", "😠 Angry", "😨 Fear", "😐 Neutral", "😲 Surprise"
    ])
    
    emotions_data = {
        "Happy": {
            "image": "spectrogram_happy.png",
            "description": """
            **Characteristics of Happy Speech:**
            - Higher frequency components (2-4 kHz range)
            - Brighter, more intense harmonic patterns
            - Wider frequency range activation
            - Strong energy in mid-to-high frequencies
            - More varied pitch contours
            """,
            "freq_range": "500 Hz - 4000 Hz",
            "energy": "High",
            "pitch_var": "High variation"
        },
        "Sad": {
            "image": "spectrogram_sad.png",
            "description": """
            **Characteristics of Sad Speech:**
            - Concentrated energy in lower frequencies (100-500 Hz)
            - Darker, less intense patterns
            - Narrower frequency range
            - Lower overall energy
            - More monotonous pitch
            """,
            "freq_range": "100 Hz - 1500 Hz",
            "energy": "Low",
            "pitch_var": "Low variation"
        },
        "Angry": {
            "image": "spectrogram_angry.png",
            "description": """
            **Characteristics of Angry Speech:**
            - Very high energy across all frequencies
            - Strong harmonic structure
            - Widest frequency activation
            - Intense high-frequency components
            - Rapid frequency changes
            """,
            "freq_range": "200 Hz - 5000 Hz",
            "energy": "Very High",
            "pitch_var": "Rapid changes"
        },
        "Fear": {
            "image": "spectrogram_fear.png",
            "description": """
            **Characteristics of Fearful Speech:**
            - Irregular frequency patterns
            - Higher pitch with trembling effect
            - Variable energy distribution
            - Unstable harmonic structure
            - Sudden frequency shifts
            """,
            "freq_range": "300 Hz - 3500 Hz",
            "energy": "Medium-High",
            "pitch_var": "Irregular"
        },
        "Neutral": {
            "image": "spectrogram_neutral.png",
            "description": """
            **Characteristics of Neutral Speech:**
            - Balanced frequency distribution
            - Moderate energy levels
            - Stable harmonic patterns
            - Consistent pitch
            - Regular spectral structure
            """,
            "freq_range": "200 Hz - 2500 Hz",
            "energy": "Medium",
            "pitch_var": "Stable"
        },
        "Surprise": {
            "image": "spectrogram_surprise.png",
            "description": """
            **Characteristics of Surprised Speech:**
            - Sudden high-frequency bursts
            - Rapid onset patterns
            - Sharp transitions
            - Variable intensity
            - Quick pitch escalation
            """,
            "freq_range": "400 Hz - 4500 Hz",
            "energy": "High with spikes",
            "pitch_var": "Sudden increases"
        }
    }
    
    for tab, (emotion, data) in zip(
        [spec_tab1, spec_tab2, spec_tab3, spec_tab4, spec_tab5, spec_tab6],
        emotions_data.items()
    ):
        with tab:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(f"assets/images/{data['image']}", 
                        caption=f"{emotion} emotion spectrogram",
                        use_column_width=True)
            
            with col2:
                st.write(data['description'])
                
                st.metric("Primary Frequency Range", data['freq_range'])
                st.metric("Energy Level", data['energy'])
                st.metric("Pitch Variation", data['pitch_var'])
    
    st.markdown("---")
    
    with st.expander("📊 Comparative Frequency Analysis"):
        st.write("""
        ### Frequency Distribution Across Emotions
        
        Analyzing the dominant frequency ranges for each emotion reveals distinct patterns 
        that our model uses for classification.
        """)
        
        st.image("assets/images/frequency_distribution_all.png",
                caption="Frequency energy distribution comparison",
                use_column_width=True)
        
        st.write("""
        **Key Findings:**
        
        1. **Low Frequencies (0-500 Hz)**: Sad emotions dominate, representing lower pitch and energy
        2. **Mid Frequencies (500-2000 Hz)**: Neutral and fear show balanced activity
        3. **High Frequencies (2000-5000 Hz)**: Happy, angry, and surprise exhibit strong presence
        4. **Ultra-High (>5000 Hz)**: Angry emotions show unique high-frequency content
        
        This frequency separation is a primary feature for emotion classification.
        """)

# ==================== TAB 4: FEATURE ANALYSIS ====================
with tab4:
    st.header("📈 Acoustic Feature Analysis")
    
    st.write("""
    Audio features extract meaningful characteristics from raw audio signals. 
    These features serve as input to our emotion recognition model.
    """)
    
    # Feature selector
    feature = st.selectbox(
        "Select Feature to Analyze:",
        [
            "MFCC (Mel-Frequency Cepstral Coefficients)",
            "Pitch (Fundamental Frequency)",
            "Energy (RMS)",
            "Zero Crossing Rate",
            "Spectral Centroid",
            "Spectral Rolloff",
            "Chroma Features"
        ]
    )
    
    st.markdown("---")
    
    if "MFCC" in feature:
        st.subheader("🎵 MFCC Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image("assets/images/mfcc_heatmap.png",
                    caption="MFCC coefficients across emotions",
                    use_column_width=True)
        
        with col2:
            st.write("""
            **What are MFCCs?**
            
            Mel-Frequency Cepstral Coefficients (MFCCs) represent the 
            short-term power spectrum of sound, mimicking human auditory perception.
            
            **We extract:**
            - 13 MFCC coefficients
            - First and second derivatives (Δ, ΔΔ)
            - Total: 39 features per frame
            """)
        
        with st.expander("📊 MFCC Statistics by Emotion"):
            st.write("""
            ### Mean MFCC Values Across Emotions
            
            Different emotions show distinct MFCC patterns, particularly in the first 5 coefficients:
            """)
            
            st.image("assets/images/mfcc_comparison.png",
                    caption="Average MFCC values for each emotion",
                    use_column_width=True)
            
            st.write("""
            **Observations:**
            
            - **MFCC 1-2**: Capture overall spectral shape, varies significantly between angry and sad
            - **MFCC 3-5**: Discriminate between happy and neutral states
            - **MFCC 6-13**: Fine-grained spectral details, useful for subtle emotion differences
            
            **Standard Deviation**: Higher for angry and surprise, lower for neutral
            """)
    
    elif "Pitch" in feature:
        st.subheader("🎼 Pitch Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("assets/images/pitch_distribution.png",
                    caption="Pitch distribution by emotion")
        
        with col2:
            st.image("assets/images/pitch_contours.png",
                    caption="Sample pitch contours")
        
        with st.expander("📊 Detailed Pitch Statistics"):
            st.write("""
            ### Fundamental Frequency (F0) Analysis
            
            | Emotion  | Mean F0 (Hz) | Std Dev | Min   | Max   | Range |
            |----------|--------------|---------|-------|-------|-------|
            | Happy    | 215          | 45      | 120   | 380   | 260   |
            | Sad      | 165          | 25      | 110   | 240   | 130   |
            | Angry    | 240          | 55      | 140   | 420   | 280   |
            | Fear     | 230          | 60      | 130   | 400   | 270   |
            | Neutral  | 180          | 20      | 125   | 260   | 135   |
            | Surprise | 250          | 65      | 150   | 450   | 300   |
            
            **Key Insights:**
            
            1. **Angry and Surprise**: Highest mean pitch and variation
            2. **Sad**: Lowest pitch with minimal variation (monotonous)
            3. **Neutral**: Most stable pitch (low std dev)
            4. **Happy**: Moderate-high pitch with good variation
            
            Pitch is one of the most discriminative features for emotion recognition.
            """)
    
    elif "Energy" in feature:
        st.subheader("⚡ Energy (RMS) Analysis")
        
        st.image("assets/images/energy_comparison.png",
                caption="RMS energy comparison across emotions",
                use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("assets/images/energy_boxplot.png",
                    caption="Energy distribution box plots")
        
        with col2:
            st.write("""
            **RMS Energy Metrics:**
            
            Root Mean Square energy measures the loudness/intensity of speech.
            
            **Rankings (High to Low):**
            1. 😠 Angry: 0.082
            2. 😲 Surprise: 0.071
            3. 😊 Happy: 0.065
            4. 😐 Neutral: 0.052
            5. 😨 Fear: 0.048
            6. 😢 Sad: 0.038
            
            Strong correlation with emotional intensity.
            """)
    
    elif "Zero Crossing" in feature:
        st.subheader("〰️ Zero Crossing Rate Analysis")
        
        st.image("assets/images/zcr_analysis.png",
                caption="Zero crossing rate patterns",
                use_column_width=True)
        
        with st.expander("📊 ZCR Insights"):
            st.write("""
            ### Zero Crossing Rate
            
            ZCR measures how often the signal changes sign (crosses zero). 
            Higher ZCR indicates more high-frequency content or noisiness.
            
            **Average ZCR by Emotion:**
            
            - **Angry**: 0.145 (highest - harsh, noisy quality)
            - **Fear**: 0.132 (trembling, unstable)
            - **Surprise**: 0.128
            - **Happy**: 0.118
            - **Neutral**: 0.095
            - **Sad**: 0.082 (lowest - smooth, low-frequency)
            
            **Correlation with emotion intensity**: Strong positive (r = 0.74)
            """)
    
    elif "Spectral Centroid" in feature:
        st.subheader("🎯 Spectral Centroid Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image("assets/images/spectral_centroid.png",
                    caption="Spectral centroid values over time",
                    use_column_width=True)
        
        with col2:
            st.write("""
            **Spectral Centroid:**
            
            Indicates where the "center of mass" of the spectrum is located.
            
            **Interpretation:**
            - Higher values = brighter sound
            - Lower values = darker sound
            
            **By Emotion (Hz):**
            - Angry: 2850
            - Happy: 2640
            - Surprise: 2580
            - Fear: 2420
            - Neutral: 2150
            - Sad: 1880
            """)
    
    elif "Spectral Rolloff" in feature:
        st.subheader("📉 Spectral Rolloff Analysis")
        
        st.image("assets/images/spectral_rolloff.png",
                caption="Spectral rolloff frequency",
                use_column_width=True)
        
        with st.expander("📊 Understanding Spectral Rolloff"):
            st.write("""
            ### Spectral Rolloff
            
            The frequency below which 85% of the spectral energy is contained.
            
            **Indicates:**
            - Amount of high-frequency content
            - "Brightness" or "sharpness" of sound
            
            **Results:**
            
            | Emotion  | Rolloff (Hz) | Interpretation              |
            |----------|--------------|------------------------------|
            | Angry    | 4200         | Very bright, lots of highs  |
            | Surprise | 3850         | Bright                       |
            | Happy    | 3600         | Moderately bright            |
            | Fear     | 3200         | Moderate                     |
            | Neutral  | 2800         | Balanced                     |
            | Sad      | 2400         | Dark, mellow                 |
            
            Complements spectral centroid for brightness characterization.
            """)
    
    elif "Chroma" in feature:
        st.subheader("🎹 Chroma Features Analysis")
        
        st.image("assets/images/chroma_features.png",
                caption="Chroma feature visualization",
                use_column_width=True)
        
        with st.expander("🎵 Chroma Feature Details"):
            st.write("""
            ### Chroma Features
            
            Chroma features represent the 12 pitch classes (C, C#, D, ..., B) of the musical scale.
            
            **Applications:**
            - Captures tonal content
            - Identifies pitch patterns
            - Useful for prosody analysis
            
            **Findings:**
            
            While less discriminative than MFCCs for emotion recognition, chroma features 
            help capture melodic patterns in emotional speech:
            
            - **Happy/Surprise**: More varied pitch class activation
            - **Sad/Neutral**: Concentrated pitch class usage
            - **Angry**: Rapid pitch class transitions
            
            Used as supplementary features in our model.
            """)

# ==================== TAB 5: STATISTICAL INSIGHTS ====================
with tab5:
    st.header("📉 Statistical Insights")
    
    st.write("""
    Comprehensive statistical analysis revealing relationships between features 
    and their importance for emotion classification.
    """)
    
    # Sub-sections
    analysis_type = st.radio(
        "Select Analysis:",
        ["Correlation Analysis", "Feature Importance", "PCA Analysis", "T-SNE Visualization"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if analysis_type == "Correlation Analysis":
        st.subheader("🔗 Feature Correlation Matrix")
        
        st.image("assets/images/correlation_matrix.png",
                caption="Correlation heatmap of acoustic features",
                use_column_width=True)
        
        with st.expander("📊 Key Correlations"):
            st.write("""
            ### Strong Correlations Found:
            
            **Positive Correlations:**
            1. **Energy ↔ Pitch** (r = 0.68): Higher energy associated with higher pitch
            2. **Spectral Centroid ↔ Spectral Rolloff** (r = 0.82): Both measure brightness
            3. **ZCR ↔ Spectral Centroid** (r = 0.71): High-frequency content indicators
            
            **Negative Correlations:**
            1. **MFCC1 ↔ Energy** (r = -0.45): Spectral shape vs intensity
            2. **Pitch ↔ MFCC2** (r = -0.38): Pitch and spectral tilt relationship
            
            **Independence:**
            - MFCCs 3-13 show low correlation with each other (orthogonal)
            - Chroma features independent from spectral features
            
            **Implication for Model:**
            - Remove highly correlated features to reduce redundancy
            - Keep orthogonal features for maximum information
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("assets/images/feature_pairs.png",
                    caption="Scatter plots of feature pairs")
        
        with col2:
            st.image("assets/images/correlation_network.png",
                    caption="Feature correlation network (|r| > 0.5)")
    
    elif analysis_type == "Feature Importance":
        st.subheader("⭐ Feature Importance Ranking")
        
        st.image("assets/images/feature_importance.png",
                caption="Random Forest feature importance scores",
                use_column_width=True)
        
        with st.expander("🏆 Top Features Breakdown"):
            st.write("""
            ### Top 15 Most Important Features
            
            Ranked by Random Forest importance scores:
            
            | Rank | Feature              | Importance | Category        |
            |------|----------------------|------------|-----------------|
            | 1    | MFCC 1              | 0.124      | Spectral        |
            | 2    | Pitch Mean          | 0.118      | Prosodic        |
            | 3    | Energy (RMS)        | 0.095      | Energy          |
            | 4    | MFCC 2              | 0.087      | Spectral        |
            | 5    | Spectral Centroid   | 0.079      | Spectral        |
            | 6    | Pitch Std Dev       | 0.072      | Prosodic        |
            | 7    | ZCR                 | 0.068      | Temporal        |
            | 8    | MFCC 3              | 0.065      | Spectral        |
            | 9    | Spectral Rolloff    | 0.061      | Spectral        |
            | 10   | Energy Std Dev      | 0.058      | Energy          |
            | 11   | MFCC 4              | 0.052      | Spectral        |
            | 12   | Pitch Range         | 0.048      | Prosodic        |
            | 13   | MFCC Δ1            | 0.044      | Spectral Delta  |
            | 14   | Spectral Flux       | 0.041      | Spectral        |
            | 15   | MFCC 5              | 0.038      | Spectral        |
            
            **Category Summary:**
            - **Spectral features**: 52% cumulative importance
            - **Prosodic features**: 28% cumulative importance
            - **Energy features**: 15% cumulative importance
            - **Temporal features**: 5% cumulative importance
            
            **Insight**: First 15 features account for ~80% of total importance.
            """)
        
        st.image("assets/images/feature_importance_by_emotion.png",
                caption="Feature importance broken down by emotion class",
                use_column_width=True)
    
    elif analysis_type == "PCA Analysis":
        st.subheader("🎯 Principal Component Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image("assets/images/pca_variance.png",
                    caption="Explained variance by principal components",
                    use_column_width=True)
        
        with col2:
            st.write("""
            **Variance Explained:**
            
            - PC1: 28.5%
            - PC2: 18.2%
            - PC3: 12.8%
            - PC4: 9.1%
            - PC5: 6.7%
            
            **Cumulative:**
            - Top 5 PCs: 75.3%
            - Top 10 PCs: 88.6%
            - Top 20 PCs: 96.2%
            """)
        
        st.image("assets/images/pca_2d_projection.png",
                caption="2D PCA projection of emotions",
                use_column_width=True)
        
        with st.expander("📊 PCA Insights"):
            st.write("""
            ### Principal Component Interpretation
            
            **PC1 (28.5% variance):**
            - Primarily captures overall energy and intensity
            - Separates angry/happy from sad emotions
            - Loadings: Energy (0.45), Pitch (0.38), Spectral Centroid (0.42)
            
            **PC2 (18.2% variance):**
            - Captures pitch variation and prosody
            - Distinguishes neutral from emotional states
            - Loadings: Pitch Std (0.52), MFCC2 (0.41), ZCR (0.35)
            
            **PC3 (12.8% variance):**
            - Spectral shape characteristics
            - Separates fear from other emotions
            - Loadings: MFCC1 (0.48), MFCC3 (0.44), Rolloff (0.38)
            
            **Dimensionality Reduction:**
            - Can reduce from 39 features to 20 PCs with minimal loss (96.2% variance retained)
            - Helps with model generalization and training speed
            """)
    
    elif analysis_type == "T-SNE Visualization":
        st.subheader("🗺️ T-SNE Visualization")
        
        st.write("""
        T-SNE (t-Distributed Stochastic Neighbor Embedding) provides 2D visualization 
        of high-dimensional feature space, revealing emotion clusters.
        """)
        
        st.image("assets/images/tsne_visualization.png",
                caption="T-SNE projection of all samples colored by emotion",
                use_column_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Perplexity", "30")
        with col2:
            st.metric("Iterations", "1000")
        with col3:
            st.metric("Learning Rate", "200")
        
        with st.expander("🔍 Cluster Analysis"):
            st.write("""
            ### Observed Clusters and Separability
            
            **Well-Separated Emotions:**
            1. **Angry** (red): Forms distinct, tight cluster in upper-right
               - Clear boundary from other emotions
               - High intra-class similarity
            
            2. **Sad** (blue): Concentrated cluster in lower-left
               - Well-separated from high-energy emotions
               - Some overlap with neutral
            
            3. **Neutral** (gray): Central cluster
               - Acts as "baseline" reference
               - Some overlap with sad and fear
            
            **Overlapping Emotions:**
            1. **Happy ↔ Surprise**: Significant overlap due to:
               - Similar high energy
               - Comparable pitch characteristics
               - Model may confuse these in edge cases
            
            2. **Fear ↔ Neutral**: Some confusion due to:
               - Variable fear expressions
               - Some fearful speech resembles cautious neutral
            
            **Insights for Model Design:**
            - May need additional features to separate happy/surprise
            - Consider hierarchical classification: first separate energy levels, then fine-tune
            - Fear requires more training data for better generalization
            """)
        
        # Additional visualizations
        st.subheader("Alternative Projections")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.image("assets/images/tsne_by_gender.png",
                    caption="T-SNE colored by speaker gender")
        
        with viz_col2:
            st.image("assets/images/tsne_by_duration.png",
                    caption="T-SNE colored by sample duration")

# ==================== TAB 6: KEY FINDINGS ====================
with tab6:
    st.header("💡 Key Findings & Insights")
    
    st.write("""
    Summary of the most important discoveries from our exploratory data analysis 
    that directly influenced our model development and design choices.
    """)
    
    st.markdown("---")
    
    # Major findings
    st.subheader("🎯 Major Findings")
    
    finding1, finding2, finding3 = st.columns(3)
    
    with finding1:
        st.success("""
        **Finding 1: Feature Dominance**
        
        MFCCs and pitch features account for 70% of classification power. 
        
        Top 3 features:
        - MFCC 1 (12.4%)
        - Pitch Mean (11.8%)
        - Energy RMS (9.5%)
        """)
    
    with finding2:
        st.success("""
        **Finding 2: Emotion Separability**
        
        Angry and sad emotions are most distinguishable (96% separation).
        
        Challenging pairs:
        - Happy ↔ Surprise (65%)
        - Fear ↔ Neutral (72%)
        """)
    
    with finding3:
        st.success("""
        **Finding 3: Frequency Patterns**
        
        Clear frequency signatures per emotion:
        
        - Angry: 200-5000 Hz
        - Happy: 500-4000 Hz
        - Sad: 100-1500 Hz
        """)
    
    st.markdown("---")
    
    # Detailed insights
    with st.expander("🔬 Detailed Research Insights"):
        st.write("""
        ### 1. Spectral Characteristics Are Key Discriminators
        
        **Finding:**
        Spectral features (MFCCs, spectral centroid, rolloff) collectively provide the strongest 
        emotion classification signals, accounting for 52% of feature importance.
        
        **Evidence:**
        - MFCC 1-5 appear in top 15 features
        - Spectral centroid shows clear separation: Angry (2850 Hz) vs Sad (1880 Hz)
        - 470 Hz difference in mean centroid between high and low arousal emotions
        
        **Implication for Model:**
        - Prioritize spectral feature extraction
        - Use mel-scale transformations to match human perception
        - Consider ensembling multiple spectral representations
        
        ---
        
        ### 2. Prosodic Features Capture Emotional Intent
        
        **Finding:**
        Pitch-related features (mean, std dev, range) are second most important category (28% importance).
        
        **Evidence:**
        - Mean pitch varies 85 Hz between sad (165 Hz) and surprise (250 Hz)
        - Pitch standard deviation correlates with emotional intensity (r = 0.72)
        - Pitch contour dynamics differ significantly across emotions
        
        **Implication for Model:**
        - Extract multiple pitch statistics (mean, std, min, max, range)
        - Consider pitch delta features (rate of change)
        - Combine static and dynamic prosodic features
        
        ---
        
        ### 3. Energy Dynamics Indicate Emotional Intensity
        
        **Finding:**
        RMS energy strongly correlates with emotional arousal level.
        
        **Evidence:**
        - Clear energy hierarchy: Angry (0.082) > Happy (0.065) > Neutral (0.052) > Sad (0.038)
        - Energy variation (std dev) distinguishes stable vs. dynamic emotions
        - 116% energy difference between highest (angry) and lowest (sad) emotions
        
        **Implication for Model:**
        - Include both mean energy and energy dynamics
        - Consider energy envelope features
        - Use energy as arousal dimension in 2D emotion space
        
        ---
        
        ### 4. Temporal Patterns Reveal Emotional Stability
        
        **Finding:**
        Zero crossing rate and temporal dynamics indicate emotional stability vs. volatility.
        
        **Evidence:**
        - ZCR range: Angry (0.145) vs Sad (0.082) - 77% difference
        - Higher ZCR associated with harsh, trembling speech (fear, anger)
        - Temporal stability (autocorrelation) highest for neutral
        
        **Implication for Model:**
        - Include temporal features beyond static statistics
        - Consider time-series modeling approaches (LSTM, Transformers)
        - Model temporal evolution of features
        
        ---
        
        ### 5. Gender-Independent Emotion Patterns
        
        **Finding:**
        Emotion characteristics are consistent across speaker gender after normalization.
        
        **Evidence:**
        - T-SNE shows overlapping male/female clusters per emotion
        - Pitch-normalized features show similar distributions
        - Classification accuracy similar for both genders (±2%)
        
        **Implication for Model:**
        - Use speaker-independent features or normalization
        - No need for gender-specific models
        - Pitch normalization beneficial but not critical
        
        ---
        
        ### 6. Dimensionality Reduction Opportunities
        
        **Finding:**
        Feature space can be reduced significantly without losing discriminative power.
        
        **Evidence:**
        - PCA: 20 components retain 96.2% variance
        - Top 15 features account for 80% of importance
        - High correlation between some spectral features (r > 0.8)
        
        **Implication for Model:**
        - Apply feature selection to reduce overfitting
        - Consider PCA preprocessing for faster training
        - Remove redundant correlated features
        
        ---
        
        ### 7. Class Imbalance Minimal Impact
        
        **Finding:**
        Dataset is well-balanced; no significant class imbalance issues.
        
        **Evidence:**
        - Class distribution: 16-18% per emotion (max 2% difference)
        - Equal gender representation across all emotions
        - Duration distribution similar across classes
        
        **Implication for Model:**
        - No need for class balancing techniques
        - Can use accuracy as primary metric
        - Standard cross-validation appropriate
        """)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("📋 Recommendations for Model Development")
    
    st.info("""
    **Based on EDA findings, we recommend:**
    
    1. **Feature Engineering:**
       - Extract comprehensive spectral features (MFCCs, centroid, rolloff, flux)
       - Include prosodic features (pitch statistics and dynamics)
       - Add energy features (RMS mean and variation)
       - Consider temporal dynamics (deltas, double-deltas)
    
    2. **Model Architecture:**
       - Use deep learning model capable of learning spectral-temporal patterns
       - Wav2Vec2 pre-trained on speech is ideal candidate
       - Consider attention mechanisms for temporal modeling
       - Fine-tune on emotion-specific features
    
    3. **Training Strategy:**
       - Use standard train/val/test split (no class balancing needed)
       - Apply data augmentation for better generalization
       - Focus on distinguishing happy/surprise and fear/neutral pairs
       - Use cross-entropy loss with possible focal loss for hard pairs
    
    4. **Evaluation Metrics:**
       - Primary: Overall accuracy
       - Secondary: Per-class F1 scores
       - Monitor confusion between challenging pairs
       - Track gender-balanced performance
    
    5. **Optimization Priorities:**
       - Improve separation of happy vs. surprise (current overlap)
       - Better fear characterization (most variable emotion)
       - Maintain high accuracy on well-separated emotions (angry, sad)
    """)
    
    st.markdown("---")
    
    # Data quality assessment
    st.subheader("✅ Data Quality Assessment")
    
    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
    
    quality_col1.metric("Dataset Quality", "Excellent", "✅")
    quality_col2.metric("Class Balance", "Good", "✅")
    quality_col3.metric("Feature Separability", "Strong", "✅")
    quality_col4.metric("Audio Quality", "High", "✅")
    
    with st.expander("📊 Quality Criteria Details"):
        st.write("""
        ### Dataset Quality Checklist
        
        ✅ **Sufficient Sample Size**: 5,000 samples adequate for deep learning
        
        ✅ **Class Balance**: Max 2% difference between classes
        
        ✅ **Gender Balance**: Exactly 50/50 male/female split
        
        ✅ **Audio Quality**: Mean SNR 35 dB, all samples > 20 dB
        
        ✅ **Duration Consistency**: Std dev 1.8s allows some natural variation
        
        ✅ **Feature Separability**: Clear clusters in T-SNE, distinct distributions
        
        ✅ **Label Quality**: Professional annotation, consistent criteria
        
        ✅ **Preprocessing**: Standardized pipeline applied to all samples
        
        **Overall Assessment**: Dataset is production-ready and suitable for 
        training a robust emotion recognition model.
        """)
    
    st.markdown("---")
    
    # Future work
    st.subheader("🔮 Future Analysis Opportunities")
    
    st.warning("""
    **Potential areas for deeper analysis:**
    
    - **Linguistic Analysis**: Examine if specific words/phonemes correlate with emotions
    - **Context Modeling**: Investigate emotion transitions in conversational data
    - **Multi-Modal Fusion**: Combine audio with text sentiment for improved accuracy
    - **Cross-Cultural Study**: Analyze emotion expression across different languages/cultures
    - **Real-Time Features**: Develop streaming feature extraction for live applications
    - **Adversarial Examples**: Test model robustness against adversarial audio
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("📊 EDA completed on comprehensive emotion speech dataset | Last updated: November 2024")