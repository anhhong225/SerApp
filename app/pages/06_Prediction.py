from platform import processor
from xml.parsers.expat import model
import streamlit as st
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
import tempfile
import whisper
from st_audiorec import st_audiorec
import os
from pathlib import Path
import toml
import sys
import numpy as np
import wave
from utils.chatbot_ui import load_css

st.set_page_config(page_title="Emotion Chatbot", page_icon="🎤", layout="centered")

# Load global CSS and chatbot-specific CSS
load_css("global.css", "chatbot.css")

# Function to get HF token from secrets or local file
def get_hf_token():
    """Get Hugging Face token from Streamlit secrets or local secrets file"""
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except:
        pass
    
    try:
        current_dir = Path(__file__).parent.parent
        secrets_path = current_dir / ".streamlit" / "secrets.toml"
        
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            if "HF_TOKEN" in secrets:
                return secrets["HF_TOKEN"]
    except Exception as e:
        st.error(f"Error loading secrets file: {e}")
    
    return None

def check_audio_has_speech(audio_path, threshold=0.01):
    """
    Check if audio file contains actual speech by analyzing amplitude.
    Returns True if audio has significant content, False if it's mostly silence.
    """
    try:
        with wave.open(audio_path, 'rb') as wf:
            # Read audio data
            frames = wf.readframes(wf.getnframes())
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Calculate RMS (Root Mean Square) energy
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Check if RMS is above threshold
            return rms > threshold
    except Exception as e:
        st.error(f"Error checking audio: {e}")
        return False

@st.cache_resource
def load_asr(model_name):
    return whisper.load_model(model_name)

asr_model = load_asr("base")

@st.cache_resource
def load_emotion_model():
    model_id = "anhhong225/wav2vec2-emotion"
    try:
        hf_token = get_hf_token()
        
        if not hf_token:
            st.error("Hugging Face token not found. Please add it to your secrets.toml file or Streamlit secrets.")
            st.info("Create a file at: app/.streamlit/secrets.toml with:\nHF_TOKEN = \"your_token_here\"")
            st.stop()

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, token=hf_token)
        model = AutoModelForAudioClassification.from_pretrained(model_id, token=hf_token)
        pipe = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

        return pipe
    except Exception as e:
        st.error(f"Failed to load emotion model. Please check the model ID and your token: {model_id}")
        st.exception(e)
        st.stop()

emotion_pipe = load_emotion_model()

# ----------------------------
#  STREAMLIT UI
# ----------------------------
st.title("🎧 Emotion Chatbot")
st.markdown("Click **Start** to record your voice")

# Initialize chat history and last audio tracker
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How are you feeling today?"}]
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Audio recorder at the bottom
wav_audio_data = st_audiorec()

# Check for new audio data to process
if wav_audio_data is not None and wav_audio_data != st.session_state.get("last_audio"):
    # Store the new audio data to prevent reprocessing
    st.session_state.last_audio = wav_audio_data

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_audio_data)
        tmp_path = tmp.name

    # Process and respond
    with st.spinner("Thinking..."):
        # 1. Check if audio actually contains speech (not just silence)
        has_speech = check_audio_has_speech(tmp_path, threshold=0.01)
        
        if not has_speech:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I didn't hear anything. Please speak clearly and try again."
            })
        else:
            # 2. Transcribe speech with options to reduce hallucination
            result = asr_model.transcribe(
                tmp_path, 
                fp16=False,
                language="en",
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.6  # Higher = more aggressive at detecting silence
            )
            text = result["text"].strip()
            
            # 3. Check if text is too generic (hallucination indicators)
            hallucination_phrases = [
                "thank you", "thanks for watching", "bye", "goodbye",
                ".", "", "you", "the", "a", "uh", "um", "hmm"
            ]
            
            # If text is a hallucination phrase or too short
            if text.lower() in hallucination_phrases or len(text) < 3:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I didn't catch that clearly. Could you please repeat?"
                })
            elif text:
                # 4. Predict emotion
                preds = emotion_pipe(tmp_path)
                preds = sorted(preds, key=lambda x: x['score'], reverse=True)
                top_pred = preds[0]
                emotion = f"**{top_pred['label']}**"

                # 5. Formulate response
                bot_reply = f"I sense you might be feeling {emotion}."
                
                # 6. Update chat history
                st.session_state.messages.append({"role": "user", "content": text})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "I couldn't understand that. Please try again."
                })

    # Clean up the temporary file
    os.remove(tmp_path)

    # Rerun the app to display the new message
    st.rerun()