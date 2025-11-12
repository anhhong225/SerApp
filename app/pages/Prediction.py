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
        # 1. Transcribe speech with options to reduce hallucination
        result = asr_model.transcribe(
            tmp_path, 
            fp16=False,
            language="en",  # Specify language to reduce hallucination
            condition_on_previous_text=False,  # Don't use context from previous audio
            temperature=0.0  # Use greedy decoding for more deterministic results
        )
        text = result["text"].strip()
        
        # 2. Check if audio is too short or text is too generic (hallucination indicators)
        audio_size = len(wav_audio_data)
        hallucination_phrases = [
            "thank you", "thanks for watching", "bye", "goodbye",
            ".", "", "you", "the", "a"
        ]
        
        # If audio is very small (less than 10KB) or text is a hallucination phrase
        if audio_size < 10000 or text.lower() in hallucination_phrases or len(text) < 2:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I didn't hear anything. Please speak clearly and try again."
            })
        elif text:
            # 3. Predict emotion
            preds = emotion_pipe(tmp_path)
            preds = sorted(preds, key=lambda x: x['score'], reverse=True)
            top_pred = preds[0]
            emotion = f"**{top_pred['label']}**"

            # 4. Formulate response
            bot_reply = f"I sense you might be feeling {emotion}."
            
            # 5. Update chat history
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