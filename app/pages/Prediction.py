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
from utils.chatbot_ui import load_css, render_chat_header, initialize_chat_session, display_chat_history, add_message

st.set_page_config(page_title="Emotion Chatbot", page_icon="🎤", layout="centered")

# Load global CSS and chatbot-specific CSS
load_css("global.css", "chatbot.css")
# Function to get HF token from secrets or local file
def get_hf_token():
    """Get Hugging Face token from Streamlit secrets or local secrets file"""
    try:
        # First, try to get from Streamlit secrets (works in deployment)
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except:
        pass
    
    # If not found, try to load from local secrets.toml file
    try:
        # Get the directory where this script is located
        current_dir = Path(__file__).parent.parent  # Go up to app directory
        secrets_path = current_dir / ".streamlit" / "secrets.toml"
        
        if secrets_path.exists():
            secrets = toml.load(secrets_path)
            if "HF_TOKEN" in secrets:
                return secrets["HF_TOKEN"]
    except Exception as e:
        st.error(f"Error loading secrets file: {e}")
    
    # If still not found, return None
    return None

# Load whisper (speech-to-text)
st.sidebar.title("Settings")
st.sidebar.markdown("**Speech-to-text model:** base")
st.sidebar.markdown("---")

@st.cache_resource
def load_asr(model_name):
    return whisper.load_model(model_name)

asr_model = load_asr("base")

# Load emotion recognition model from Hugging Face
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
st.markdown("Click the microphone to record your voice, and the bot will analyze your emotion.")

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

    # Display user's recorded audio
    st.chat_message("user").audio(wav_audio_data, format='audio/wav')

    # Process and respond
    with st.spinner("Thinking..."):
        # 1. Transcribe speech
        result = asr_model.transcribe(tmp_path, fp16=False) # Set fp16=False if not using a GPU
        text = result["text"].strip()

        if text:
            # 2. Predict emotion
            preds = emotion_pipe(tmp_path)
            # The output format of the pipeline might be a list of lists
            # We sort by score to get the most likely emotion
            preds = sorted(preds, key=lambda x: x['score'], reverse=True)
            top_pred = preds[0]
            emotion = f"**{top_pred['label']}** (confidence: {top_pred['score']:.2f})"

            # 3. Formulate and display response
            bot_reply = f"You said: *'{text}'*\n\nI sense you might be feeling {emotion}."
        else:
            bot_reply = "I didn't catch that. Could you please try again?"
        
        # Update chat history
        st.session_state.messages.append({"role": "user", "content": f"(Audio recording)"})
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Clean up the temporary file
    os.remove(tmp_path)

    # Rerun the app to display the new message
    st.rerun()