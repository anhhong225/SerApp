import streamlit as st
from pathlib import Path

def load_css():
    """Load custom CSS from the assets folder"""
    css_file = Path(__file__).parent.parent / "assets" / "chatbot.css"
    
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at {css_file}")

def render_chat_header(title: str, subtitle: str):
    """Render the chat header with title and subtitle"""
    st.title(title)
    st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)

def initialize_chat_session():
    """Initialize chat session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How are you feeling today?"}
        ]
    if "last_audio" not in st.session_state:
        st.session_state.last_audio = None

def display_chat_history():
    """Display all messages in the chat history"""
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def add_message(role: str, content: str):
    """Add a message to the chat history"""
    st.session_state.messages.append({"role": role, "content": content})