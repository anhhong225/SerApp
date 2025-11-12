import streamlit as st
from pathlib import Path
import time

def load_css(*css_files):
    """
    Load one or more CSS files from the assets/css folder with cache busting
    """
    assets_dir = Path(__file__).parent.parent / "assets" / "css"
    
    # Add timestamp to force browser reload
    timestamp = int(time.time())
    
    for css_file in css_files:
        css_path = assets_dir / css_file
        
        if css_path.exists():
            try:
                with open(css_path, encoding='utf-8') as f:
                    css_content = f.read()
                
                # Inject with timestamp comment to bypass cache
                st.markdown(
                    f"<style>/* {css_file} - {timestamp} */\n{css_content}</style>", 
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error loading CSS {css_file}: {e}")
        else:
            st.warning(f"CSS file not found: {css_file}")

def render_chat_header(title: str, subtitle: str):
    """Render the chat header with title and subtitle"""
    st.markdown('<div class="chat-header">', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="chat-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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