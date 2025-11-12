import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.chatbot_ui import load_css

st.set_page_config(page_title="CSS Test", page_icon="🧪", layout="wide")

# Load ONLY the test CSS (not global)
load_css("global.css", "models.css")

st.title("CSS TEST PAGE")
st.write("If you see:")
st.write("- Orange background")
st.write("- White text on black background for title")
st.write("- Blue text for paragraphs")
st.write("Then CSS is working!")

st.markdown("---")
st.write("Check the SIDEBAR for debug information →")