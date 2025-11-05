import streamlit as st

st.set_page_config(page_title="Data Collection", page_icon="📊")

st.title("📊 Data Collection")

st.markdown("""
## Dataset Information

Describe your dataset here:
- Source of the data
- Number of samples
- Emotion categories
- Audio characteristics (sample rate, duration, etc.)

### Sample Distribution

You can add charts and visualizations here showing the distribution of emotions in your dataset.
""")

# Add your data collection visualizations and information here