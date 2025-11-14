import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main app
import subprocess
import os

# Change to app directory and run Home.py
os.chdir(app_dir)
subprocess.run([sys.executable, "-m", "streamlit", "run", "Home.py"])