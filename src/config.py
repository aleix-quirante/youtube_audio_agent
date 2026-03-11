import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- VALIDATE CRITICAL ENVIRONMENT VARIABLES ---
REQUIRED_KEYS = ["GEMINI_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]
if missing_keys:
    print(
        f"⚠️ WARNING: Missing critical environment variables: {', '.join(missing_keys)}"
    )
    # In a real production app we might raise an error here,
    # but for Streamlit we'll print a warning and let the UI handle specific failures.

# --- PATHS ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
SEPARATED_DIR = DATA_DIR / "separated"
AGENT_DB_DIR = DATA_DIR / "agent_db"

# Create directories if they don't exist
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SEPARATED_DIR.mkdir(parents=True, exist_ok=True)
AGENT_DB_DIR.mkdir(parents=True, exist_ok=True)

# --- CONSTANTS ---
PINECONE_INDEX_NAME = "youtube-agent-musical"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# DSP Constants
HOP_LENGTH = 512
INTENSITY_PERCENTILE = 65
MIN_FRAMES_SECONDS = 1.5
