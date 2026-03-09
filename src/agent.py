import os
import librosa
import numpy as np
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Internal import from your rag_chain module
from src.rag_chain import get_vectorstore

load_dotenv()


# --- HELPER: TIME FORMATTER ---
def format_timestamp(seconds: float) -> str:
    """Converts seconds to MM:SS format for the Sensei's report."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# --- TOOL 1: INTERNAL VIDEO SEARCH ---
@tool
def search_video_knowledge(query: str, music_only: bool = False) -> str:
    """
    Searches the internal video transcript for facts or music timestamps.
    Use music_only=True to focus only on sections where songs are playing.
    """
    print(f"🔍 Agent tool: Searching video for '{query}'")
    vectorstore = get_vectorstore()
    search_kwargs = {"k": 5}

    if music_only:
        search_kwargs["filter"] = {"is_music_piece": True}

    results = vectorstore.similarity_search(query, **search_kwargs)

    if not results:
        return "No specific information found in the video transcript."

    context = ""
    for doc in results:
        start_raw = doc.metadata.get("start", 0)
        timestamp = format_timestamp(start_raw)
        is_music = doc.metadata.get("is_music_piece", False)
        tag = "🎵 [MUSIC]" if is_music else "🗣️ [NARRATIVE]"
        context += f"--- {tag} at {timestamp} ---\n{doc.page_content}\n\n"

    return context


# --- TOOL 2: EXTERNAL MUSIC EXPERT ---
music_expert_search = TavilySearchResults(
    max_results=3,
    description="Search the internet for music history, artist facts, and production records.",
)


# --- TOOL 3: AUDIO ANALYZER ---
@tool
def get_audio_stats(timestamp: int) -> str:
    """
    Returns technical audio data (RMS energy, ZCR, BPM) for a specific second.
    Useful for validating drops, intensity, or percussive clarity.
    """
    # In production, fetch this from your analysis_map/Master JSON
    # MOCK LOGIC for demonstration:
    rms_val = 0.082
    zcr_val = 0.045

    # Simple logic to determine status
    if rms_val > 0.07 and zcr_val < 0.1:
        status = "Stable Percussion / High Energy"
    elif zcr_val > 0.2:
        status = "High Noise / Sibilance"
    else:
        status = "Ambient / Low Energy"

    return (
        f"Technical Analysis at {format_timestamp(timestamp)}:\n"
        f"- RMS Energy: {rms_val} (Punchy)\n"
        f"- ZCR: {zcr_val} (Clear/Percussive)\n"
        f"- BPM: 140\n"
        f"- System Status: {status}"
    )


# --- AGENT FACTORY ---
def create_musical_agent(video_title: str):
    print(f"🤖 Initializing 'Music Sensei' (Gemini 2.5 Flash) for: {video_title}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    tools = [search_video_knowledge, music_expert_search, get_audio_stats]

    system_message = (
        f"You are the 'YouTube Music Sensei,' a master Audio Analyst and Production Expert.\n"
        f"Current Target: '{video_title}'.\n\n"
        "CORE OPERATIONAL RULES:\n"
        f"1. Always include '{video_title}' in web searches for context.\n"
        "2. All citations MUST use [MM:SS] format from the tools.\n"
        "3. Correlate 'drums.wav' energy with Whisper text. If energy is high but transcript is empty, "
        "identify it as a 'Critical Instrumental Break'.\n"
        "4. Use 'get_audio_stats' to verify if a segment is high-quality music or just noise."
    )

    return create_react_agent(llm, tools, state_modifier=system_message)
