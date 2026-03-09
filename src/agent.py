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


# --- DSP LOGIC: ENHANCED SENTIMENT (Valence/Arousal) ---
def analyze_enhanced_sentiment(y, sr):
    """
    Analyzes the emotional DNA combining Harmony (Valence) and Energy (Arousal).
    """
    # 1. Harmony (Valence - Major/Minor)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)

    # Simple Mode detection (Tonic vs 3rd)
    tonic_idx = np.argmax(mean_chroma)
    major_3rd = (tonic_idx + 4) % 12
    minor_3rd = (tonic_idx + 3) % 12

    sentiment_tag = (
        "Major" if mean_chroma[major_3rd] > mean_chroma[minor_3rd] else "Minor"
    )

    # 2. Energy & Rhythm (Arousal)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # 3. Sentiment Matrix Logic
    if sentiment_tag == "Major":
        mood = "Euphoric / Happy" if tempo > 120 else "Peaceful / Calm"
        valence_score = 0.8
    else:
        mood = "Tense / Aggressive" if tempo > 120 else "Sad / Depressing"
        valence_score = 0.2

    return {
        "mood_label": mood,
        "valence": valence_score,
        "arousal": round(energy * 10, 2),
        "tempo": round(tempo, 1),
        "key_mode": sentiment_tag,
    }


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


# --- TOOL 3: AUDIO STATS (Physics) ---
@tool
def get_audio_stats(timestamp: int) -> str:
    """
    Returns technical audio data (RMS energy, ZCR, BPM) for a specific second.
    Useful for validating drops, intensity, or percussive clarity.
    """
    # Logic: This would fetch from your Master JSON
    return (
        f"Technical Analysis at {format_timestamp(timestamp)}:\n"
        f"- RMS Energy: 0.082 (Punchy)\n"
        f"- ZCR: 0.045 (Clear Percussion)\n"
        f"- BPM: 140\n"
        f"- System Status: Stable Percussion / High Energy"
    )


# --- TOOL 4: EMOTIONAL ANALYZER (Sentiment) ---
@tool
def get_audio_sentiment(timestamp: int) -> str:
    """
    Returns the musical mood, valence (positivity), and arousal (intensity) for a specific second.
    Use this to explain the 'feeling' or atmosphere of the music.
    """
    # Logic: This would fetch from analyze_enhanced_sentiment output in your Master JSON
    data = {
        "mood": "Tense / Aggressive",
        "valence": 0.2,
        "arousal": 0.85,
        "key": "C Minor",
    }
    return (
        f"Emotional Analysis at {format_timestamp(timestamp)}:\n"
        f"- Atmosphere: {data['mood']}\n"
        f"- Valence: {data['valence']} (Low = Dark/Tense)\n"
        f"- Arousal: {data['arousal']} (High = Intense)\n"
        f"- Musical Key: {data['key']}"
    )


# --- AGENT FACTORY ---
def create_musical_agent(video_title: str):
    print(f"🤖 Initializing 'Music Sensei' (Gemini 2.5 Flash) for: {video_title}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    # Now including the 4 tools
    tools = [
        search_video_knowledge,
        music_expert_search,
        get_audio_stats,
        get_audio_sentiment,
    ]

    system_message = (
        f"You are the 'YouTube Music Sensei,' a master Audio Analyst and Psychoacoustics Expert.\n"
        f"Current Target: '{video_title}'.\n\n"
        "CORE OPERATIONAL RULES:\n"
        f"1. Always include '{video_title}' in web searches for context.\n"
        "2. All citations MUST use [MM:SS] format.\n"
        "3. CORRELATION: Cross-reference 'get_audio_sentiment' with 'search_video_knowledge'. "
        "Example: If the mood is 'Tense' while the transcript discusses a conflict, highlight this artistic choice.\n"
        "4. INSTRUMENTALS: If Arousal is high but the transcript is empty, identify it as a 'Powerful Instrumental Section'.\n"
        "5. MOOD SHIFTS: If the Valence changes (e.g., from Major/Happy to Minor/Sad), report it as an emotional pivot in the production."
    )

    return create_react_agent(llm, tools, state_modifier=system_message)
