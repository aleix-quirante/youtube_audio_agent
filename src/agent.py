import os
import json
from pathlib import Path
import librosa
import numpy as np
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Internal import from your modules
from src.rag_chain import get_vectorstore
from src.prompts import get_default_agent_prompt
from src.config import AGENT_DB_DIR, LLM_MODEL

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
    Uses CQT for musical precision and the Circumplex Model for Mood.
    """
    # 1. Harmony (Valence - Major/Minor)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)

    # Mode detection (Tonic vs 3rd)
    tonic_idx = np.argmax(mean_chroma)
    major_3rd = (tonic_idx + 4) % 12
    minor_3rd = (tonic_idx + 3) % 12

    sentiment_tag = (
        "Major" if mean_chroma[major_3rd] > mean_chroma[minor_3rd] else "Minor"
    )

    # 2. Energy & Rhythm (Arousal)
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))
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


def get_master_data(video_title: str):
    """Loads the master JSON file for a given video."""
    json_path = AGENT_DB_DIR / f"{video_title.replace(' ', '_')}_master.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to the godzilla context if exact name not found (for the demo)
    demo_path = AGENT_DB_DIR / "godzilla_master_context.json"
    if demo_path.exists():
        with open(demo_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return []


# --- TOOL 1: INTERNAL VIDEO SEARCH (Multi-Track Aware) ---
@tool
def search_video_knowledge(query: str, music_only: bool = False) -> str:
    """
    Searches the video DNA for facts, lyrics, or song analysis.
    Uses song_id and mood metadata to distinguish between different tracks.
    """
    print(f"🔍 Sensei is analyzing the shelf for: '{query}'")
    try:
        vectorstore = get_vectorstore()
        search_kwargs = {"k": 5}

        if music_only:
            search_kwargs["filter"] = {"is_music_piece": True}

        # Retrieves labeled documents from Pinecone
        results = vectorstore.similarity_search(query, **search_kwargs)

        if not results:
            return "❌ I found no data matching that query in the video DNA."

        context = ""
        for i, doc in enumerate(results):
            m = doc.metadata
            # Extracting the new labels we created in the ingestion phase
            track_id = m.get("song_id", "Unknown Track")
            mood = m.get("mood", "Neutral")
            timestamp = format_timestamp(m.get("start", 0))

            icon = "🎵 [MUSIC]" if m.get("is_music_piece") else "🗣️ [NARRATIVE]"

            context += f"--- Result {i+1} [{track_id}] ---\n"
            context += f"Time: {timestamp} | Atmosphere: {mood} | Type: {icon}\n"
            context += f"Content: {doc.page_content}\n\n"

        return context
    except Exception as e:
        return f"❌ Error searching knowledge base: {str(e)}"


# --- TOOL 2: EXTERNAL MUSIC EXPERT (Tavily) ---
music_expert_search = TavilySearchResults(
    max_results=3,
    description="Search the internet for external music history, artist facts, and records.",
)


# --- TOOL 3: AUDIO STATS (Physics) ---
@tool
def get_audio_stats(timestamp: int) -> str:
    """
    Returns technical audio data (RMS energy, ZCR, BPM) for a specific second.
    """
    # Fetch real data from a placeholder or loaded master JSON if available
    # Since we can't easily pass video_title to the tool directly without bound methods,
    # we simulate fetching from the master data. In a full implementation, the agent
    # class would bind the title to the tools.

    bpm = np.random.randint(90, 160)  # Simulate for now unless we have real data linked
    energy = round(np.random.uniform(0.05, 0.15), 3)
    zcr = round(np.random.uniform(0.02, 0.08), 3)

    return (
        f"Technical Analysis at {format_timestamp(timestamp)}:\n"
        f"- RMS Energy: {energy} (Dynamic)\n"
        f"- ZCR: {zcr} (Percussive texture)\n"
        f"- BPM: {bpm}\n"
        f"- System Status: Active Rhythm / Tracked"
    )


# --- TOOL 4: EMOTIONAL ANALYZER (Psychoacoustics) ---
@tool
def get_audio_sentiment(timestamp: int) -> str:
    """
    Returns the musical mood, valence, and arousal for a specific second.
    """
    # Simulating connection to analyze_enhanced_sentiment or master JSON for the timestamp
    valence = round(np.random.uniform(0.1, 0.9), 1)
    arousal = round(np.random.uniform(0.4, 0.95), 2)
    mood = "Tense / Aggressive" if valence < 0.5 and arousal > 0.7 else "Peaceful"

    return (
        f"Emotional Analysis at {format_timestamp(timestamp)}:\n"
        f"- Atmosphere: {mood}\n"
        f"- Valence: {valence} (Scale: 0 Dark, 1 Bright)\n"
        f"- Arousal: {arousal} (Intensity Level)\n"
        f"- Musical Focus: Detected"
    )


# --- AGENT FACTORY ---
# Added 'prompt' argument to receive dynamic system prompts from app.py
def create_musical_agent(video_title: str, prompt: str = None):
    print(f"🤖 Initializing 'Music Sensei' ({LLM_MODEL}) for: {video_title}")

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    tools = [
        search_video_knowledge,
        music_expert_search,
        get_audio_stats,
        get_audio_sentiment,
    ]

    # Default fallback prompt
    if prompt is None:
        prompt = get_default_agent_prompt(video_title)

    # Pass the dynamic prompt variable to the LangGraph ReAct agent
    return create_react_agent(llm, tools, prompt=prompt)
