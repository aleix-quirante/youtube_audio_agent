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


# --- TOOL 1: INTERNAL VIDEO SEARCH (Multi-Track Aware) ---
@tool
def search_video_knowledge(query: str, music_only: bool = False) -> str:
    """
    Searches the video DNA for facts, lyrics, or song analysis.
    Uses song_id and mood metadata to distinguish between different tracks.
    """
    print(f"🔍 Sensei is analyzing the shelf for: '{query}'")
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
    # In production, this fetches from your Master JSON
    return (
        f"Technical Analysis at {format_timestamp(timestamp)}:\n"
        f"- RMS Energy: 0.082 (Punchy)\n"
        f"- ZCR: 0.045 (Clear Percussion)\n"
        f"- BPM: 140\n"
        f"- System Status: Stable Percussion / High Energy"
    )


# --- TOOL 4: EMOTIONAL ANALYZER (Psychoacoustics) ---
@tool
def get_audio_sentiment(timestamp: int) -> str:
    """
    Returns the musical mood, valence, and arousal for a specific second.
    """
    # In production, this fetches from the analyze_enhanced_sentiment output
    return (
        f"Emotional Analysis at {format_timestamp(timestamp)}:\n"
        f"- Atmosphere: Tense / Aggressive\n"
        f"- Valence: 0.2 (Scale: 0 Dark, 1 Bright)\n"
        f"- Arousal: 0.85 (High Intensity)\n"
        f"- Musical Key: C Minor"
    )


# --- AGENT FACTORY ---
def create_musical_agent(video_title: str):
    print(f"🤖 Initializing 'Music Sensei' (Gemini 2.5 Flash) for: {video_title}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )

    tools = [
        search_video_knowledge,
        music_expert_search,
        get_audio_stats,
        get_audio_sentiment,
    ]

    # Sensei is now aware of multi-track structures and mood metadata
    system_message = (
        f"You are the 'YouTube Music Sensei,' a master Audio Analyst for: '{video_title}'.\n\n"
        "CORE OPERATIONAL RULES:\n"
        f"1. CONTEXT: Always include '{video_title}' in web searches.\n"
        "2. MULTI-TRACK AWARENESS: This video has multiple songs. Use [Track_X] labels to identify them.\n"
        "3. MOOD ANALYSIS: Use the 'Atmosphere' metadata to explain the emotional vibe (Valence vs Arousal).\n"
        "4. CITATIONS: All citations MUST use the [MM:SS] format provided by the tools.\n"
        "5. CORRELATION: If the mood is 'Tense' while the transcript discusses a conflict, highlight this production choice.\n"
        "6. INSTRUMENTALS: Identify 'Powerful Instrumental Sections' if arousal is high but transcript is empty."
    )

    # Using 'prompt=' for compatibility with your LangGraph version
    return create_react_agent(llm, tools, prompt=system_message)
