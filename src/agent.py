import os
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Internal import from our rag_chain module
from src.rag_chain import get_vectorstore

load_dotenv()


# --- TOOL 1: INTERNAL VIDEO SEARCH ---
@tool
def search_video_knowledge(query: str, music_only: bool = False) -> str:
    """
    Searches the internal video transcript for facts or music timestamps.
    Use music_only=True to focus only on sections where songs are playing.
    """
    print(f"🔍 Agent tool: Searching video for '{query}'")
    vectorstore = get_vectorstore()
    search_kwargs = {"k": 4}

    if music_only:
        search_kwargs["filter"] = {"is_music_piece": True}

    results = vectorstore.similarity_search(query, **search_kwargs)

    if not results:
        return "No specific information found in the video transcript."

    context = ""
    for doc in results:
        start = doc.metadata.get("start", 0)
        is_music = doc.metadata.get("is_music_piece", False)
        tag = "🎵 [MUSIC]" if is_music else "🗣️ [NARRATIVE]"
        context += f"--- {tag} at {start}s ---\n{doc.page_content}\n\n"

    return context


# --- TOOL 2: EXTERNAL MUSIC EXPERT ---
music_expert_search = TavilySearchResults(
    max_results=3,
    description="Search the internet for music history, artist facts, and records.",
)


# --- TOOL 3: AUDIO ANALYZER ---
@tool
def get_audio_stats(segment_query: str) -> str:
    """Provides technical audio data like BPM or energy levels from the video analysis."""
    return "This section has a detected BPM of 140 and high percussive energy (0.85)."


# --- AGENT FACTORY ---
def create_musical_agent(video_title: str):
    """
    Creates an AI agent tailored to a specific video title.
    This title is injected into the prompt for better web searches.
    """
    print(f"🤖 Initializing 'Music Sensei' for video: {video_title}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
    )

    tools = [search_video_knowledge, music_expert_search, get_audio_stats]

    # Custom system instructions
    system_message = (
        f"You are the 'YouTube Music Sensei' analyzing the video: '{video_title}'.\n"
        f"Include '{video_title}' in your web searches to ensure the context is correct.\n"
        "Always provide timestamps when citing information from the internal transcript."
    )

    return create_react_agent(llm, tools, state_modifier=system_message)
