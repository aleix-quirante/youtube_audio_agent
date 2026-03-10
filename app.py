import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import numpy as np
import librosa
from src.pipeline import run_holy_grail_pipeline

# Import your custom agent factory
from src.agent import create_musical_agent

load_dotenv()


# --- SENSEI SYSTEM PROMPT (Your Combined Rules) ---
def get_sensei_prompt(video_title):
    return (
        f"You are the 'YouTube Music Sensei,' the supreme Audio Analyst for: '{video_title}'.\n\n"
        "CORE OPERATIONAL EDICTS:\n"
        f"1. THE SACRED CONTEXT: Always include '{video_title}' in web searches.\n"
        "2. MULTI-TRACK PATH: Use [Track_X] labels to identify different songs in this video.\n"
        "3. THE HARMONY OF DUALITY: Use 'Atmosphere' metadata (Valence vs Arousal) to explain emotions.\n"
        "4. MARKING THE FOOTPRINTS: All citations MUST use the [MM:SS] format.\n"
        "5. THE UNION OF WORD AND SOUND: Correlate mood (e.g., 'Tense') with transcript conflicts.\n"
        "6. THE SILENCE THAT SPEAKS: Identify 'Powerful Instrumental Sections' when arousal is high but words are absent."
    )


# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Music Sensei AI", page_icon="🏮", layout="wide")

st.title("🏮 YouTube Music Sensei")
st.markdown("*“Precision in code, wisdom in sound.”*")

tab_chat, tab_new_video = st.tabs(["💬 The Dojo (Chat)", "📥 The Archive (Add Video)"])

# ==========================================
# TAB 1: THE DOJO
# ==========================================
with tab_chat:
    with st.sidebar:
        st.header("⚙️ Dojo Settings")
        video_title = st.text_input(
            "Active Scroll (Video Title):", value="Eminem - Godzilla Analysis"
        )

        if (
            "agent" not in st.session_state
            or st.session_state.get("current_video") != video_title
        ):
            with st.status("The Sensei is meditating...", expanded=False):
                # Pass the video_title and the combined prompt to your agent factory
                prompt = get_sensei_prompt(video_title)
                st.session_state.agent = create_musical_agent(
                    video_title, prompt=prompt
                )
                st.session_state.current_video = video_title
                st.success("Sensei is Ready.")

    # Chat History logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_input := st.chat_input("Seeker, what do you wish to decode?"):
        st.chat_message("user").markdown(prompt_input)
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        with st.chat_message("assistant"):
            with st.spinner("Consulting the frequencies..."):
                response = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=prompt_input)]}
                )
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

# ==========================================
# TAB 2: ADD NEW VIDEO
# ==========================================
with tab_new_video:
    st.header("📥 The Holy Grail Pipeline")
    st.markdown(
        "Paste a YouTube URL. The Sensei will download, separate, transcribe, and index the audio automatically."
    )

    url = st.text_input(
        "YouTube URL:", placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("🚀 Begin Deep Analysis", type="primary"):
        if url:
            with st.status(
                "Deconstructing the Artifact... (This will take several minutes)",
                expanded=True,
            ) as status:
                st.write("Initiating Deep Neural Processing...")

                # Calling the orchestrator
                success, result_message = run_holy_grail_pipeline(url)

                if success:
                    status.update(
                        label="✅ Integration Complete!",
                        state="complete",
                        expanded=False,
                    )
                    st.success(
                        f"Successfully digested: '{result_message}'. You may now consult the Sensei in The Dojo."
                    )
                else:
                    status.update(
                        label="❌ Analysis Failed", state="error", expanded=True
                    )
                    st.error(f"Error details: {result_message}")
        else:
            st.warning("Seeker, you must provide a valid scroll (URL).")
