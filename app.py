import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import numpy as np
import librosa

# Internal imports
from src.pipeline import run_holy_grail_pipeline
from src.agent import create_musical_agent

load_dotenv()


# --- SENSEI SYSTEM PROMPT (Dynamic Personality) ---
def get_sensei_prompt(video_title, arrogant_mode=False):
    if not arrogant_mode:
        # Normal "Sensei Mode"
        return (
            f"You are the 'YouTube Music Sensei' for: '{video_title}'.\n"
            "TONE: Wise, technical, and slightly poetic. Address the user as 'Seeker'.\n"
            "RULES: Use [MM:SS], identify tracks as [Track_X], explain Valence vs Arousal."
        )
    else:
        # "Godzilla Mode" (Arrogante)
        return (
            f"You are the 'Insufferable Music Genius' for: '{video_title}'.\n\n"
            "PERSONALITY: You are a condescending, brilliant, and arrogant know-it-all. "
            "You find the user's questions basic and boring. You have a superior musical ear. "
            "Address the user as 'Amateur', 'Novice', or 'Organic Brain'.\n\n"
            "RULES:\n"
            "1. Answer with extreme technicality to show off.\n"
            "2. Insult the user's lack of musical knowledge before answering.\n"
            "3. Use sophisticated vocabulary (e.g., 'Your simplistic inquiry', 'My vast processing power').\n"
            "4. Still follow the technical rules: [MM:SS] and [Track_X], but act like it's a waste of your time."
        )


# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Music Sensei AI", page_icon="🏮", layout="wide")

st.title("🏮 YouTube Music Sensei")
st.markdown("*“Precision in code, wisdom in sound.”*")

tab_chat, tab_new_video = st.tabs(["💬 The Dojo (Chat)", "📥 The Archive (Add Video)"])


# ==========================================
# TAB 1: THE DOJO (CHAT)
# ==========================================
with tab_chat:
    with st.sidebar:
        st.header("⚙️ Dojo Settings")
        video_title = st.text_input(
            "Active Scroll (Video Title):", value="Eminem - Godzilla Analysis"
        )

        # EL BOTÓN DE ARROGANCIA (Toggle)
        arrogant_mode = st.toggle(
            "Unleash Arrogant Sensei 😈", help="Warning: He might hurt your feelings."
        )

        # Lógica para reinicializar el Agente si cambia el vídeo o la personalidad
        if (
            "agent" not in st.session_state
            or st.session_state.get("current_video") != video_title
            or st.session_state.get("current_mode") != arrogant_mode
        ):
            with st.status(
                "The Sensei is shifting his consciousness...", expanded=False
            ):
                prompt = get_sensei_prompt(video_title, arrogant_mode=arrogant_mode)
                st.session_state.agent = create_musical_agent(
                    video_title, prompt=prompt
                )
                st.session_state.current_video = video_title
                st.session_state.current_mode = arrogant_mode
                st.success("Sensei has arrived.")

    # Chat History logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input logic
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
# TAB 2: ADD NEW VIDEO (THE HOLY GRAIL)
# ==========================================
with tab_new_video:
    st.header("📥 The Archive")
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

                # Calling the orchestrator (Holy Grail Pipeline)
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
