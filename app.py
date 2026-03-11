import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Internal imports
import src.config  # This triggers the critical env vars validation check
from src.pipeline import run_holy_grail_pipeline
from src.agent import create_musical_agent
from src.prompts import get_sensei_prompt

load_dotenv()

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
                try:
                    prompt = get_sensei_prompt(video_title, arrogant_mode=arrogant_mode)
                    st.session_state.agent = create_musical_agent(
                        video_title, prompt=prompt
                    )
                    st.session_state.current_video = video_title
                    st.session_state.current_mode = arrogant_mode
                    st.success("Sensei has arrived.")
                except Exception as e:
                    st.error(f"Error initializing Sensei: {str(e)}")
                    st.stop()

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
                try:
                    response = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt_input)]}
                    )
                    answer = response["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"The Sensei's connection was disrupted: {str(e)}")


# ==========================================
# TAB 2: ADD NEW VIDEO
# ==========================================
with tab_new_video:
    st.header("📥 The Archive (Beta)")
    st.markdown(
        "Paste a YouTube URL. The Sensei will download, separate, transcribe, and index the audio automatically."
    )

    url = st.text_input(
        "YouTube URL:", placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("🚀 Begin Deep Analysis", type="primary"):
        if url:
            # Elegant Work in Progress message instead of running the broken pipeline
            st.info(
                "🚧 **Feature in Development (v2.0)**\n\n"
                "YouTube recently updated its bot-protection matrix. "
                "The Sensei's extraction protocols are currently being upgraded to bypass these new security measures. "
                "For this presentation, the Dojo is running on our pre-indexed, high-fidelity Master JSONs."
            )
        else:
            st.warning("Seeker, you must provide a valid scroll (URL).")
