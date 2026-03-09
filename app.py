import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import numpy as np
import librosa

# Import your agent factory
from src.agent import create_musical_agent

# Load secrets
load_dotenv()

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Music Sensei AI", page_icon="🏮", layout="wide")

# Custom CSS for that "Sensei" vibe
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSecondaryBlock { background-color: #1a1c24; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏮 YouTube Music Sensei")
st.markdown("*“Listen with your mind, analyze with your soul.”*")

# --- 2. TABS: THE DOJO & THE ARCHIVE ---
tab_chat, tab_upload = st.tabs(
    ["💬 The Sensei's Dojo", "📥 The Archive (New Video)"]
)

# ==========================================
# TAB 1: THE DOJO (The Chat)
# ==========================================
with tab_chat:
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("⚙️ Dojo Settings")
        video_title = st.text_input(
            "Current Video in Study:", value="Eminem - Godzilla Analysis"
        )
        
        st.divider()
        st.info("The Sensei is currently analyzing the harmonic layers and emotional weight of this track.")

        # Agent Initialization Logic
        if "agent" not in st.session_state or st.session_state.get("current_video") != video_title:
            with st.status("The Sensei is meditating on the frequencies...", expanded=False):
                st.session_state.agent = create_musical_agent(video_title)
                st.session_state.current_video = video_title
                st.success("Sensei is Ready.")

    # --- CHAT HISTORY ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the dojo. Ask me about the chords, the vibe, or the secrets hidden in this audio."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- CHAT LOGIC ---
    if prompt := st.chat_input("Master, what's the emotional vibe at 02:30?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Sensei response
        with st.chat_message("assistant"):
            with st.spinner("Consulting the musical scrolls..."):
                try:
                    # Invoke your LangChain agent
                    response = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]}
                    )
                    final_answer = response["messages"][-1].content
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final