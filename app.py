import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Importamos tu fábrica de agentes desde tu código limpio
from src.agent import create_musical_agent

# Cargar contraseñas
load_dotenv()

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Music Sensei AI", page_icon="🎵", layout="wide")
st.title("🎧 YouTube Music Sensei")
st.markdown("Ask me anything about the video's audio, music history, or timestamps!")

# --- 2. CREACIÓN DE PESTAÑAS ---
tab_chat, tab_nuevo_video = st.tabs(
    ["💬 Chat con el Sensei", "📥 Analizar Nuevo Vídeo"]
)

# ==========================================
# PESTAÑA 1: EL CHAT (Lo que ya teníamos)
# ==========================================
with tab_chat:
    # --- BARRA LATERAL (Solo visible en el chat) ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        video_title = st.text_input(
            "Video Title (Base de Datos):", value="Eminem - Godzilla Analysis"
        )

        if (
            "agent" not in st.session_state
            or st.session_state.get("current_video") != video_title
        ):
            with st.spinner("Initializing Music Sensei..."):
                st.session_state.agent = create_musical_agent(video_title)
                st.session_state.current_video = video_title
                st.success("Agent Ready!")

    # --- MEMORIA DEL CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- LÓGICA DEL CHAT ---
    if prompt := st.chat_input("Ej: What happens at second 110?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Sensei is analyzing tools and thinking..."):
                try:
                    response = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]}
                    )
                    final_answer = response["messages"][-1].content
                    st.markdown(final_answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )
                except Exception as e:
                    st.error(f"Ups! Something went wrong: {e}")

# ==========================================
# PESTAÑA 2: AÑADIR NUEVO VÍDEO
# ==========================================
with tab_nuevo_video:
    st.header("📥 Añade un nuevo vídeo a la Base de Datos")
