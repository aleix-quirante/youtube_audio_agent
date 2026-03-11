# 🏮 YouTube Music Sensei 

> An advanced Multimodal RAG Agent that decodes the "musical DNA" of YouTube videos using audio source separation, signal processing, and LLM reasoning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?logo=pinecone&logoColor=white)

## 🧠 The Problem: The "Audio Chaos"
Analyzing music documentary videos (like Genius News) is incredibly difficult for standard AI because the narrator's voice and the background music constantly overlap. Traditional silence-detection fails.

**The Solution:** This project implements **Meta's Demucs** to isolate the rhythmic skeleton (drums/bass) from the vocals. By combining **OpenAI's Whisper** (transcription) and **Librosa** (RMS energy & tempo), we generated a perfectly synced **Master JSON**. This allows our AI agent to know exactly *when* the music is playing and *what* is being said.

## 🚀 Key Features

* **🎧 Audio Source Separation:** Uses AI models to deconstruct mixed audio signals.
* **🧠 ReAct Reasoning Agent:** Powered by **LangGraph** and **Gemini 2.5 Flash**, the agent doesn't just answer questions; it reasons, thinks, and decides which tools to use.
* **💾 Vector Knowledge (RAG):** Built on **Pinecone** to instantly retrieve exact timestamps, lyrics, and metadata from the video's DNA.
* **🌍 Real-Time Web Search:** Integrated with **Tavily API** to fetch external music history, artist facts, and records on the fly.
* **😈 "Godzilla Mode" Personality:** A custom UI toggle that transforms the wise AI Sensei into an arrogant, highly-technical music snob (just for fun and to demonstrate Prompt Engineering).

## 🏗️ Architecture & Tech Stack

1. **Data Ingestion:** `yt-dlp` -> `Demucs` -> `Whisper` -> `Librosa`
2. **Vectorization:** `Gemini Embeddings` -> `Pinecone`
3. **Agent Orchestration:** `LangChain` & `LangGraph`
4. **Interface:** `Streamlit`

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/youtube-music-sensei.git](https://github.com/yourusername/youtube-music-sensei.git)
   cd youtube-music-sensei