import os
import json
import subprocess
import yt_dlp
import whisper
import librosa
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    RAW_AUDIO_DIR,
    SEPARATED_DIR,
    AGENT_DB_DIR,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    HOP_LENGTH,
    INTENSITY_PERCENTILE,
    MIN_FRAMES_SECONDS,
)

load_dotenv()


# --- 1. DOWNLOAD AUDIO ---
def download_audio(video_url: str):
    print(f"📥 Downloading audio from: {video_url}")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
        "outtmpl": str(RAW_AUDIO_DIR / "%(title)s.%(ext)s"),
        "quiet": True,
        "cookiesfrombrowser": ("chrome",),
        "extractor_args": {"youtube": {"client": ["ios", "android"]}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            title = info.get("title", "Unknown_Video")
            # yt-dlp cleans up some characters, but generally matches this:
            audio_path = RAW_AUDIO_DIR / f"{title}.mp3"

        return str(audio_path), title
    except Exception as e:
        raise RuntimeError(f"Failed to download audio: {str(e)}")


# --- 2. DEMUCS SEPARATION ---
def run_demucs(audio_path: str, title: str):
    print(
        "🎛️ Running Demucs to isolate the rhythmic skeleton... (This may take a few minutes)"
    )

    try:
        # Run Demucs via command line automatically
        subprocess.run(
            ["demucs", audio_path, "-o", str(SEPARATED_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )

        # Demucs removes the .mp3 extension for the folder name
        base_name = Path(audio_path).stem
        drums_path = SEPARATED_DIR / "htdemucs" / base_name / "drums.wav"

        if not drums_path.exists():
            raise FileNotFoundError(
                f"Demucs completed but drums.wav not found at {drums_path}"
            )

        return str(drums_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs separation failed: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error during Demucs processing: {str(e)}")


# --- 3. AUDIO ANALYSIS & JSON CREATION ---
def process_audio_to_json(audio_path: str, drums_path: str, title: str):
    print("🧠 Transcribing with Whisper and analyzing sustained drum power...")

    try:
        # Whisper Transcription
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, fp16=False)

        # Librosa RMS Analysis
        y_drums, sr = librosa.load(drums_path)
        rms_drums = librosa.feature.rms(y=y_drums, hop_length=HOP_LENGTH)[0]
        times_rms = librosa.frames_to_time(
            range(len(rms_drums)), sr=sr, hop_length=HOP_LENGTH
        )

        intensity_threshold = np.percentile(rms_drums, INTENSITY_PERCENTILE)
        power_mask = rms_drums > intensity_threshold
        min_frames = int(MIN_FRAMES_SECONDS * sr / HOP_LENGTH)

        validated_music_blocks = []
        count = 0
        start_frame = 0

        for i, active in enumerate(power_mask):
            if active:
                if count == 0:
                    start_frame = i
                count += 1
            else:
                if count >= min_frames:
                    validated_music_blocks.append(
                        {
                            "start": float(times_rms[start_frame]),
                            "end": float(times_rms[i - 1]),
                        }
                    )
                count = 0

        # Cross-reference Whisper and Music Blocks
        agent_database = []
        for segment in result["segments"]:
            s_start = float(segment["start"])
            s_end = float(segment["end"])
            text = str(segment["text"]).strip()

            is_music_piece = False
            for block in validated_music_blocks:
                overlap = min(s_end, block["end"]) - max(s_start, block["start"])
                if overlap > 0.5:
                    is_music_piece = True
                    break

            if text:  # Only save if there's actual text
                agent_database.append(
                    {
                        "start": round(s_start, 2),
                        "end": round(s_end, 2),
                        "is_music_piece": is_music_piece,
                        "text": text,
                        "song_id": title,  # Tagging the specific video
                    }
                )

        # Save JSON
        json_path = AGENT_DB_DIR / f"{title.replace(' ', '_')}_master.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(agent_database, f, indent=4, ensure_ascii=False)

        return str(json_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process audio or create JSON: {str(e)}")


# --- 4. PINECONE INGESTION ---
def ingest_to_pinecone(json_path: str):
    print("💾 Vectorizing and uploading to Pinecone...")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document",
        )

        with open(json_path, "r", encoding="utf-8") as f:
            agent_data = json.load(f)

        documents = []
        for segment in agent_data:
            doc = Document(
                page_content=segment["text"],
                metadata={
                    "start": segment["start"],
                    "end": segment["end"],
                    "is_music_piece": segment["is_music_piece"],
                    "song_id": segment.get("song_id", "Unknown"),
                },
            )
            documents.append(doc)

        PineconeVectorStore.from_documents(
            documents,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to ingest to Pinecone: {str(e)}")


# --- 🚀 THE MASTER ORCHESTRATOR ---
def run_holy_grail_pipeline(youtube_url: str):
    """Executes the entire ETL pipeline from URL to Vector DB."""
    try:
        audio_path, title = download_audio(youtube_url)
        drums_path = run_demucs(audio_path, title)
        json_path = process_audio_to_json(audio_path, drums_path, title)
        ingest_to_pinecone(json_path)

        return True, title
    except RuntimeError as re:
        print(f"❌ Pipeline Step Error: {re}")
        return False, str(re)
    except Exception as e:
        print(f"❌ Unhandled Pipeline Error: {e}")
        return False, str(e)
