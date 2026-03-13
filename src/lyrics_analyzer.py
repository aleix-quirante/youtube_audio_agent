import os
import json
from pathlib import Path
from dotenv import load_dotenv

# We use google-genai SDK for Gemini 2.5 Flash as requested
from google import genai
from google.genai import types

from src.config import AGENT_DB_DIR

load_dotenv()


def analyze_track_lyrics(track_id: str, lyrics: str) -> dict:
    """
    Sends the compiled lyrics to Gemini 2.5 Flash to perform semantic analysis.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY no encontrada en .env")
        return {}

    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash"

    prompt = f"""
    Eres un analista experto en música y literatura.
    A continuación, se presenta la transcripción de la letra (lyrics) correspondiente al segmento musical '{track_id}'.
    Es posible que la transcripción contenga ruido o comentarios del narrador del video mezclados con la canción.

    Letra transcrita:
    \"\"\"{lyrics}\"\"\"

    Por favor, analiza esta letra y proporciona un resumen estructurado en formato JSON con las siguientes claves exactas:
    - "theme": Tema principal o mensaje central de esta sección de la canción.
    - "sentiment": Sentimiento emocional de la letra (ej. empoderamiento, tristeza, agresión, alegría).
    - "poetic_elements": Elementos poéticos destacados (metáforas, rimas, estilo de flow).
    - "summary": Un breve resumen (1-2 oraciones) de lo que dice la letra.

    Devuelve SOLO un objeto JSON válido, sin bloques de código markdown extra como ```json, solo el JSON raw.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        analysis_result = json.loads(response.text)
        return analysis_result

    except Exception as e:
        print(f"❌ Error al analizar la letra del track {track_id} con Gemini: {e}")
        return {}


def process_master_json_for_lyrics(video_title: str):
    """
    Reads the master JSON file, extracts lyrics for each music track,
    analyzes them using Gemini, and appends the analysis back to the JSON.
    """
    safe_title = video_title.replace(" ", "_")
    json_path = AGENT_DB_DIR / f"{safe_title}_master.json"

    # Fallback to godzilla for demo
    if not json_path.exists():
        json_path = AGENT_DB_DIR / "godzilla_master_context.json"

    if not json_path.exists():
        print(f"❌ Error: No se encontró la base de datos para {video_title}")
        return

    print(f"📖 Leyendo base de datos: {json_path.name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Agrupar el texto de cada canción
    tracks_lyrics = {}
    for segment in data:
        if segment.get("is_music_piece") and segment.get("text"):
            song_id = segment.get("song_id")
            if song_id not in tracks_lyrics:
                tracks_lyrics[song_id] = []
            tracks_lyrics[song_id].append(segment["text"])

    if not tracks_lyrics:
        print("⚠️ No se encontraron segmentos de música con texto en este video.")
        return

    # 2. Analizar cada canción con Gemini 2.5 Flash
    track_analyses = {}
    print(
        f"🤖 Analizando letras con Gemini 2.5 Flash ({len(tracks_lyrics)} tracks encontrados)..."
    )
    for track_id, text_list in tracks_lyrics.items():
        full_lyrics = " ".join(text_list)
        # Skip very short fragments that aren't really lyrics
        if len(full_lyrics.strip()) < 15:
            continue

        print(f"  -> Analizando {track_id}...")
        analysis = analyze_track_lyrics(track_id, full_lyrics)
        if analysis:
            track_analyses[track_id] = analysis

    # 3. Inyectar el análisis de vuelta en el JSON master
    # Opcionalmente, creamos un documento 'resumen' por canción al final de la base de datos
    # para que RAG pueda consumirlo fácilmente como un bloque de contexto.

    new_segments = []
    for track_id, analysis in track_analyses.items():
        summary_text = (
            f"LYRICS ANALYSIS FOR {track_id}:\n"
            f"Theme: {analysis.get('theme', 'N/A')}\n"
            f"Sentiment: {analysis.get('sentiment', 'N/A')}\n"
            f"Poetic Elements: {analysis.get('poetic_elements', 'N/A')}\n"
            f"Summary: {analysis.get('summary', 'N/A')}"
        )

        # Encontramos cuándo empieza y termina esta canción para asignar tiempos
        track_segments = [s for s in data if s.get("song_id") == track_id]
        start_time = track_segments[0]["start"] if track_segments else 0
        end_time = track_segments[-1]["end"] if track_segments else 0

        analysis_doc = {
            "start": start_time,
            "end": end_time,
            "is_music_piece": True,
            "song_id": track_id,
            "mood": track_segments[0]["mood"] if track_segments else "Neutral",
            "text": summary_text,
            "is_lyrics_analysis": True,  # Flag especial para identificar este documento
        }
        new_segments.append(analysis_doc)

    if new_segments:
        data.extend(new_segments)
        # Sort by start time
        data = sorted(data, key=lambda x: x["start"])

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ Análisis de letras completado y guardado en {json_path.name}")
    else:
        print("⚠️ No se generó análisis de letras.")


if __name__ == "__main__":
    # Test function
    process_master_json_for_lyrics("godzilla")
