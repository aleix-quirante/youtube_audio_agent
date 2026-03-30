import os
import json
from dotenv import load_dotenv

# We use google-genai SDK for Gemini 2.5 Flash as requested
from google import genai
from google.genai import types

from src.agent import create_musical_agent

load_dotenv()


def evaluate_rag():
    print("🚀 Starting RAG Agent Evaluation (LLM-as-a-judge)...")

    # 1. Definir los Test Cases
    test_cases = [
        {
            "question": "¿Cuál es el mood principal y el BPM del Track 1?",
            "expected_context": ["Euphoric", "Happy", "bpm", "tempo", "120"],
        },
        {
            "question": "¿De qué trata la letra de la primera canción que aparece en el video?",
            "expected_context": ["theme", "sentiment", "poetic"],
        },
        {
            "question": "Resume en una línea qué dice el narrador sobre el coro de la canción.",
            "expected_context": ["chorus", "narrator", "vocals"],
        },
    ]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY no encontrada en .env")
        return

    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash"

    agent = create_musical_agent()
    if not agent:
        print("❌ Error: No se pudo crear el agente musical.")
        return

    evaluations = []
    total_faithfulness = 0
    total_relevance = 0

    print(f"📋 Ejecutando {len(test_cases)} Test Cases...\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['question']}")

        # Obtener respuesta del RAG
        try:
            agent_response = agent.invoke({"input": test["question"]})
            answer = agent_response["answer"]
        except Exception as e:
            print(f"  ❌ Error al consultar el agente: {e}")
            continue

        print(f"  🤖 Agent Answer: {answer[:100]}...\n")

        # Prompt para LLM-as-a-judge
        eval_prompt = f"""
Eres un juez experto encargado de evaluar el rendimiento de un sistema RAG (Retrieval-Augmented Generation).
Se te proporcionará una pregunta del usuario, la respuesta generada por el sistema y un conjunto de palabras clave o conceptos esperados.

Pregunta del Usuario: "{test['question']}"
Respuesta del Sistema: "{answer}"
Contexto/Conceptos Esperados: {test['expected_context']}

Debes evaluar la respuesta en dos métricas, puntuando cada una del 1 al 5 (donde 1 es muy pobre y 5 es excelente):
1. "Faithfulness" (Fidelidad): ¿La respuesta parece basarse en información real y no alucina datos?
2. "Answer Relevance" (Relevancia de la respuesta): ¿La respuesta aborda directamente la pregunta del usuario?

Devuelve SOLO un objeto JSON válido con este formato exacto:
{{
  "faithfulness_score": <int 1-5>,
  "relevance_score": <int 1-5>,
  "feedback": "<Tus comentarios justificando las puntuaciones>"
}}
No incluyas bloques de código markdown extra como ```json.
"""

        try:
            eval_response = client.models.generate_content(
                model=model_id,
                contents=eval_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            result_json = json.loads(eval_response.text)

            f_score = result_json.get("faithfulness_score", 0)
            r_score = result_json.get("relevance_score", 0)
            feedback = result_json.get("feedback", "")

            total_faithfulness += f_score
            total_relevance += r_score

            evaluations.append(
                {
                    "test_case_id": i,
                    "question": test["question"],
                    "agent_answer": answer,
                    "faithfulness_score": f_score,
                    "relevance_score": r_score,
                    "feedback": feedback,
                }
            )

            print(f"  ✅ Faithfulness: {f_score}/5 | Relevance: {r_score}/5")

        except Exception as e:
            print(f"  ❌ Error durante la evaluación con Gemini: {e}")

    # Generar Reporte
    if not evaluations:
        print("⚠️ No se pudieron completar las evaluaciones.")
        return

    avg_faithfulness = total_faithfulness / len(evaluations)
    avg_relevance = total_relevance / len(evaluations)
    overall_score = (avg_faithfulness + avg_relevance) / 2

    report_content = "# 📊 RAG Evaluation Report\n\n"
    report_content += f"**Overall System Score:** {overall_score:.2f} / 5.00\n"
    report_content += f"- **Average Faithfulness:** {avg_faithfulness:.2f} / 5.00\n"
    report_content += f"- **Average Answer Relevance:** {avg_relevance:.2f} / 5.00\n\n"
    report_content += "---\n\n"

    for eval_item in evaluations:
        report_content += f"### Test Case {eval_item['test_case_id']}\n"
        report_content += f"**Question:** {eval_item['question']}\n\n"
        report_content += f"**Agent Answer:** {eval_item['agent_answer']}\n\n"
        report_content += f"**Scores:**\n"
        report_content += f"- Faithfulness: {eval_item['faithfulness_score']}/5\n"
        report_content += f"- Relevance: {eval_item['relevance_score']}/5\n\n"
        report_content += f"**Judge Feedback:** {eval_item['feedback']}\n\n"
        report_content += "---\n\n"

    report_path = "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n🎉 Evaluación completada! Reporte guardado en: {report_path}")
    print(f"Nota media del sistema: {overall_score:.2f} / 5.00")


if __name__ == "__main__":
    evaluate_rag()
