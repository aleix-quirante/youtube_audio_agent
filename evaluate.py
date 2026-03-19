import json
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage

from src.agent import create_musical_agent

load_dotenv()

TEST_CASES = [
    {
        "question": "¿Cuál es el mood principal y el tempo de Track_1?",
        "expected_context": ["mood", "tempo", "Euphoric", "Tense", "Sad", "Peaceful"],
    },
    {
        "question": "¿De qué trata la letra de Track_2?",
        "expected_context": ["theme", "sentiment", "summary", "poetic"],
    },
    {
        "question": "¿Qué pasa en el minuto 01:30?",
        "expected_context": ["1:30", "Atmosphere", "narrative", "music"],
    },
]


def evaluate_response(question, response, expected_context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY no encontrada en .env")
        return {
            "faithfulness": 0,
            "answer_relevance": 0,
            "reasoning": "Missing API Key",
        }

    client = genai.Client(api_key=api_key)
    model_id = "gemini-2.5-flash"

    prompt = f"""
    Eres un juez experto evaluando un sistema RAG (Retrieval-Augmented Generation).
    
    Pregunta del usuario: "{question}"
    Respuesta generada por el agente: "{response}"
    Contexto esperado (palabras clave que deberían estar presentes o inferirse): {expected_context}

    Evalúa la respuesta en dos métricas de 1 a 5:
    1. Faithfulness (Fidelidad): ¿La respuesta parece basada en hechos del contexto musical/video (sin alucinaciones)?
    2. Answer Relevance (Relevancia de la respuesta): ¿La respuesta aborda directamente la pregunta del usuario?

    Devuelve un JSON estricto con las siguientes claves:
    - "faithfulness": número entero del 1 al 5.
    - "answer_relevance": número entero del 1 al 5.
    - "reasoning": breve explicación de las puntuaciones.

    Solo devuelve el JSON raw, sin bloques ```json.
    """

    try:
        eval_response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return json.loads(eval_response.text)
    except Exception as e:
        print(f"Error evaluando con Gemini: {e}")
        return {
            "faithfulness": 0,
            "answer_relevance": 0,
            "reasoning": f"Error: {str(e)}",
        }


def main():
    print("🚀 Iniciando evaluación LLM-as-a-judge...")
    agent = create_musical_agent("godzilla")

    results = []
    total_faithfulness = 0
    total_relevance = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\\nTest Case {i}: {tc['question']}")

        # Invoke Agent
        response_state = agent.invoke(
            {"messages": [HumanMessage(content=tc["question"])]}
        )
        agent_response = response_state["messages"][-1].content
        print(f"Agente: {agent_response[:100]}...")

        # Evaluate
        eval_result = evaluate_response(
            tc["question"], agent_response, tc["expected_context"]
        )
        print(f"Evaluación: {eval_result}")

        results.append(
            {
                "test_case": i,
                "question": tc["question"],
                "agent_response": agent_response,
                "evaluation": eval_result,
            }
        )

        total_faithfulness += eval_result.get("faithfulness", 0)
        total_relevance += eval_result.get("answer_relevance", 0)

    # Generate Report
    avg_faithfulness = total_faithfulness / len(TEST_CASES)
    avg_relevance = total_relevance / len(TEST_CASES)
    overall_score = (avg_faithfulness + avg_relevance) / 2

    report_content = f"# Evaluación del Sistema RAG (LLM-as-a-judge)\\n\\n"
    report_content += f"## Resumen de Puntuaciones\\n"
    report_content += f"- **Faithfulness Medio**: {avg_faithfulness:.2f}/5.0\\n"
    report_content += f"- **Answer Relevance Medio**: {avg_relevance:.2f}/5.0\\n"
    report_content += f"- **Nota Media Global**: {overall_score:.2f}/5.0\\n\\n"
    report_content += f"## Detalles por Caso de Prueba\\n\\n"

    for r in results:
        ev = r["evaluation"]
        report_content += f"### Test Case {r['test_case']}\\n"
        report_content += f"**Pregunta**: {r['question']}\\n\\n"
        report_content += f"**Respuesta del Agente**: {r['agent_response']}\\n\\n"
        report_content += f"**Evaluación**: Faithfulness: {ev.get('faithfulness', 0)}/5 | Relevance: {ev.get('answer_relevance', 0)}/5\\n"
        report_content += f"**Razonamiento**: {ev.get('reasoning', 'N/A')}\\n\\n"
        report_content += "---\\n\\n"

    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\\n✅ Evaluación completada. Reporte guardado en 'evaluation_report.md'")


if __name__ == "__main__":
    main()
