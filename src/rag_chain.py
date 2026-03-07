import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()


def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_query",  # Optimized for search
    )

    return PineconeVectorStore(
        index_name="youtube-agent-musical",
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
