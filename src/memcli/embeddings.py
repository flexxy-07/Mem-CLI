import os
import warnings

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
load_dotenv()

_client: GoogleGenerativeAIEmbeddings | None = None


def get_model_name() -> str:
    """Return the configured embedding model name."""
    return os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")


def get_embedding_client() -> GoogleGenerativeAIEmbeddings:
    """Initialize or return the cached GoogleGenerativeAIEmbeddings client."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing Gemini API key. Set GOOGLE_API_KEY, GEMINI_API_KEY, or LLM_API_KEY in .env"
            )

        _client = GoogleGenerativeAIEmbeddings(
            model=get_model_name(),
            google_api_key=api_key,
        )
    return _client


def get_embedding(text: str) -> list[float]:
    """Generate a vector embedding for a single text string."""
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")
    client = get_embedding_client()
    return client.embed_query(text.strip())
