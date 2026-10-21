"""
Manual sanity check: Gemini LLM connectivity

NOT a pytest test — run directly to confirm the API key and model
are correctly configured before running the full test suite.

Usage:
    python -m tests.manual.check_llm_connection
"""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model_name = os.getenv("GEMINI_MODEL") or os.getenv("MODEL")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

if not model_name:
    raise ValueError("Missing model name in .env (expected GEMINI_MODEL or MODEL)")
if not api_key:
    raise ValueError("Missing API key in .env (expected GOOGLE_API_KEY, GEMINI_API_KEY, or LLM_API_KEY)")

llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
response = llm.invoke("Reply with exactly: MemCLI Gemini connection successful!")

content = response.content
if isinstance(content, list):
    content = "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )

print(f"Model : {model_name}")
print(f"Status: {content.strip()}")
