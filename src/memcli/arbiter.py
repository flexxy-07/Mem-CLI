import json
import os
import warnings

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
load_dotenv()

_llm: ChatGoogleGenerativeAI | None = None


def _get_arbiter_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("Missing Gemini API key in environment.")
        model_name = os.getenv("MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        _llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=0,
        )
    return _llm


def detect_conflict(existing_memory: str, new_memory: str) -> dict:
    """Analyze if a new memory contradicts, updates, or supersedes an existing memory.

    Returns:
        dict: {"conflict": bool, "reason": str}
    """
    llm = _get_arbiter_llm()

    prompt = f"""You are an expert memory conflict arbiter.
Determine if a NEW memory contradicts, invalidates, updates, or supersedes an EXISTING memory.

Existing Memory: "{existing_memory}"
New Memory: "{new_memory}"

Rules:
- A CONFLICT / UPDATE occurs when the new memory changes a preference, updates a status, replaces a role/location, or directly opposes the existing memory for the same context.
  Examples of CONFLICT (conflict: true):
  * "The user prefers Java for interviews" vs "The user prefers Python for interviews" (changed preference)
  * "The user lives in Boston" vs "The user moved to London" (changed location)
  * "The user is learning LangGraph" vs "The user finished learning LangGraph" (updated status)
  * "The user dislikes Docker" vs "The user loves using Docker" (reversed opinion)

- COMPATIBLE memories do NOT conflict. They can both be true simultaneously in different or additive contexts.
  Examples of COMPATIBLE (conflict: false):
  * "The user prefers Java for interviews" vs "The user likes Python for machine learning" (different domain)
  * "The user is learning LangGraph" vs "The user knows Python" (additive skill)
  * "The user works as a software engineer" vs "The user lives in Seattle" (different topic)

Return ONLY valid JSON matching this schema:
{{"conflict": true, "reason": "<brief explanation>"}}
OR
{{"conflict": false, "reason": "<brief explanation>"}}
"""

    result = llm.invoke(prompt)
    content = result.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    content = content.strip().strip("`")
    if content.startswith("json"):
        content = content[4:].strip()

    try:
        data = json.loads(content)
        return {
            "conflict": bool(data.get("conflict", False)),
            "reason": str(data.get("reason", "")),
        }
    except Exception:
        return {"conflict": False, "reason": "Failed to parse arbiter response"}
