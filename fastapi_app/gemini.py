# gemini.py
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from google import genai
except Exception:
    genai = None


class GeminiWrapper:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self._client = None
        if genai is not None and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                logger.exception("Failed to initialize Google GenAI client")
                self._client = None
        else:
            logger.warning("google.genai missing or GEMINI_API_KEY not set")

    def is_available(self) -> bool:
        return self._client is not None

    def analyze(
        self,
        prompt_system: str,
        prompt_user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,   # <— hochsetzen (vorher 2048)
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Gemini client not available or GEMINI_API_KEY not set")

        try:
            contents = f"{prompt_system}\n\n{prompt_user}"

            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_output_tokens),
                    # <— wichtig: JSON-only Output (reduziert fences + truncation issues)
                    "response_mime_type": "application/json",
                },
            )

            text = getattr(resp, "text", None)
            if isinstance(text, str) and text.strip():
                return text

            return str(resp)

        except Exception:
            logger.exception("Gemini API call failed")
            raise