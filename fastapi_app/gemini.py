# gemini.py
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    # New Google Gen AI SDK
    from google import genai
except Exception:
    genai = None


class GeminiWrapper:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # New standard env var name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Keep your env override, but use a more current default model name if not set
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        self._client = None
        if genai is not None and self.api_key:
            try:
                # New SDK uses a Client object
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                logger.exception("Failed to initialize Google GenAI client")
                self._client = None
        else:
            logger.warning(
                "google.genai package missing or GEMINI_API_KEY not set; GeminiWrapper will not call Gemini."
            )

    def is_available(self) -> bool:
        return self._client is not None

    def analyze(
        self,
        prompt_system: str,
        prompt_user: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> str:
        """
        Returns raw model string (expected JSON). Raises RuntimeError if unavailable or call fails.
        """
        if not self.is_available():
            raise RuntimeError("Gemini client not available or GEMINI_API_KEY not set")

        try:
            # The new SDK expects a single 'contents' payload.
            # We concatenate system + user prompts to keep your current behavior.
            contents = f"{prompt_system}\n\n{prompt_user}"

            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_output_tokens),
                },
            )

            # Most convenient accessor in examples is resp.text
            text = getattr(resp, "text", None)
            if isinstance(text, str) and text.strip():
                return text

            # Fallback to string to avoid hard failures on SDK response shape changes
            return str(resp)

        except Exception:
            logger.exception("Gemini API call failed")
            raise
