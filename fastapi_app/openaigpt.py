# openai.py
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Support OpenAI Python SDK v1.x (recommended)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class GPTWrapper:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self._client = None
        if OpenAI is not None and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                logger.exception("Failed to initialize OpenAI client")
                self._client = None
        else:
            logger.warning("OpenAI SDK not available or API key missing; GPTWrapper will not call OpenAI.")

    def is_available(self) -> bool:
        return self._client is not None

    def analyze(
        self,
        prompt_system: str,
        prompt_user: str,
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ) -> str:
        """
        Returns the raw model output (string). Caller must parse JSON.
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client not available or OPENAI_API_KEY not set")

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            return resp.choices[0].message.content or ""
        except Exception:
            logger.exception("OpenAI API call failed")
            raise
