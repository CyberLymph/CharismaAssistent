# api.py
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import ValidationError


from schemas import AnalyzeRequest, AnalyzeResponse, CLTAnalysis
from openai import GPTWrapper
from gemini import GeminiWrapper

logger = logging.getLogger("charisma_api")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="CharismaAssistent API", version="1.0.0")

# -----------------------------
# Wrappers
# -----------------------------
GPT = GPTWrapper()
GEMINI = GeminiWrapper()

# -----------------------------
# Prompt loading (from .env paths)
# -----------------------------
DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "You are an expert rater of speeches for Charismatic Leadership Tactics (CLTs). "
    "You MUST output ONLY JSON that exactly matches the required schema. "
    "Do not output any prose, explanation, markdown, or text outside the JSON object. "
    "Analyze only the nine verbal CLTs. Exclude nonverbal behaviors. "
    "For each CLT provide: clt_id, label, present (bool), strength (0..3), "
    "confidence (0.0..1.0), evidence (list of quote strings), rationale_short (<=280 chars)."
)

DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK = (
    "Text to analyze:\n\n'''{text}'''\n\n"
    "Rules/Guidance:\n"
    "- Only analyze the nine verbal CLTs.\n"
    "- Exclude nonverbal tactics explicitly.\n"
    "- Use 'evidence' as short verbatim quotes found in the text (no paraphrases).\n"
    "- Output valid JSON matching the CLTAnalysis structure: meta, overall_score, items[].\n"
    "- Output ONLY JSON.\n\n"
    "Return exactly one top-level JSON object.\n"
)

def _read_text_file(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_file():
        logger.warning("Prompt file not found: %s", p)
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        logger.exception("Failed to read prompt file: %s", p)
        return None

def load_prompts() -> tuple[str, str]:
    """
    Loads prompts from environment paths.
    Expected:
      SYSTEM_PROMPT_DEFAULT_PATH=prompts/system_default.txt
      USER_PROMPT_DEFAULT_PATH=prompts/user_default.txt (optional but recommended)
    """
    system_path = os.getenv("SYSTEM_PROMPT_DEFAULT_PATH")
    user_path = os.getenv("USER_PROMPT_DEFAULT_PATH")

    system_prompt = _read_text_file(system_path) or DEFAULT_SYSTEM_PROMPT_FALLBACK
    user_template = _read_text_file(user_path) or DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK

    # Ensure user template contains {text} placeholder
    if "{text}" not in user_template:
        logger.warning("User prompt template missing '{text}' placeholder; appending text block.")
        user_template = user_template.strip() + "\n\nText:\n'''{text}'''\n"

    return system_prompt, user_template

# Cache prompts in memory (reload on startup)
SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts()

# -----------------------------
# JSON extraction helpers
# -----------------------------


def extract_json(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None

    # 1) Direct parse
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    # 2) Find first balanced {...} that parses as JSON
    start = raw.find("{")
    if start == -1:
        return None

    in_str = False
    escape = False
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = raw[start:i+1]
                try:
                    json.loads(cand)
                    return cand
                except Exception:
                    # if this candidate isn't valid JSON, keep searching
                    # by looking for the next '{' after the current start
                    next_start = raw.find("{", start + 1)
                    if next_start == -1:
                        return None
                    start = next_start
                    # reset scan to new start
                    in_str = False
                    escape = False
                    depth = 0
                    # restart loop from new start-1 so next iteration is start
                    return extract_json(raw[start:])

    return None


# -----------------------------
# Mock fallback 

# -----------------------------
def fallback_mock(text: str, model: str) -> CLTAnalysis:
        # simple in-file fallback (same behavior as earlier mock_analyze)
        from .api import CLTS  # if defined in same module, else reconstruct
        # For safety: minimal mock using same structure as earlier (quick and deterministic)
        # Here we reuse a very small deterministic seed approach:
        seed = int(hashlib.sha256((model + "::" + text).encode("utf-8")).hexdigest()[:8], 16)
        items = []
        # If CLTS not present, define minimally:
        CLTS_LOCAL = [
            ("moral_conviction", "Moral Conviction"),
            ("collective_sentiment", "Collective Sentiment"),
            ("lists_repetition", "Lists / Repetition"),
            ("rhetorical_question", "Rhetorical Question"),
            ("contrast", "Contrast"),
            ("story_anecdote", "Story / Anecdote"),
            ("metaphor_simile", "Metaphor / Simile"),
            ("ambitious_goals", "Ambitious Goals"),
            ("confidence_in_goals", "Confidence in Goals"),
        ]
        sentences = [s.strip() for s in text.replace("\r\n","\n").split("\n") if s.strip()]
        for idx, (cid, label) in enumerate(CLTS_LOCAL):
            strength = (seed + idx) % 4
            present = strength >= 1
            confidence = max(0.5, min(0.95, 0.6 - idx*0.02))
            evidence = [{"quote": sentences[min(idx, len(sentences)-1)][:160]}] if present and sentences else []
            items.append({
                "clt_id": cid,
                "label": label,
                "present": present,
                "strength": strength,
                "confidence": confidence,
                "evidence": evidence,
                "rationale_short": "Fallback mock"
            })
        overall = round((sum((it["strength"]/3)*it["confidence"] for it in items)/len(items))*100)
        meta = {"setup": {"clts":9, "llm_runs":1, "mode": f"{model.upper()} (fallback-mock)"}, "consistency": None}
        return CLTAnalysis.parse_obj({"meta": meta, "overall_score": overall, "items": items})

fallback_mock = fallback_mock



def deterministic_seed(text: str, model: str) -> int:
    import hashlib
    h = hashlib.sha256((model + "::" + text).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def mock_analyze(text: str, model: str) -> CLTAnalysis:
    """
    Minimal deterministic mock that conforms to CLTAnalysis.
    """
    from .schemas import CLTItem, EvidenceItem, AnalysisMeta, AnalysisMetaSetup

    CLTS = [
        ("moral_conviction", "Moral Conviction"),
        ("collective_sentiment", "Collective Sentiment"),
        ("lists_repetition", "Lists / Repetition"),
        ("rhetorical_question", "Rhetorical Question"),
        ("contrast", "Contrast"),
        ("story_anecdote", "Story / Anecdote"),
        ("metaphor_simile", "Metaphor / Simile"),
        ("ambitious_goals", "Ambitious Goals"),
        ("confidence_in_goals", "Confidence in Goals"),
    ]

    seed = deterministic_seed(text, model)
    base_conf = 0.62 if model == "gpt" else 0.58
    bump = (seed % 17) / 100.0
    conf0 = min(0.92, base_conf + bump)

    sentences = [s.strip() for s in text.replace("\r\n", "\n").split("\n") if s.strip()]
    pick = lambda i: (sentences[i] if i < len(sentences) else text[:160]).strip()[:160]

    items = []
    for idx, (cid, label) in enumerate(CLTS):
        strength = (seed + idx) % 4
        present = strength >= 1
        confidence = max(0.50, min(0.95, conf0 - idx * 0.03))
        evidence = [EvidenceItem(quote=pick(min(idx, max(0, len(sentences) - 1))))] if present else []
        items.append(CLTItem(
            clt_id=cid,
            label=label,
            present=present,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            rationale_short="Mock-Analyse (LLM nicht verfügbar)."
        ))

    overall = round((sum((it.strength / 3) * it.confidence for it in items) / len(items)) * 100)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(clts=9, llm_runs=1, mode=f"{model.upper()} (Mock) · Rule+LLM placeholder"),
        consistency=None
    )
    return CLTAnalysis(meta=meta, overall_score=overall, items=items)

# -----------------------------
# LLM analysis + validation
# -----------------------------
def analyze_with_wrapper(wrapper, text: str, model_name: str) -> CLTAnalysis:
    """
    Calls wrapper.analyze, extracts JSON, validates into CLTAnalysis.
    Raises on failure (caller handles fallback).
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(text=text)
    raw = wrapper.analyze(SYSTEM_PROMPT, user_prompt, temperature=0.0)

    if not isinstance(raw, str):
        raw = str(raw)

    json_text = extract_json(raw)
    if not json_text:
        raise ValueError(f"No JSON extracted from {model_name} output")

    parsed = json.loads(json_text)

    try:
        analysis = CLTAnalysis.parse_obj(parsed)
    except ValidationError:
        logger.exception("Pydantic validation failed for %s output", model_name)
        raise

    # Ensure meta.setup.mode mentions actual model used (useful for UI + traceability)
    try:
        if analysis.meta and analysis.meta.setup:
            mode = analysis.meta.setup.mode or ""
            if model_name.upper() not in mode:
                analysis.meta.setup.mode = (mode + f" · {model_name.upper()}").strip(" ·")
    except Exception:
        # Not critical
        pass

    return analysis

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(req.enable_compare)

    analyses: Dict[str, CLTAnalysis] = {}

    # GPT (primary)
    if GPT.is_available():
        try:
            analyses["gpt"] = analyze_with_wrapper(GPT, text, "gpt")
        except Exception:
            logger.exception("GPT analysis failed; falling back to mock")
            analyses["gpt"] = mock_analyze(text, "gpt")
    else:
        logger.info("GPT not available; using mock")
        analyses["gpt"] = mock_analyze(text, "gpt")

    # Gemini (optional compare)
    if enable_compare:
        if GEMINI.is_available():
            try:
                analyses["gemini"] = analyze_with_wrapper(GEMINI, text, "gemini")
            except Exception:
                logger.exception("Gemini analysis failed; falling back to mock")
                analyses["gemini"] = mock_analyze(text, "gemini")
        else:
            logger.info("Gemini not available; using mock")
            analyses["gemini"] = mock_analyze(text, "gemini")

    return AnalyzeResponse(analyses=analyses)
