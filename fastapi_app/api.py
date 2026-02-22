# api.py
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI
from pydantic import ValidationError

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("charisma_api")
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load .env reliably (optional)
# -----------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    BASE_DIR = Path(__file__).resolve().parent  # fastapi_app/
    # Try common locations: project root ".env" and fastapi_app/.env
    candidates = [
        BASE_DIR.parent / ".env",
        BASE_DIR / ".env",
    ]
    loaded = False
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
            logger.info("Loaded .env from %s", p)
            loaded = True
            break
    if not loaded:
        # fallback: current working directory
        load_dotenv()
        logger.info("Loaded .env from current working directory (fallback)")
except Exception:
    logger.warning("python-dotenv not installed; env vars must be set by the shell/IDE")

# -----------------------------
# Imports (absolute, because you run `uvicorn api:app` inside fastapi_app)
# -----------------------------
from schemas import (  # noqa: E402
    AnalyzeRequest,
    AnalyzeResponse,
    CLTAnalysis,
    CLTItem,
    EvidenceItem,
    AnalysisMeta,
    AnalysisMetaSetup,
)
from openaigpt import GPTWrapper  # noqa: E402
from gemini import GeminiWrapper  # noqa: E402

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
# CLTs (global constant)
# -----------------------------
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

CLT_ID_ORDER = [cid for cid, _ in CLTS]  # list of ids in order
CLT_LABEL_MAP = {cid: label for cid, label in CLTS}

# -----------------------------
# Prompt defaults (local prompts folder under fastapi_app/)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent  # fastapi_app/
PROMPTS_DIR = BASE_DIR / "prompts"

DEFAULT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_default.txt"
DEFAULT_USER_PROMPT_PATH = PROMPTS_DIR / "user_default.txt"

DEFAULT_SYSTEM_PROMPT_FALLBACK = (
    "You are an expert rater of speeches for Charismatic Leadership Tactics (CLTs). "
    "You MUST output ONLY JSON that exactly matches the required schema. "
    "Do not output any prose, explanation, markdown, or text outside the JSON object. "
    "Analyze only the nine verbal CLTs. Exclude nonverbal behaviors. "
    "For each CLT provide: clt_id, label, present (bool), strength (0..3), "
    "confidence (0.0..1.0), evidence (list of quote objects with quote), rationale_short (<=280 chars)."
)

DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK = (
    "Text to analyze:\n\n'''{text}'''\n\n"
    "Rules/Guidance:\n"
    "- Only analyze the nine verbal CLTs.\n"
    "- Exclude nonverbal tactics explicitly.\n"
    "- Use evidence as short verbatim quotes found in the text.\n"
    "- Output valid JSON matching the CLTAnalysis structure: meta, overall_score, items[].\n"
    "- Output ONLY JSON.\n\n"
    "Return exactly one top-level JSON object.\n"
)

# -----------------------------
# NEW: Safe template rendering (prevents KeyError when template contains JSON braces)
# -----------------------------
def render_user_prompt(template: str, text: str) -> str:
    """
    Safe renderer:
    - Replaces ONLY the literal token '{text}' (no str.format()).
    - This avoids KeyError when user_default.txt contains JSON with { ... }.
    """
    template = template or ""
    text = text or ""

    if "{text}" in template:
        return template.replace("{text}", text)

    # If no placeholder exists, append text block
    return template.rstrip() + "\n\nText:\n'''\n" + text + "\n'''\n"


def normalize_llm_output(parsed: dict, model_name: str) -> dict:
    """
    Best-effort normalization to match CLTAnalysis schema.
    Keeps it deterministic and transparent.
    """
    if not isinstance(parsed, dict):
        return parsed

    # --- meta.setup repair ---
    meta = parsed.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        parsed["meta"] = meta

    setup = meta.get("setup")
    if not isinstance(setup, dict):
        meta["setup"] = {"clts": 9, "llm_runs": 1, "mode": f"{model_name.upper()} (LLM)"}
    else:
        setup.setdefault("clts", 9)
        setup.setdefault("llm_runs", 1)
        setup.setdefault("mode", f"{model_name.upper()} (LLM)")

    meta.setdefault("consistency", None)

    # --- overall_score int repair ---
    if "overall_score" in parsed:
        try:
            v = parsed["overall_score"]
            if isinstance(v, float):
                parsed["overall_score"] = int(round(v))
            elif isinstance(v, str) and v.strip():
                parsed["overall_score"] = int(round(float(v)))
        except Exception:
            pass

    # --- items repair ---
    items = parsed.get("items")
    if not isinstance(items, list):
        return parsed

    for it in items:
        if not isinstance(it, dict):
            continue

        # clt_id numeric -> map to string
        cid = it.get("clt_id")
        if isinstance(cid, int):
            if 1 <= cid <= len(CLT_ID_ORDER):
                it["clt_id"] = CLT_ID_ORDER[cid - 1]
        elif isinstance(cid, str):
            if cid.isdigit():
                n = int(cid)
                if 1 <= n <= len(CLT_ID_ORDER):
                    it["clt_id"] = CLT_ID_ORDER[n - 1]

        # label missing -> set from map
        cid2 = it.get("clt_id")
        if isinstance(cid2, str):
            it.setdefault("label", CLT_LABEL_MAP.get(cid2, cid2))

        # strength as float -> int
        if "strength" in it and isinstance(it["strength"], float):
            it["strength"] = int(round(it["strength"]))

        # present as "yes"/"no" -> bool
        if isinstance(it.get("present"), str):
            it["present"] = it["present"].strip().lower() in ("true", "yes", "1")

        # evidence normalize: list[str] -> list[{quote}]
        ev = it.get("evidence")
        if isinstance(ev, list) and ev and isinstance(ev[0], str):
            it["evidence"] = [{"quote": e[:240]} for e in ev if isinstance(e, str) and e.strip()]
        elif isinstance(ev, str):
            it["evidence"] = [{"quote": ev[:240]}] if ev.strip() else []

        # rationale_short ensure string
        if "rationale_short" in it and not isinstance(it["rationale_short"], str):
            it["rationale_short"] = str(it["rationale_short"])

    return parsed


def _read_text_file(path: Path) -> Optional[str]:
    if not path:
        return None
    if not path.is_file():
        logger.warning("Prompt file not found: %s", path)
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        logger.exception("Failed to read prompt file: %s", path)
        return None


def load_prompts() -> Tuple[str, str, Path, Optional[Path]]:
    """
    Load prompts with precedence:
      1) ENV path overrides (SYSTEM_PROMPT_DEFAULT_PATH / USER_PROMPT_DEFAULT_PATH)
      2) default local paths under fastapi_app/prompts/
      3) fallback hardcoded strings
    Returns: (system_prompt, user_template, system_path_used, user_path_used_or_None)
    """
    system_env = os.getenv("SYSTEM_PROMPT_DEFAULT_PATH")
    user_env = os.getenv("USER_PROMPT_DEFAULT_PATH")

    system_path = Path(system_env) if system_env else DEFAULT_SYSTEM_PROMPT_PATH
    user_path = Path(user_env) if user_env else DEFAULT_USER_PROMPT_PATH

    system_prompt = _read_text_file(system_path) or DEFAULT_SYSTEM_PROMPT_FALLBACK
    user_template = _read_text_file(user_path) or DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK

    # Ensure placeholder exists (for our renderer)
    if "{text}" not in user_template:
        logger.warning("User prompt template missing '{text}' placeholder; appending text block.")
        user_template = user_template.strip() + "\n\nText:\n'''{text}'''\n"

    user_path_used = user_path if user_path.is_file() else None
    system_path_used = system_path if system_path.is_file() else DEFAULT_SYSTEM_PROMPT_PATH

    return system_prompt, user_template, system_path_used, user_path_used


SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, SYSTEM_PATH_USED, USER_PATH_USED = load_prompts()
logger.info("System prompt loaded from: %s", SYSTEM_PATH_USED)
logger.info("User prompt loaded from: %s", USER_PATH_USED if USER_PATH_USED else "(fallback string)")

# -----------------------------
# JSON extraction helper (balanced braces)
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

    # 2) Find first balanced {...} that parses
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
                cand = raw[start : i + 1]
                try:
                    json.loads(cand)
                    return cand
                except Exception:
                    nxt = raw.find("{", start + 1)
                    if nxt == -1:
                        return None
                    return extract_json(raw[nxt:])

    return None

# -----------------------------
# Deterministic mock
# -----------------------------
def deterministic_seed(text: str, model: str) -> int:
    h = hashlib.sha256((model + "::" + text).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def mock_analyze(text: str, model: str) -> CLTAnalysis:
    seed = deterministic_seed(text, model)
    base_conf = 0.62 if model == "gpt" else 0.58
    bump = (seed % 17) / 100.0
    conf0 = min(0.92, base_conf + bump)

    sentences = [s.strip() for s in (text or "").replace("\r\n", "\n").split("\n") if s.strip()]

    def pick(i: int) -> str:
        if sentences:
            return sentences[min(i, len(sentences) - 1)][:160]
        return (text or "")[:160]

    items = []
    for idx, (cid, label) in enumerate(CLTS):
        strength = (seed + idx) % 4
        present = strength >= 1
        confidence = max(0.50, min(0.95, conf0 - idx * 0.03))
        evidence = [EvidenceItem(quote=pick(idx))] if present else []

        items.append(
            CLTItem(
                clt_id=cid,
                label=label,
                present=present,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                rationale_short="Mock-Analyse (LLM nicht verfügbar).",
            )
        )

    overall = round((sum((it.strength / 3) * it.confidence for it in items) / len(items)) * 100)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(clts=9, llm_runs=1, mode=f"{model.upper()} (Mock)"),
        consistency=None,
    )
    return CLTAnalysis(meta=meta, overall_score=overall, items=items)

# -----------------------------
# LLM analysis + validation
# -----------------------------
def _validate_clt_analysis(parsed: dict) -> CLTAnalysis:
    if hasattr(CLTAnalysis, "model_validate"):
        return CLTAnalysis.model_validate(parsed)  # pydantic v2
    return CLTAnalysis.parse_obj(parsed)  # pydantic v1

def analyze_with_wrapper(wrapper, text: str, model_name: str) -> CLTAnalysis:
    # IMPORTANT: safe rendering (no .format(), avoids KeyError when template contains JSON braces)
    user_prompt = render_user_prompt(USER_PROMPT_TEMPLATE, text)

    raw = wrapper.analyze(SYSTEM_PROMPT, user_prompt, temperature=0.0)

    if not isinstance(raw, str):
        raw = str(raw)

    json_text = extract_json(raw)
    if not json_text:
        logger.error("No JSON extracted from %s output. Raw head: %s", model_name, raw[:250])
        raise ValueError(f"No JSON extracted from {model_name} output")

    parsed = json.loads(json_text)

    # Prevent hard-fails on common LLM schema drift
    parsed = normalize_llm_output(parsed, model_name)

    try:
        analysis = _validate_clt_analysis(parsed)
    except ValidationError:
        try:
            logger.error("Validation failed for %s. Parsed keys: %s", model_name, list(parsed.keys()))
            if isinstance(parsed.get("meta"), dict):
                logger.error("Validation failed for %s. meta keys: %s", model_name, list(parsed["meta"].keys()))
            if isinstance(parsed.get("items"), list) and parsed["items"]:
                logger.error(
                    "Validation failed for %s. first item sample: %s",
                    model_name,
                    json.dumps(parsed["items"][0], ensure_ascii=False)[:400],
                )
        except Exception:
            pass

        logger.exception("Pydantic validation failed for %s output", model_name)
        raise

    # Ensure meta.setup.mode mentions the model used
    try:
        if analysis.meta and analysis.meta.setup:
            mode = analysis.meta.setup.mode or ""
            tag = model_name.upper()
            if tag not in mode:
                analysis.meta.setup.mode = (mode + f" · {tag}").strip(" ·")
    except Exception:
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
            logger.info("GPT LIVE used (no mock).")
        except Exception:
            logger.exception("GPT analysis failed; falling back to mock")
            analyses["gpt"] = mock_analyze(text, "gpt")
    else:
        logger.warning("GPT not available; using mock (check OPENAI_API_KEY / SDK).")
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