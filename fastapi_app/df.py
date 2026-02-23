# api.py
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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

    BASE_DIR_ENV = Path(__file__).resolve().parent  # fastapi_app/
    candidates = [
        BASE_DIR_ENV.parent / ".env",  # project root
        BASE_DIR_ENV / ".env",         # fastapi_app/.env
    ]
    loaded = False
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p)
            logger.info("Loaded .env from %s", p)
            loaded = True
            break
    if not loaded:
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

CLT_ID_ORDER = [cid for cid, _ in CLTS]
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
    "Analyze only the nine verbal CLTs. Exclude nonverbal behaviors."
)

# This template will be used for the HYBRID candidate-only prompting.
DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK = (
    "You will be given:\n"
    "1) The full text (for context)\n"
    "2) Candidate sentences per CLT (detected via deterministic rules)\n\n"
    "TASK:\n"
    "- Decide EACH CLT ONLY using the provided candidate sentences.\n"
    "- If candidates are empty, set present=false, strength=0, confidence<=0.35, evidence=[]\n"
    "- Evidence must be exact quotes copied from candidates.\n"
    "- Return ONLY JSON matching CLTAnalysis.\n\n"
    "FULL TEXT:\n'''{text}'''\n\n"
    "CANDIDATES (by CLT):\n{candidates_json}\n"
)

# -----------------------------
# Safe template rendering (prevents KeyError with JSON braces)
# -----------------------------
def render_user_prompt(template: str, text: str, candidates_json: str) -> str:
    template = template or ""
    text = text or ""
    candidates_json = candidates_json or "{}"

    # Replace only known tokens; no .format() to avoid brace issues
    if "{text}" in template:
        template = template.replace("{text}", text)
    else:
        template = template.rstrip() + "\n\nFULL TEXT:\n'''\n" + text + "\n'''\n"

    if "{candidates_json}" in template:
        template = template.replace("{candidates_json}", candidates_json)
    else:
        template = template.rstrip() + "\n\nCANDIDATES (by CLT):\n" + candidates_json + "\n"

    return template

# -----------------------------
# Prompt loading
# -----------------------------
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
    system_env = os.getenv("SYSTEM_PROMPT_DEFAULT_PATH")
    user_env = os.getenv("USER_PROMPT_DEFAULT_PATH")

    system_path = Path(system_env) if system_env else DEFAULT_SYSTEM_PROMPT_PATH
    user_path = Path(user_env) if user_env else DEFAULT_USER_PROMPT_PATH

    system_prompt = _read_text_file(system_path) or DEFAULT_SYSTEM_PROMPT_FALLBACK
    user_template = _read_text_file(user_path) or DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK

    # Ensure both placeholders exist for hybrid mode (we can append if missing)
    if "{text}" not in user_template:
        logger.warning("User prompt template missing '{text}'. Appending.")
        user_template = user_template.strip() + "\n\nFULL TEXT:\n'''{text}'''\n"
    if "{candidates_json}" not in user_template:
        logger.warning("User prompt template missing '{candidates_json}'. Appending.")
        user_template = user_template.strip() + "\n\nCANDIDATES (by CLT):\n{candidates_json}\n"

    user_path_used = user_path if user_path.is_file() else None
    system_path_used = system_path if system_path.is_file() else DEFAULT_SYSTEM_PROMPT_PATH
    return system_prompt, user_template, system_path_used, user_path_used

SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, SYSTEM_PATH_USED, USER_PATH_USED = load_prompts()
logger.info("System prompt loaded from: %s", SYSTEM_PATH_USED)
logger.info("User prompt loaded from: %s", USER_PATH_USED if USER_PATH_USED else "(fallback string)")

# -----------------------------
# Helpers
# -----------------------------
def extract_json(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None

    # direct parse
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

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

def deterministic_seed(text: str, model: str) -> int:
    h = hashlib.sha256((model + "::" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))

def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def _validate_clt_analysis(parsed: dict) -> CLTAnalysis:
    if hasattr(CLTAnalysis, "model_validate"):
        return CLTAnalysis.model_validate(parsed)  # pydantic v2
    return CLTAnalysis.parse_obj(parsed)  # pydantic v1

# -----------------------------
# Rule-based candidate extraction (deterministic)
# -----------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ“\"‘'])")

def split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    t = t.replace("\r\n", "\n")
    # if user wrote line breaks, treat each line as a sentence chunk too
    parts = []
    for chunk in t.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([s.strip() for s in _SENT_SPLIT_RE.split(chunk) if s.strip()])
    return parts

def _pick_unique(items: List[str], limit: int = 4) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = x.strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(x.strip())
        if len(out) >= limit:
            break
    return out

def find_candidates(text: str) -> Dict[str, List[str]]:
    """
    Deterministic rules -> candidate sentences per CLT.
    This is intentionally conservative: it provides *candidates*, not final decisions.
    """
    sents = split_sentences(text)

    # Precompiled lightweight patterns (EN-only, candidate finding)
    pat_rhet_q = re.compile(r"\?\s*['\"\)\]]*\s*$")  # allow ?", ?), ?]
    pat_contrast = re.compile(r"\b(but|yet|however|instead|rather than|nevertheless|nonetheless|still|whereas|while)\b",re.I)
    pat_lists = re.compile(r"\b(first|second|third|fourth|finally|next|then|to begin with|in addition|moreover)\b"
    r"|(\b\d+\s*[\)\.]\s+)"              
    r"|([,;:]\s*\w+.*[,;:]\s*\w+)",re.I
    )
    pat_collective = re.compile(r"\b(we|our|us|together|as one|people|nation|community|all of us)\b", re.I)
    pat_moral = re.compile(r"\b(should|must|ought|right|wrong|justice|unjust|moral|duty|responsibility|values)\b", re.I)
    pat_story = re.compile(r"\b(i remember|i met|i saw|once|when i|last (week|month|year)|years ago|one day|today i)\b",re.I)
    
    pat_metaphor = re.compile(
    r"\b(like|as if|as though)\b"               
    r"|(\b(is|are|was|were)\s+(a|an)\s+\w+)",   
    re.I)
    pat_goals = re.compile(
    r"\b(let us|we will|we must|we can|we are going to|i will|i pledge to)\b.*\b(build|create|achieve|reach|win|deliver|transform|change|reform)\b",
    re.I)
    pat_conf = re.compile(r"\b(i am sure|we can|we will|certain|undeniable|no doubt|inevitable|will prevail|cannot fail)\b", re.I)



    

    cand: Dict[str, List[str]] = {cid: [] for cid in CLT_ID_ORDER}

    for s in sents:
        if pat_rhet_q.search(s):
            cand["rhetorical_question"].append(s)
        if pat_contrast.search(s):
            cand["contrast"].append(s)
        if pat_lists.search(s):
            cand["lists_repetition"].append(s)
        if pat_collective.search(s):
            cand["collective_sentiment"].append(s)
        if pat_moral.search(s):
            cand["moral_conviction"].append(s)
        if pat_story.search(s):
            cand["story_anecdote"].append(s)
        if pat_metaphor.search(s):
            cand["metaphor_simile"].append(s)
        if pat_goals.search(s):
            cand["ambitious_goals"].append(s)
        if pat_conf.search(s):
            cand["confidence_in_goals"].append(s)

    # limit + unique for stability
    for cid in cand:
        cand[cid] = _pick_unique(cand[cid], limit=4)

    return cand

# -----------------------------
# Schema normalization (repair common drift)
# -----------------------------
def normalize_llm_output(parsed: dict, model_name: str, llm_runs: int = 1, mode: str = "") -> dict:
    if not isinstance(parsed, dict):
        return parsed

    meta = parsed.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        parsed["meta"] = meta

    setup = meta.get("setup")
    if not isinstance(setup, dict):
        meta["setup"] = {"clts": 9, "llm_runs": llm_runs, "mode": mode or f"{model_name.upper()} (LLM)"}
    else:
        setup.setdefault("clts", 9)
        setup.setdefault("llm_runs", llm_runs)
        setup.setdefault("mode", mode or f"{model_name.upper()} (LLM)")

    meta.setdefault("consistency", None)

    if "overall_score" in parsed:
        try:
            v = parsed["overall_score"]
            if isinstance(v, float):
                parsed["overall_score"] = int(round(v))
            elif isinstance(v, str) and v.strip():
                parsed["overall_score"] = int(round(float(v)))
        except Exception:
            pass

    items = parsed.get("items")
    if not isinstance(items, list):
        return parsed

    for it in items:
        if not isinstance(it, dict):
            continue

        cid = it.get("clt_id")
        if isinstance(cid, int):
            if 1 <= cid <= len(CLT_ID_ORDER):
                it["clt_id"] = CLT_ID_ORDER[cid - 1]
        elif isinstance(cid, str) and cid.isdigit():
            n = int(cid)
            if 1 <= n <= len(CLT_ID_ORDER):
                it["clt_id"] = CLT_ID_ORDER[n - 1]

        cid2 = it.get("clt_id")
        if isinstance(cid2, str):
            it.setdefault("label", CLT_LABEL_MAP.get(cid2, cid2))

        if "strength" in it and isinstance(it["strength"], float):
            it["strength"] = int(round(it["strength"]))
        if isinstance(it.get("present"), str):
            it["present"] = it["present"].strip().lower() in ("true", "yes", "1")

        ev = it.get("evidence")
        if isinstance(ev, list) and ev and isinstance(ev[0], str):
            it["evidence"] = [{"quote": e[:240]} for e in ev if isinstance(e, str) and e.strip()]
        elif isinstance(ev, str):
            it["evidence"] = [{"quote": ev[:240]}] if ev.strip() else []

        if "rationale_short" in it and not isinstance(it["rationale_short"], str):
            it["rationale_short"] = str(it["rationale_short"])

        # clamp
        try:
            it["strength"] = _clamp_int(int(it.get("strength", 0)), 0, 3)
        except Exception:
            it["strength"] = 0
        try:
            it["confidence"] = _clamp_float(float(it.get("confidence", 0.0)), 0.0, 1.0)
        except Exception:
            it["confidence"] = 0.0

    return parsed

# -----------------------------
# Mock
# -----------------------------
def mock_analyze(text: str, model: str, llm_runs: int = 1) -> CLTAnalysis:
    seed = deterministic_seed(text, model)
    base_conf = 0.62 if model == "gpt" else 0.58
    bump = (seed % 17) / 100.0
    conf0 = min(0.92, base_conf + bump)

    sents = split_sentences(text)
    def pick(i: int) -> str:
        if sents:
            return sents[min(i, len(sents) - 1)][:160]
        return (text or "")[:160]

    items: List[CLTItem] = []
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
                rationale_short="Mock analysis (LLM unavailable).",
            )
        )

    overall = round((sum((it.strength / 3) * it.confidence for it in items) / len(items)) * 100)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(clts=9, llm_runs=int(llm_runs), mode=f"{model.upper()} (Mock)"),
        consistency=None,
    )
    return CLTAnalysis(meta=meta, overall_score=int(overall), items=items)

# -----------------------------
# Hybrid LLM analysis (candidate-only decision)
# -----------------------------
def analyze_with_wrapper_hybrid(wrapper, text: str, model_name: str, llm_runs: int = 1) -> CLTAnalysis:
    candidates = find_candidates(text)
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)

    mode = f"{model_name.upper()} · HybridCandidates · LLM(validator)"
    user_prompt = render_user_prompt(USER_PROMPT_TEMPLATE, text, candidates_json)

    raw = wrapper.analyze(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    if not isinstance(raw, str):
        raw = str(raw)

    json_text = extract_json(raw)
    if not json_text:
        logger.error("No JSON extracted from %s output. Raw head: %s", model_name, raw[:250])
        raise ValueError(f"No JSON extracted from {model_name} output")

    parsed = json.loads(json_text)
    parsed = normalize_llm_output(parsed, model_name, llm_runs=llm_runs, mode=mode)

    try:
        analysis = _validate_clt_analysis(parsed)
    except ValidationError:
        logger.exception("Pydantic validation failed for %s output (hybrid)", model_name)
        raise

    # Ensure mode tag
    try:
        analysis.meta.setup.mode = mode
        analysis.meta.setup.llm_runs = int(llm_runs)
    except Exception:
        pass

    return analysis

# -----------------------------
# Ensemble / Aggregation (majority present + mean strength/conf)
# -----------------------------
def compute_consistency_present(runs: List[CLTAnalysis]) -> float:
    if not runs:
        return 0.0

    matrix: Dict[str, List[bool]] = {cid: [] for cid in CLT_ID_ORDER}
    for r in runs:
        by_id = {it.clt_id: it for it in r.items}
        for cid in CLT_ID_ORDER:
            it = by_id.get(cid)
            matrix[cid].append(bool(it.present) if it else False)

    agree = 0
    for cid, vals in matrix.items():
        if vals and all(v == vals[0] for v in vals):
            agree += 1
    return agree / len(CLT_ID_ORDER)

def aggregate_runs(model_name: str, runs: List[CLTAnalysis]) -> CLTAnalysis:
    if not runs:
        return mock_analyze("", model_name, llm_runs=1)

    run_dicts = [{it.clt_id: it for it in r.items} for r in runs]
    aggregated_items: List[CLTItem] = []

    for cid in CLT_ID_ORDER:
        label = CLT_LABEL_MAP.get(cid, cid)

        presents: List[bool] = []
        strengths: List[int] = []
        confidences: List[float] = []
        evid_quotes: List[str] = []
        rationales: List[str] = []

        for rd in run_dicts:
            it = rd.get(cid)
            if not it:
                continue
            presents.append(bool(it.present))
            strengths.append(int(it.strength))
            confidences.append(float(it.confidence))
            for ev in (it.evidence or []):
                q = (ev.quote or "").strip()
                if q:
                    evid_quotes.append(q)
            rs = (it.rationale_short or "").strip()
            if rs:
                rationales.append(rs)

        true_count = sum(1 for v in presents if v)
        present = true_count > (len(presents) / 2) if presents else False

        strength_avg = round(sum(strengths) / max(1, len(strengths))) if strengths else 0
        confidence_avg = (sum(confidences) / max(1, len(confidences))) if confidences else 0.0

        strength = _clamp_int(strength_avg, 0, 3)
        confidence = _clamp_float(confidence_avg, 0.0, 1.0)

        uniq: List[str] = []
        seen = set()
        for q in evid_quotes:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(q)
            if len(uniq) >= 3:
                break

        evidence = [EvidenceItem(quote=q[:240]) for q in uniq] if present else []
        rationale = (rationales[0] if rationales else "Ensemble aggregation (hybrid).")[:280]

        aggregated_items.append(
            CLTItem(
                clt_id=cid,
                label=label,
                present=present,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                rationale_short=rationale,
            )
        )

    overall = round(
        (sum((it.strength / 3) * it.confidence for it in aggregated_items) / len(aggregated_items)) * 100
    )

    consistency = compute_consistency_present(runs)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(
            clts=9,
            llm_runs=len(runs),
            mode=f"{model_name.upper()} · Ensemble({len(runs)}) · HybridCandidates",
        ),
        consistency=_clamp_float(consistency, 0.0, 1.0),
    )

    return CLTAnalysis(
        meta=meta,
        overall_score=_clamp_int(overall, 0, 100),
        items=aggregated_items,
    )

def run_ensemble_hybrid(wrapper, text: str, model_name: str, llm_runs: int) -> CLTAnalysis:
    llm_runs = int(llm_runs or 1)
    if llm_runs < 1:
        llm_runs = 1

    if not wrapper.is_available():
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    runs: List[CLTAnalysis] = []
    for i in range(llm_runs):
        try:
            runs.append(analyze_with_wrapper_hybrid(wrapper, text, model_name, llm_runs=llm_runs))
        except Exception:
            logger.exception("%s run %d/%d failed (hybrid)", model_name.upper(), i + 1, llm_runs)

    if not runs:
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    if len(runs) == 1:
        r = runs[0]
        try:
            r.meta.setup.llm_runs = llm_runs
            r.meta.setup.mode = f"{model_name.upper()} · SingleRun(1/{llm_runs}) · HybridCandidates"
            r.meta.consistency = None
        except Exception:
            pass
        return r

    return aggregate_runs(model_name, runs)

# -----------------------------
# NDJSON stream endpoint (optional, for run distribution UI)
# -----------------------------
@app.post("/analyze_stream")
def analyze_stream(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(req.enable_compare)
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)

    def event(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def event_gen():
        # GPT
        if GPT.is_available():
            gpt_runs: List[CLTAnalysis] = []
            for i in range(llm_runs):
                try:
                    a = analyze_with_wrapper_hybrid(GPT, text, "gpt", llm_runs=llm_runs)
                    gpt_runs.append(a)
                    yield event({"type": "run_done", "model": "gpt", "run": i + 1, "overall_score": a.overall_score})
                except Exception as e:
                    logger.exception("GPT run %d/%d failed (stream)", i + 1, llm_runs)
                    yield event({"type": "run_done", "model": "gpt", "run": i + 1, "overall_score": None, "note": "failed"})

            gpt_final = aggregate_runs("gpt", gpt_runs) if len(gpt_runs) >= 2 else (gpt_runs[0] if gpt_runs else mock_analyze(text, "gpt", llm_runs))
        else:
            gpt_final = mock_analyze(text, "gpt", llm_runs)

        analyses: Dict[str, CLTAnalysis] = {"gpt": gpt_final}

        # Gemini (optional)
        if enable_compare:
            if GEMINI.is_available():
                gem_runs: List[CLTAnalysis] = []
                for i in range(llm_runs):
                    try:
                        a = analyze_with_wrapper_hybrid(GEMINI, text, "gemini", llm_runs=llm_runs)
                        gem_runs.append(a)
                        yield event({"type": "run_done", "model": "gemini", "run": i + 1, "overall_score": a.overall_score})
                    except Exception:
                        logger.exception("Gemini run %d/%d failed (stream)", i + 1, llm_runs)
                        yield event({"type": "run_done", "model": "gemini", "run": i + 1, "overall_score": None, "note": "failed"})
                gem_final = aggregate_runs("gemini", gem_runs) if len(gem_runs) >= 2 else (gem_runs[0] if gem_runs else mock_analyze(text, "gemini", llm_runs))
            else:
                gem_final = mock_analyze(text, "gemini", llm_runs)

            analyses["gemini"] = gem_final

        yield event({"type": "final", "analyses": json.loads(AnalyzeResponse(analyses=analyses).model_dump_json() if hasattr(AnalyzeResponse, "model_dump_json") else AnalyzeResponse(analyses=analyses).json())["analyses"]})

    return StreamingResponse(
        event_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# -----------------------------
# Standard endpoint (non-stream)
# -----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(req.enable_compare)
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)

    analyses: Dict[str, CLTAnalysis] = {}

    # GPT (hybrid ensemble)
    try:
        analyses["gpt"] = run_ensemble_hybrid(GPT, text, "gpt", llm_runs)
        logger.info("GPT analysis done (llm_runs=%s, hybrid).", llm_runs)
    except Exception:
        logger.exception("GPT ensemble failed; falling back to mock")
        analyses["gpt"] = mock_analyze(text, "gpt", llm_runs=llm_runs)

    # Gemini (optional compare)
    if enable_compare:
        try:
            analyses["gemini"] = run_ensemble_hybrid(GEMINI, text, "gemini", llm_runs)
            logger.info("Gemini analysis done (llm_runs=%s, hybrid).", llm_runs)
        except Exception:
            logger.exception("Gemini ensemble failed; falling back to mock")
            analyses["gemini"] = mock_analyze(text, "gemini", llm_runs=llm_runs)

    return AnalyzeResponse(analyses=analyses)