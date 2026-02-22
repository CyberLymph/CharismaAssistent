# api.py
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Generator, Any

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
# Safe template rendering (prevents KeyError when template contains JSON braces)
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


def mock_analyze(text: str, model: str, llm_runs: int = 1) -> CLTAnalysis:
    seed = deterministic_seed(text, model)
    base_conf = 0.62 if model == "gpt" else 0.58
    bump = (seed % 17) / 100.0
    conf0 = min(0.92, base_conf + bump)

    sentences = [s.strip() for s in (text or "").replace("\r\n", "\n").split("\n") if s.strip()]

    def pick(i: int) -> str:
        if sentences:
            return sentences[min(i, len(sentences) - 1)][:160]
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
                strength=int(strength),
                confidence=float(confidence),
                evidence=evidence,
                rationale_short="Mock analysis (LLM not available).",
            )
        )

    overall = round((sum((it.strength / 3) * it.confidence for it in items) / len(items)) * 100)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(
            clts=9,
            llm_runs=int(llm_runs),
            mode=f"{model.upper()} (Mock) · Ensemble({int(llm_runs)})",
        ),
        consistency=None,
    )
    return CLTAnalysis(meta=meta, overall_score=int(overall), items=items)


# -----------------------------
# Normalization (schema repair)
# -----------------------------
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
# Ensemble / Aggregation
# -----------------------------
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def compute_consistency_present(runs: List[CLTAnalysis]) -> float:
    """
    Consistency = share of CLTs where all runs agree on 'present' (0..1).
    """
    if not runs:
        return 0.0

    matrix: Dict[str, List[bool]] = {cid: [] for cid in CLT_ID_ORDER}
    for r in runs:
        by_id = {it.clt_id: it for it in r.items}
        for cid in CLT_ID_ORDER:
            it = by_id.get(cid)
            matrix[cid].append(bool(it.present) if it else False)

    agree = 0
    for _, vals in matrix.items():
        if vals and all(v == vals[0] for v in vals):
            agree += 1

    return agree / len(CLT_ID_ORDER)


def aggregate_runs(model_name: str, runs: List[CLTAnalysis]) -> CLTAnalysis:
    """
    Majority vote (present), mean strength/confidence, merge evidence.
    """
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

        # majority vote (ties -> False)
        true_count = sum(1 for v in presents if v)
        present = true_count > (len(presents) / 2) if presents else False

        strength_avg = round(sum(strengths) / max(1, len(strengths))) if strengths else 0
        confidence_avg = (sum(confidences) / max(1, len(confidences))) if confidences else 0.0

        strength = _clamp_int(strength_avg, 0, 3)
        confidence = _clamp_float(confidence_avg, 0.0, 1.0)

        # evidence unique + limit
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

        # rationale: take first; keep compact
        rationale = (rationales[0] if rationales else "Ensemble aggregation (no rationale provided).")[:280]

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
            mode=f"{model_name.upper()} · Ensemble({len(runs)}) · Rule+LLM",
        ),
        consistency=_clamp_float(consistency, 0.0, 1.0),
    )

    return CLTAnalysis(
        meta=meta,
        overall_score=_clamp_int(overall, 0, 100),
        items=aggregated_items,
    )


def run_ensemble(wrapper, text: str, model_name: str, llm_runs: int) -> CLTAnalysis:
    """
    Executes llm_runs calls and aggregates.
    Falls back to mock if wrapper unavailable or all calls fail.
    """
    llm_runs = int(llm_runs or 1)
    if llm_runs < 1:
        llm_runs = 1

    if not wrapper.is_available():
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    runs: List[CLTAnalysis] = []
    for i in range(llm_runs):
        try:
            runs.append(analyze_with_wrapper(wrapper, text, model_name))
        except Exception:
            logger.exception("%s run %d/%d failed", model_name.upper(), i + 1, llm_runs)

    if not runs:
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    if len(runs) == 1:
        r = runs[0]
        try:
            r.meta.setup.llm_runs = llm_runs
            r.meta.setup.mode = f"{model_name.upper()} · SingleRun(1/{llm_runs})"
            r.meta.consistency = None
        except Exception:
            pass
        return r

    return aggregate_runs(model_name, runs)


# -----------------------------
# NEW: Streaming Ensemble (NDJSON events for UI progress)
# -----------------------------
def _ndjson(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def run_ensemble_stream(wrapper, text: str, model_name: str, llm_runs: int) -> Generator[str, None, CLTAnalysis]:
    """
    Streams run completion events as NDJSON lines.
    Yields:
      - {"type":"run_done","model":"gpt","run":1,"overall_score":..}
    Returns:
      - aggregated CLTAnalysis (via generator return)
    """
    llm_runs = int(llm_runs or 1)
    if llm_runs < 1:
        llm_runs = 1

    if not wrapper.is_available():
        # No LLM: stream mock runs as if completed
        mock_run = mock_analyze(text, model_name, llm_runs=1)
        for i in range(1, llm_runs + 1):
            yield _ndjson(
                {"type": "run_done", "model": model_name, "run": i, "overall_score": int(mock_run.overall_score)}
            )
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    runs: List[CLTAnalysis] = []
    for i in range(1, llm_runs + 1):
        try:
            r = analyze_with_wrapper(wrapper, text, model_name)
            runs.append(r)
            yield _ndjson({"type": "run_done", "model": model_name, "run": i, "overall_score": int(r.overall_score)})
        except Exception as e:
            logger.exception("%s run %d/%d failed", model_name.upper(), i, llm_runs)

            # Keep UI moving: send a run_done anyway (mock fallback for this run)
            fallback = mock_analyze(text, model_name, llm_runs=1)
            yield _ndjson(
                {
                    "type": "run_done",
                    "model": model_name,
                    "run": i,
                    "overall_score": int(fallback.overall_score),
                    "note": "fallback_mock",
                    "error": str(e)[:200],
                }
            )

    if not runs:
        return mock_analyze(text, model_name, llm_runs=llm_runs)

    if len(runs) == 1 and llm_runs == 1:
        # already correct meta from LLM; adjust to show llm_runs
        try:
            runs[0].meta.setup.llm_runs = 1
        except Exception:
            pass
        return runs[0]

    return aggregate_runs(model_name, runs)


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(req.enable_compare)
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)

    analyses: Dict[str, CLTAnalysis] = {}

    # GPT (primary) — Ensemble
    try:
        analyses["gpt"] = run_ensemble(GPT, text, "gpt", llm_runs)
        logger.info("GPT analysis done (llm_runs=%s).", llm_runs)
    except Exception:
        logger.exception("GPT ensemble failed; falling back to mock")
        analyses["gpt"] = mock_analyze(text, "gpt", llm_runs=llm_runs)

    # Gemini (optional compare) — Ensemble
    if enable_compare:
        try:
            analyses["gemini"] = run_ensemble(GEMINI, text, "gemini", llm_runs)
        except Exception:
            logger.exception("Gemini ensemble failed; falling back to mock")
            analyses["gemini"] = mock_analyze(text, "gemini", llm_runs=llm_runs)

    return AnalyzeResponse(analyses=analyses)


@app.post("/analyze_stream")
def analyze_stream(req: AnalyzeRequest):
    """
    Streams NDJSON so the UI can show:
      Run 1 done, Run 2 done, Run 3 done, ...
    Final line contains the aggregated result:
      {"type":"final","analyses":{...}}
    """
    text = req.text
    enable_compare = bool(req.enable_compare)
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)

    def event_gen() -> Generator[str, None, None]:
        analyses: Dict[str, Any] = {}

        # Optional "progress" header event (UI supports it)
        yield _ndjson({"type": "progress", "message": f"Starting ensemble: GPT runs={llm_runs}"})

        # --- GPT streamed runs ---
        try:
            gpt_stream = run_ensemble_stream(GPT, text, "gpt", llm_runs)
            gpt_result: Optional[CLTAnalysis] = None
            while True:
                try:
                    line = next(gpt_stream)
                    yield line
                except StopIteration as si:
                    gpt_result = si.value  # aggregated CLTAnalysis
                    break
            analyses["gpt"] = gpt_result if gpt_result else mock_analyze(text, "gpt", llm_runs=llm_runs)
            logger.info("GPT streamed analysis done (llm_runs=%s).", llm_runs)
        except Exception as e:
            logger.exception("GPT streaming failed; falling back to mock")
            analyses["gpt"] = mock_analyze(text, "gpt", llm_runs=llm_runs)
            yield _ndjson({"type": "error", "message": f"GPT stream failed: {str(e)[:200]}"})

        # --- Gemini streamed runs (optional) ---
        if enable_compare:
            yield _ndjson({"type": "progress", "message": f"Starting ensemble: GEMINI runs={llm_runs}"})
            try:
                gem_stream = run_ensemble_stream(GEMINI, text, "gemini", llm_runs)
                gem_result: Optional[CLTAnalysis] = None
                while True:
                    try:
                        line = next(gem_stream)
                        yield line
                    except StopIteration as si:
                        gem_result = si.value
                        break
                analyses["gemini"] = gem_result if gem_result else mock_analyze(text, "gemini", llm_runs=llm_runs)
            except Exception as e:
                logger.exception("Gemini streaming failed; falling back to mock")
                analyses["gemini"] = mock_analyze(text, "gemini", llm_runs=llm_runs)
                yield _ndjson({"type": "error", "message": f"Gemini stream failed: {str(e)[:200]}"})

        # Final payload (the UI consumes this to render everything)
        final_obj = {"type": "final", "analyses": analyses}
        yield _ndjson(final_obj)

    return StreamingResponse(
    event_gen(),
    media_type="application/x-ndjson",
    headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    },
)