# api.py
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

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
DEFAULT_HYBRID_USER_PROMPT_PATH = PROMPTS_DIR / "user_hybrid_default.txt"  # optional

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

DEFAULT_HYBRID_USER_PROMPT_TEMPLATE_FALLBACK = (
    "You will NOT analyze the full text. You will ONLY use the candidate sentences below.\n"
    "Task:\n"
    "- For each CLT decide present (true/false) based ONLY on the candidates.\n"
    "- Assign strength (0..3) and confidence (0.0..1.0).\n"
    "- Evidence must be verbatim quotes copied from the candidates.\n"
    "- Output ONLY JSON matching CLTAnalysis.\n\n"
    "CANDIDATES (grouped by CLT):\n"
    "{candidates}\n"
)

# -----------------------------
# Safe template rendering
# -----------------------------
def render_user_prompt(template: str, text: str) -> str:
    template = template or ""
    text = text or ""

    if "{text}" in template:
        return template.replace("{text}", text)

    return template.rstrip() + "\n\nText:\n'''\n" + text + "\n'''\n"

def render_hybrid_prompt(template: str, candidates_block: str) -> str:
    template = template or ""
    if "{candidates}" in template:
        return template.replace("{candidates}", candidates_block or "")
    return template.rstrip() + "\n\nCANDIDATES:\n" + (candidates_block or "") + "\n"

# -----------------------------
# Prompt loading
# -----------------------------
def _read_text_file(path: Path) -> Optional[str]:
    if not path:
        return None
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        logger.exception("Failed to read prompt file: %s", path)
        return None

def load_prompts() -> Tuple[str, str, str, Path, Optional[Path], Optional[Path]]:
    """
    Precedence:
      1) ENV path overrides
      2) default local under fastapi_app/prompts/
      3) fallback hardcoded strings
    """
    system_env = os.getenv("SYSTEM_PROMPT_DEFAULT_PATH")
    user_env = os.getenv("USER_PROMPT_DEFAULT_PATH")
    hybrid_env = os.getenv("HYBRID_USER_PROMPT_DEFAULT_PATH")

    system_path = Path(system_env) if system_env else DEFAULT_SYSTEM_PROMPT_PATH
    user_path = Path(user_env) if user_env else DEFAULT_USER_PROMPT_PATH
    hybrid_path = Path(hybrid_env) if hybrid_env else DEFAULT_HYBRID_USER_PROMPT_PATH

    system_prompt = _read_text_file(system_path) or DEFAULT_SYSTEM_PROMPT_FALLBACK
    user_template = _read_text_file(user_path) or DEFAULT_USER_PROMPT_TEMPLATE_FALLBACK
    hybrid_template = _read_text_file(hybrid_path) or DEFAULT_HYBRID_USER_PROMPT_TEMPLATE_FALLBACK

    if "{text}" not in user_template:
        logger.warning("user_default.txt missing '{text}' placeholder; appending.")
        user_template = user_template.strip() + "\n\nText:\n'''{text}'''\n"

    if "{candidates}" not in hybrid_template:
        logger.warning("user_hybrid_default.txt missing '{candidates}' placeholder; appending.")
        hybrid_template = hybrid_template.strip() + "\n\n{candidates}\n"

    return (
        system_prompt,
        user_template,
        hybrid_template,
        system_path,
        user_path if user_path.is_file() else None,
        hybrid_path if hybrid_path.is_file() else None,
    )

SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, HYBRID_USER_PROMPT_TEMPLATE, SYSTEM_PATH_USED, USER_PATH_USED, HYBRID_PATH_USED = load_prompts()
logger.info("System prompt loaded from: %s", SYSTEM_PATH_USED)
logger.info("User prompt loaded from: %s", USER_PATH_USED if USER_PATH_USED else "(fallback string)")
logger.info("Hybrid prompt loaded from: %s", HYBRID_PATH_USED if HYBRID_PATH_USED else "(fallback string)")

# -----------------------------
# JSON extraction helper
# -----------------------------
def extract_json(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None

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

# -----------------------------
# Deterministic mock
# -----------------------------
def deterministic_seed(text: str, model: str) -> int:
    h = hashlib.sha256((model + "::" + (text or "")).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def mock_analyze(text: str, model: str, llm_runs: int = 1, mode_tag: str = "Mock") -> CLTAnalysis:
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
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                rationale_short="Mock analysis (LLM not available).",
            )
        )

    overall = round((sum((it.strength / 3) * it.confidence for it in items) / len(items)) * 100)
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(
            clts=9,
            llm_runs=int(llm_runs),
            mode=f"{model.upper()} ({mode_tag}) · Ensemble({int(llm_runs)})",
        ),
        consistency=None,
    )
    return CLTAnalysis(meta=meta, overall_score=int(overall), items=items)

# -----------------------------
# Candidate extraction (English only)
# -----------------------------
_pat_rhet_q = re.compile(r"\?\s*$")
_pat_contrast = re.compile(r"\b(but|yet|however|instead|rather than|on the one hand|on the other hand)\b", re.I)
_pat_lists = re.compile(r"\b(first|second|third|finally)\b|[,;].+[,;].+", re.I)
_pat_collective = re.compile(r"\b(we|our|us|together|as one|people|nation|community)\b", re.I)
_pat_moral = re.compile(r"\b(should|must|right|wrong|justice|unjust|moral|duty|responsibility)\b", re.I)
_pat_story = re.compile(r"\b(i remember|i met|once|when i|last week|years ago|one day)\b", re.I)
_pat_metaphor = re.compile(r"\b(like|as if|as though)\b|(\bis\b\s+a\b\s+\w+)", re.I)
_pat_goals = re.compile(r"\b(will|we will|let us|we must)\b.*\b(build|create|achieve|reach|win|deliver|transform)\b", re.I)
_pat_conf = re.compile(r"\b(i am sure|we can|we will|certain|undeniable|no doubt|inevitable)\b", re.I)

def split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"\'])", t.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]

def build_candidates(text: str) -> Dict[str, List[str]]:
    sents = split_sentences(text)
    out: Dict[str, List[str]] = {cid: [] for cid in CLT_ID_ORDER}

    for s in sents:
        if _pat_moral.search(s):
            out["moral_conviction"].append(s)
        if _pat_collective.search(s):
            out["collective_sentiment"].append(s)
        if _pat_lists.search(s):
            out["lists_repetition"].append(s)
        if _pat_rhet_q.search(s):
            out["rhetorical_question"].append(s)
        if _pat_contrast.search(s):
            out["contrast"].append(s)
        if _pat_story.search(s):
            out["story_anecdote"].append(s)
        if _pat_metaphor.search(s):
            out["metaphor_simile"].append(s)
        if _pat_goals.search(s):
            out["ambitious_goals"].append(s)
        if _pat_conf.search(s):
            out["confidence_in_goals"].append(s)

    # de-dup + cap per CLT
    for cid in out:
        seen = set()
        uniq = []
        for s in out[cid]:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= 6:
                break
        out[cid] = uniq

    return out

def candidates_to_block(cands: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    for cid in CLT_ID_ORDER:
        label = CLT_LABEL_MAP.get(cid, cid)
        lines.append(f"- {cid} ({label}):")
        xs = cands.get(cid, [])
        if not xs:
            lines.append("  (none)")
        else:
            for s in xs:
                lines.append(f"  • {s}")
        lines.append("")  # blank line
    return "\n".join(lines).strip()

# -----------------------------
# Normalization (schema repair)
# -----------------------------
def normalize_llm_output(parsed: dict, model_name: str) -> dict:
    """
    Best-effort normalization to match CLTAnalysis schema.
    Handles common schema drift, including:
      - meta/setup missing
      - overall_score float/str
      - items list issues
      - HYBRID alternative output: top-level keys per clt_id (no items/overall_score)
    """
    if not isinstance(parsed, dict):
        return parsed

    # -------------------------------------------------------
    # HYBRID ALT SHAPE:
    # { "moral_conviction": {...}, "collective_sentiment": {...}, ..., "meta": {...} }
    # -> convert to { meta, overall_score, items:[...] }
    # -------------------------------------------------------
    has_items = isinstance(parsed.get("items"), list)
    has_overall = "overall_score" in parsed

    # detect: many CLT keys at top-level
    clt_key_hits = [cid for cid in CLT_ID_ORDER if cid in parsed and isinstance(parsed.get(cid), dict)]
    if (not has_items or not has_overall) and len(clt_key_hits) >= 5:
        # Build items[] from these top-level CLT dicts
        items_list = []
        for cid in CLT_ID_ORDER:
            block = parsed.get(cid, {})
            if not isinstance(block, dict):
                block = {}

            label = CLT_LABEL_MAP.get(cid, cid)

            present = block.get("present", False)
            strength = block.get("strength", 0)
            confidence = block.get("confidence", 0.0)
            evidence = block.get("evidence", [])
            rationale = block.get("rationale_short", "")

            # present normalize
            if isinstance(present, str):
                present = present.strip().lower() in ("true", "yes", "1")

            # strength normalize
            try:
                if isinstance(strength, float):
                    strength = int(round(strength))
                elif isinstance(strength, str) and strength.strip():
                    strength = int(round(float(strength)))
            except Exception:
                strength = 0
            strength = max(0, min(3, int(strength)))

            # confidence normalize
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, float(confidence)))

            # evidence normalize: list[str] OR str OR list[{quote}]
            ev_out = []
            if isinstance(evidence, list):
                if evidence and isinstance(evidence[0], str):
                    ev_out = [{"quote": e[:240]} for e in evidence if isinstance(e, str) and e.strip()]
                elif evidence and isinstance(evidence[0], dict):
                    for e in evidence:
                        q = (e.get("quote") if isinstance(e, dict) else "")
                        if isinstance(q, str) and q.strip():
                            ev_out.append({"quote": q[:240]})
                else:
                    ev_out = []
            elif isinstance(evidence, str) and evidence.strip():
                ev_out = [{"quote": evidence[:240]}]

            # rationale normalize
            if not isinstance(rationale, str):
                rationale = str(rationale)
            rationale = (rationale or "")[:280]

            items_list.append({
                "clt_id": cid,
                "label": label,
                "present": bool(present),
                "strength": strength,
                "confidence": confidence,
                "evidence": ev_out if bool(present) else [],
                "rationale_short": rationale or "—",
            })

        # compute overall_score if missing
        if "overall_score" not in parsed:
            try:
                overall = round(
                    (sum((it["strength"] / 3) * it["confidence"] for it in items_list) / len(items_list)) * 100
                )
            except Exception:
                overall = 0
            parsed["overall_score"] = max(0, min(100, int(overall)))

        parsed["items"] = items_list

        # Keep meta if present, otherwise create
        meta = parsed.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            parsed["meta"] = meta

        setup = meta.get("setup")
        if not isinstance(setup, dict):
            meta["setup"] = {"clts": 9, "llm_runs": 1, "mode": f"{model_name.upper()} (LLM) · HYBRID"}
        else:
            setup.setdefault("clts", 9)
            setup.setdefault("llm_runs", 1)
            setup.setdefault("mode", f"{model_name.upper()} (LLM) · HYBRID")

        meta.setdefault("consistency", None)

        # continue below for further repairs (scores etc.)

    # --- meta.setup repair (normal path) ---
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
        elif isinstance(cid, str) and cid.isdigit():
            n = int(cid)
            if 1 <= n <= len(CLT_ID_ORDER):
                it["clt_id"] = CLT_ID_ORDER[n - 1]

        # label missing -> set from map
        cid2 = it.get("clt_id")
        if isinstance(cid2, str):
            it.setdefault("label", CLT_LABEL_MAP.get(cid2, cid2))

        # strength as float/str -> int
        if "strength" in it:
            try:
                if isinstance(it["strength"], float):
                    it["strength"] = int(round(it["strength"]))
                elif isinstance(it["strength"], str) and it["strength"].strip():
                    it["strength"] = int(round(float(it["strength"])))
            except Exception:
                it["strength"] = 0
            it["strength"] = max(0, min(3, int(it["strength"])))

        # present as "yes"/"no" -> bool
        if isinstance(it.get("present"), str):
            it["present"] = it["present"].strip().lower() in ("true", "yes", "1")

        # evidence normalize: list[str] -> list[{quote}]
        ev = it.get("evidence")
        if isinstance(ev, list) and ev and isinstance(ev[0], str):
            it["evidence"] = [{"quote": e[:240]} for e in ev if isinstance(e, str) and e.strip()]
        elif isinstance(ev, str):
            it["evidence"] = [{"quote": ev[:240]}] if ev.strip() else []
        elif isinstance(ev, list) and ev and isinstance(ev[0], dict):
            fixed = []
            for e in ev:
                q = e.get("quote") if isinstance(e, dict) else ""
                if isinstance(q, str) and q.strip():
                    fixed.append({"quote": q[:240]})
            it["evidence"] = fixed

        # rationale_short ensure string
        if "rationale_short" in it and not isinstance(it["rationale_short"], str):
            it["rationale_short"] = str(it["rationale_short"])
        if "rationale_short" in it and isinstance(it["rationale_short"], str):
            it["rationale_short"] = it["rationale_short"][:280]

    return parsed

# -----------------------------
# LLM analysis + validation
# -----------------------------
def _validate_clt_analysis(parsed: dict) -> CLTAnalysis:
    if hasattr(CLTAnalysis, "model_validate"):
        return CLTAnalysis.model_validate(parsed)  # pydantic v2
    return CLTAnalysis.parse_obj(parsed)  # pydantic v1

def analyze_with_wrapper(wrapper, text: str, model_name: str, use_hybrid: bool) -> CLTAnalysis:
    if use_hybrid:
        cands = build_candidates(text)
        candidates_block = candidates_to_block(cands)
        user_prompt = render_hybrid_prompt(HYBRID_USER_PROMPT_TEMPLATE, candidates_block)
    else:
        user_prompt = render_user_prompt(USER_PROMPT_TEMPLATE, text)

    raw = wrapper.analyze(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    if not isinstance(raw, str):
        raw = str(raw)

    json_text = extract_json(raw)
    if not json_text:
        logger.error("No JSON extracted from %s output. Raw head: %s", model_name, raw[:250])
        raise ValueError(f"No JSON extracted from {model_name} output")

    parsed = json.loads(json_text)
    parsed = normalize_llm_output(parsed, model_name)

    analysis = _validate_clt_analysis(parsed)

    # Ensure meta tag
    try:
        tag = model_name.upper()
        mode = analysis.meta.setup.mode or ""
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

def aggregate_runs(model_name: str, runs: List[CLTAnalysis], use_hybrid: bool) -> CLTAnalysis:
    if not runs:
        return mock_analyze("", model_name, llm_runs=1, mode_tag="Mock")

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
        rationale = (rationales[0] if rationales else "Ensemble aggregation.")[:280]

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

    overall = round((sum((it.strength / 3) * it.confidence for it in aggregated_items) / len(aggregated_items)) * 100)
    consistency = compute_consistency_present(runs)

    mode = f"{model_name.upper()} · Ensemble({len(runs)}) · " + ("Hybrid(Candidates)" if use_hybrid else "FullText")
    meta = AnalysisMeta(
        setup=AnalysisMetaSetup(clts=9, llm_runs=len(runs), mode=mode),
        consistency=_clamp_float(consistency, 0.0, 1.0),
    )

    return CLTAnalysis(
        meta=meta,
        overall_score=_clamp_int(overall, 0, 100),
        items=aggregated_items,
    )

def run_ensemble(wrapper, text: str, model_name: str, llm_runs: int, use_hybrid: bool) -> CLTAnalysis:
    llm_runs = int(llm_runs or 1)
    if llm_runs < 1:
        llm_runs = 1

    if not wrapper.is_available():
        return mock_analyze(text, model_name, llm_runs=llm_runs, mode_tag="Mock")

    runs: List[CLTAnalysis] = []
    for i in range(llm_runs):
        runs.append(analyze_with_wrapper(wrapper, text, model_name, use_hybrid=use_hybrid))

    if len(runs) == 1:
        r = runs[0]
        try:
            r.meta.setup.llm_runs = 1
            r.meta.setup.mode = f"{model_name.upper()} · SingleRun · " + ("Hybrid(Candidates)" if use_hybrid else "FullText")
            r.meta.consistency = None
        except Exception:
            pass
        return r

    return aggregate_runs(model_name, runs, use_hybrid=use_hybrid)

# -----------------------------
# API: /analyze (non-stream)
# -----------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(getattr(req, "enable_compare", False))
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)
    use_hybrid = bool(getattr(req, "use_hybrid", False))

    analyses: Dict[str, CLTAnalysis] = {}

    try:
        analyses["gpt"] = run_ensemble(GPT, text, "gpt", llm_runs=llm_runs, use_hybrid=use_hybrid)
        logger.info("GPT analysis done (llm_runs=%s, hybrid=%s).", llm_runs, use_hybrid)
    except Exception:
        logger.exception("GPT analysis failed; falling back to mock")
        analyses["gpt"] = mock_analyze(text, "gpt", llm_runs=llm_runs, mode_tag="Mock")

    if enable_compare:
        try:
            analyses["gemini"] = run_ensemble(GEMINI, text, "gemini", llm_runs=llm_runs, use_hybrid=use_hybrid)
        except Exception:
            logger.exception("Gemini analysis failed; falling back to mock")
            analyses["gemini"] = mock_analyze(text, "gemini", llm_runs=llm_runs, mode_tag="Mock")

    return AnalyzeResponse(analyses=analyses)

# -----------------------------
# API: /analyze_stream (NDJSON)
# - emits run_done per run
# - emits final with aggregated analyses
# -----------------------------
@app.post("/analyze_stream")
def analyze_stream(req: AnalyzeRequest):
    text = req.text
    enable_compare = bool(getattr(req, "enable_compare", False))
    llm_runs = int(getattr(req, "llm_runs", 3) or 3)
    use_hybrid = bool(getattr(req, "use_hybrid", False))

    def ndjson(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def stream():
        try:
            analyses: Dict[str, CLTAnalysis] = {}

            # GPT runs
            if not GPT.is_available():
                yield ndjson({"type": "progress", "message": "GPT not available; using mock."})
                analyses["gpt"] = mock_analyze(text, "gpt", llm_runs=llm_runs, mode_tag="Mock")
            else:
                gpt_runs: List[CLTAnalysis] = []
                for i in range(1, llm_runs + 1):
                    r = analyze_with_wrapper(GPT, text, "gpt", use_hybrid=use_hybrid)
                    gpt_runs.append(r)
                    yield ndjson({"type": "run_done", "model": "gpt", "run": i, "overall_score": r.overall_score})
                analyses["gpt"] = aggregate_runs("gpt", gpt_runs, use_hybrid=use_hybrid)

            # Gemini runs (optional)
            if enable_compare:
                if not GEMINI.is_available():
                    yield ndjson({"type": "progress", "message": "Gemini not available; using mock."})
                    analyses["gemini"] = mock_analyze(text, "gemini", llm_runs=llm_runs, mode_tag="Mock")
                else:
                    gem_runs: List[CLTAnalysis] = []
                    for i in range(1, llm_runs + 1):
                        r = analyze_with_wrapper(GEMINI, text, "gemini", use_hybrid=use_hybrid)
                        gem_runs.append(r)
                        yield ndjson({"type": "run_done", "model": "gemini", "run": i, "overall_score": r.overall_score})
                    analyses["gemini"] = aggregate_runs("gemini", gem_runs, use_hybrid=use_hybrid)

            # final payload (same shape Step2 expects)
            final = {"type": "final", "analyses": json.loads(AnalyzeResponse(analyses=analyses).model_dump_json())["analyses"]} \
                if hasattr(AnalyzeResponse, "model_dump_json") else {"type": "final", "analyses": AnalyzeResponse(analyses=analyses).dict()["analyses"]}

            yield ndjson(final)

        except Exception as e:
            logger.exception("Streaming failed")
            yield ndjson({"type": "error", "message": str(e)})

    return StreamingResponse(stream(), media_type="application/x-ndjson")