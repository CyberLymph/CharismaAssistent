# schemas.py
from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# -----------------------------
# Type aliases (Pylance-friendly)
# -----------------------------
Strength = Annotated[int, Field(ge=0, le=3)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Score100 = Annotated[int, Field(ge=0, le=100)]


# -----------------------------
# Models
# -----------------------------
class EvidenceItem(BaseModel):
    quote: str = Field(..., min_length=1, max_length=240)


class CLTItem(BaseModel):
    clt_id: str
    label: str
    present: bool
    strength: Strength
    confidence: Confidence
    evidence: List[EvidenceItem] = Field(default_factory=list)
    rationale_short: str = Field(..., max_length=280)


class AnalysisMetaSetup(BaseModel):
    clts: int = Field(default=9, ge=1)
    llm_runs: int = Field(default=1, ge=1)
    mode: str


class AnalysisMeta(BaseModel):
    setup: AnalysisMetaSetup
    consistency: Optional[Confidence] = None


class CLTAnalysis(BaseModel):
    meta: AnalysisMeta
    overall_score: Score100
    items: List[CLTItem]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3)
    enable_compare: bool = False
    # später:
    # llm_runs: int = 1
    # temperature: float = 0.2


class AnalyzeResponse(BaseModel):
    analyses: Dict[Literal["gpt", "gemini"], CLTAnalysis]
