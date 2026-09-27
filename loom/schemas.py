from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Profile(Strict):
    interests: str = Field(default="", max_length=1200)
    experience: str = Field(default="", max_length=1200)
    compute: str = Field(default="单张消费级 GPU", max_length=600)
    timeline: str = Field(default="尚未确定", max_length=300)
    goal: str = Field(default="理解方法，寻找可验证的研究切入点", max_length=1500)


class RunRequest(Strict):
    paper_ids: list[str] = Field(min_length=1, max_length=12)
    mode: Literal["understand", "plan", "question", "figure"] = "understand"
    question: str = Field(default="", max_length=2000)
    profile: Profile = Field(default_factory=Profile)
    page: int | None = Field(default=None, ge=1, le=100)
    bbox: list[float] | None = None
    parent_run_id: str | None = None
    model_policy: Literal["auto", "local", "api"] = "auto"
    allow_remote: bool = False
    allow_remote_images: bool = False

    @field_validator("paper_ids")
    @classmethod
    def unique_papers(cls, value):
        return list(dict.fromkeys(value))

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, value):
        if value is not None and (len(value) != 4 or not all(0 <= v <= 1 for v in value) or value[0] >= value[2] or value[1] >= value[3]):
            raise ValueError("bbox 须为归一化的 [x0,y0,x1,y1] 有效区域")
        return value


class Citation(Strict):
    evidence_id: str
    quote: str = Field(min_length=2, max_length=1000)


class Finding(Strict):
    category: Literal["problem", "method", "contribution", "metric", "dataset", "baseline", "ablation", "limitation", "figure"]
    statement: str = Field(min_length=1, max_length=1600)
    research_implication: str = Field(default="", max_length=900)
    citations: list[Citation] = Field(default_factory=list, max_length=5)
    basis: Literal["author_reported", "model_inference"] = "author_reported"


class Findings(Strict):
    findings: list[Finding] = Field(min_length=1, max_length=8)


class EvaluationFinding(Finding):
    category: Literal["metric", "dataset", "baseline", "ablation"]


class EvaluationFindings(Strict):
    findings: list[EvaluationFinding] = Field(min_length=1, max_length=6)


class ClaimAssessment(Strict):
    finding_id: str
    status: Literal["supported", "partial", "unsupported", "uncertain"]
    reason: str = Field(max_length=500)


class Assessments(Strict):
    assessments: list[ClaimAssessment] = Field(min_length=1, max_length=16)


class DirectionAssessment(Strict):
    direction_id: str
    recommendation: Literal["explore_after_checks", "revise_first"]
    concerns: list[str] = Field(max_length=5)
    next_check: str


class DirectionAssessments(Strict):
    assessments: list[DirectionAssessment] = Field(min_length=1, max_length=3)


class TestableRelation(Strict):
    candidate: str = Field(min_length=1, max_length=150)
    reference: str = Field(min_length=1, max_length=150)
    metric: str = Field(min_length=1, max_length=150)
    metric_direction: Literal["lower", "higher"]
    expectation: Literal["better", "not_worse", "different", "exploratory"]
    condition: str = Field(min_length=1, max_length=250)


class Experiment(Strict):
    relation: TestableRelation
    hypothesis: str
    setup: str
    baselines: list[str]
    metrics: list[str]
    ablations: list[str]
    falsification: str
    feasibility: str


class Direction(Strict):
    title: str
    motivation: str
    change: str
    assumptions: list[str]
    evidence_ids: list[str] = Field(default_factory=list)
    experiment: Experiment
    novelty_queries: list[str]
    risks: list[str]


class Directions(Strict):
    directions: list[Direction] = Field(min_length=1, max_length=3)
    open_questions: list[str]


class VisualTarget(Strict):
    paper_id: str
    page: int = Field(ge=1)
    region: Literal["full", "top", "bottom"] = "full"


class Route(Strict):
    queries: list[str] = Field(min_length=1, max_length=4)
    visual_targets: list[VisualTarget] = Field(default_factory=list, max_length=2)
    rationale: str


class VisualReading(Strict):
    description: str
    visible_text: list[str]
    uncertainties: list[str]
