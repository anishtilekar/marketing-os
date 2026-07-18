"""QAAgent — the campaign quality gate before packaging.

The :class:`QAAgent` receives a :class:`CampaignBundle` — every artifact the
pipeline produced, from :class:`~marketingos.models.business_context.BusinessContext`
through to the rendered :class:`~marketingos.agents.video_director.VideoPackage` —
and produces a :class:`QAReport`: a verdict plus the structured critique the
orchestration layer feeds back into the revise loops, and the gate
:class:`~marketingos.agents.packaging.PackagingAgent` refuses to run without.

Scope and guarantees
--------------------
* **The audit itself is deterministic.** Every hard constraint — content
  mix, per-day distribution, one caption per plan item, one creative per
  post, one video per video item, pillar anchoring, citation grounding,
  asset-reference integrity, budget ceiling — is checked in Python against
  real ids and real numbers. No language model participates in any pass/fail
  decision, so the verdict is reproducible and testable with golden files.
* **The optional model review can only add findings.** When a
  :class:`~marketingos.agents.business_analysis.LanguageModelPort` is
  injected, it performs a *brand-safety and tone* review whose output is
  narrowed to advisory findings before it is merged. A model can never
  clear a deterministic error, and it can never upgrade a report to
  ``PASSED``.
* **A false pass is structurally impossible.** :class:`QAReport` derives its
  own status from its findings in a validator: any ``ERROR`` finding forces
  ``FAILED``, any ``WARNING`` forces ``PASSED_WITH_WARNINGS``. Constructing
  a passing report that carries errors raises.
* **Findings, not exceptions.** A campaign that fails the audit is a
  successful QA execution returning a failed report. Exceptions are reserved
  for QA's *own* failures: an unusable model response (retryable) or a
  missing budget ledger (permanent).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PermanentAgentError,
    PromptRepository,
    RetryableAgentError,
    ToolRegistry,
)
from marketingos.agents.business_analysis import (
    LanguageModelPort,
    extract_json_object,
)
from marketingos.agents.copywriter import CaptionPackage
from marketingos.agents.designer import CreativePackage
from marketingos.agents.planner import (
    REQUIRED_POSTS,
    REQUIRED_VIDEOS,
    WEEK_DAYS,
    ContentFormat,
    Platform,
    WeekPlan,
)
from marketingos.agents.strategist import Strategy
from marketingos.agents.video_director import VideoPackage
from marketingos.models.business_context import BusinessContext

__all__ = [
    "BudgetLedgerPort",
    "BudgetSnapshot",
    "CampaignBundle",
    "CheckCategory",
    "CheckOutcome",
    "CheckResult",
    "Finding",
    "MalformedReviewError",
    "MissingBudgetLedgerError",
    "QAAgent",
    "QAAgentConfig",
    "QAReport",
    "QAStatus",
    "Severity",
]


# ---------------------------------------------------------------------------
# Budget contract (port)
# ---------------------------------------------------------------------------


class BudgetSnapshot(BaseModel):
    """The run's spend position at the moment QA read it."""

    model_config = ConfigDict(frozen=True)

    total_spend: Decimal = Field(ge=Decimal("0"))
    max_budget: Decimal = Field(ge=Decimal("0"))
    warning_threshold: Decimal | None = Field(
        default=None,
        ge=Decimal("0"),
        description="Absolute amount at which a run is considered close to "
        "its ceiling; None disables the warning check.",
    )
    currency: str = Field(min_length=3, max_length=3)
    entry_count: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining(self) -> Decimal:
        """Budget left before the ceiling; negative if it was breached."""
        return self.max_budget - self.total_spend


@runtime_checkable
class BudgetLedgerPort(Protocol):
    """Structural contract for the run's cost ledger.

    Satisfied by the cost-ledger service in ``marketingos.services``. QA
    depends only on the ability to *read* a snapshot: it never records,
    adjusts or refunds spend.
    """

    async def snapshot(self) -> BudgetSnapshot:
        """Return the current spend position for the run under audit."""
        ...


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class CampaignBundle(BaseModel):
    """Every artifact produced for one campaign, ready to be audited.

    Construction validates the *provenance chain* — each artifact must
    declare that it was derived from the artifact bundled with it. A bundle
    assembled from two different runs is a wiring bug, and it is rejected
    here rather than producing a meaningless audit.
    """

    model_config = ConfigDict(frozen=True)

    business_context: BusinessContext
    strategy: Strategy
    week_plan: WeekPlan
    captions: CaptionPackage
    creatives: CreativePackage
    videos: VideoPackage

    @model_validator(mode="after")
    def _validate_chain(self) -> CampaignBundle:
        """Reject bundles assembled from mismatched pipeline runs."""
        expected: tuple[tuple[str, str, str, str], ...] = (
            (
                "strategy",
                self.strategy.source_context_run_id,
                "business context",
                self.business_context.run_id,
            ),
            (
                "week plan",
                self.week_plan.source_strategy_run_id,
                "strategy",
                self.strategy.run_id,
            ),
            (
                "caption package",
                self.captions.source_plan_run_id,
                "week plan",
                self.week_plan.run_id,
            ),
            (
                "creative package",
                self.creatives.source_plan_run_id,
                "week plan",
                self.week_plan.run_id,
            ),
            (
                "creative package",
                self.creatives.source_caption_run_id,
                "caption package",
                self.captions.run_id,
            ),
            (
                "video package",
                self.videos.source_plan_run_id,
                "week plan",
                self.week_plan.run_id,
            ),
            (
                "video package",
                self.videos.source_caption_run_id,
                "caption package",
                self.captions.run_id,
            ),
            (
                "video package",
                self.videos.source_creative_run_id,
                "creative package",
                self.creatives.run_id,
            ),
        )
        for artifact, declared, upstream, actual in expected:
            if declared != actual:
                raise ValueError(
                    f"{artifact} was produced for {upstream} run "
                    f"{declared!r}, not the bundled {actual!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class QAStatus(StrEnum):
    """The verdict of one audit."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class Severity(StrEnum):
    """How much weight a finding carries.

    ``ERROR`` blocks packaging. ``WARNING`` does not block, but is surfaced
    to the approver. ``INFO`` is advisory only.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckCategory(StrEnum):
    """The constraint family a finding belongs to."""

    CONTENT_MIX = "content_mix"
    COVERAGE = "coverage"
    GROUNDING = "grounding"
    ASSET_INTEGRITY = "asset_integrity"
    PLATFORM_FIT = "platform_fit"
    BRAND_SAFETY = "brand_safety"
    BUDGET = "budget"


class CheckOutcome(StrEnum):
    """Whether a named check ran, and how it ended."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Finding(BaseModel):
    """One thing the audit found, in the terms needed to fix it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^Q\d+$")
    category: CheckCategory
    severity: Severity
    code: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable machine-readable identifier for this class of "
        "finding, e.g. 'ungrounded_goal'. Contract tests assert on codes, "
        "never on message wording.",
    )
    message: str = Field(min_length=1, max_length=1000)
    item_id: str | None = Field(
        default=None, description="Plan item the finding concerns, if any."
    )
    asset_id: str | None = Field(
        default=None, description="Asset the finding concerns, if any."
    )
    remediation: str | None = Field(
        default=None,
        max_length=1000,
        description="What the responsible agent should change on its "
        "revise pass.",
    )


class CheckResult(BaseModel):
    """The outcome of one named check, recorded whether or not it fired."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    category: CheckCategory
    outcome: CheckOutcome
    finding_count: int = Field(default=0, ge=0)
    detail: str | None = Field(
        default=None,
        max_length=500,
        description="Why a check was skipped, when it was.",
    )


class QAReport(BaseModel):
    """The audit verdict, its critique, and what was checked to reach it.

    The status is *derived*, not asserted: construction recomputes it from
    the findings and rejects any report whose declared status disagrees.
    That is what makes a false pass a construction error rather than a
    silent shipping incident.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    subject: str
    source_context_run_id: str
    source_strategy_run_id: str
    source_plan_run_id: str
    source_caption_run_id: str
    source_creative_run_id: str
    source_video_run_id: str
    status: QAStatus
    findings: tuple[Finding, ...] = ()
    checks: tuple[CheckResult, ...] = Field(min_length=1)
    budget: BudgetSnapshot | None = None
    model_reviewed: bool = Field(
        default=False,
        description="Whether an optional language-model brand-safety "
        "review contributed advisory findings.",
    )
    created_at: datetime

    @model_validator(mode="after")
    def _validate_status_and_ids(self) -> QAReport:
        """Findings determine the status; finding ids must be unique."""
        ids = [finding.id for finding in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError("finding ids must be unique")
        derived = derive_status(self.findings)
        if self.status is not derived:
            raise ValueError(
                f"status {self.status.value!r} disagrees with the findings, "
                f"which derive {derived.value!r}"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def errors(self) -> tuple[Finding, ...]:
        """Every blocking finding."""
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Every non-blocking finding that still needs an approver's eye."""
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def revision_feedback(self) -> str:
        """The critique, formatted for a revise-pass prompt.

        Empty when the campaign passed cleanly.
        """
        actionable = [
            f for f in self.findings if f.severity is not Severity.INFO
        ]
        if not actionable:
            return ""
        lines = []
        for finding in actionable:
            scope = f" [{finding.item_id}]" if finding.item_id else ""
            fix = f" Fix: {finding.remediation}" if finding.remediation else ""
            lines.append(
                f"- ({finding.severity.value}) {finding.code}{scope}: "
                f"{finding.message}{fix}"
            )
        return "\n".join(lines)


def derive_status(findings: tuple[Finding, ...]) -> QAStatus:
    """Return the only status a set of findings supports.

    Args:
        findings: Every finding produced by an audit.

    Returns:
        ``FAILED`` if any finding is an error, ``PASSED_WITH_WARNINGS`` if
        any is a warning, ``PASSED`` otherwise.
    """
    if any(finding.severity is Severity.ERROR for finding in findings):
        return QAStatus.FAILED
    if any(finding.severity is Severity.WARNING for finding in findings):
        return QAStatus.PASSED_WITH_WARNINGS
    return QAStatus.PASSED


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MalformedReviewError(RetryableAgentError):
    """The optional model review returned output QA could not use.

    Retryable: the deterministic audit is unaffected, and a regenerated
    review usually parses.
    """


class MissingBudgetLedgerError(PermanentAgentError):
    """Budget auditing was required but no ledger was injected.

    Permanent: retrying cannot conjure a ledger; the wiring must be fixed.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class QAAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`QAAgent`."""

    require_budget_audit: bool = Field(
        default=True,
        description="Fail closed when no budget ledger is injected, rather "
        "than silently skipping the spend check.",
    )
    enable_model_review: bool = Field(
        default=True,
        description="Run the advisory brand-safety review when a language "
        "model is injected. Deterministic checks always run.",
    )
    max_model_findings: int = Field(
        default=10,
        ge=0,
        description="Upper bound on advisory findings accepted from the "
        "model review, so a verbose response cannot flood the report.",
    )
    min_video_duration_seconds: float = Field(default=5.0, gt=0.0)
    max_video_duration_seconds: float = Field(default=90.0, gt=0.0)
    duration_tolerance_seconds: float = Field(
        default=1.5,
        ge=0.0,
        description="Allowed drift between a video's planned duration and "
        "the rendered asset's reported duration.",
    )
    min_alt_text_length: int = Field(default=15, ge=0)
    blocked_terms: tuple[str, ...] = Field(
        default=(),
        description="Additional client- or campaign-specific terms that "
        "must not appear in any copy, matched case-insensitively.",
    )
    system_prompt_template: str = Field(
        default="qa/system",
        description="PromptRepository template name for the review system "
        "prompt; used when a repository is injected, otherwise the "
        "built-in default prompt applies.",
    )

    @model_validator(mode="after")
    def _validate_duration_bounds(self) -> QAAgentConfig:
        """The duration window must be non-empty."""
        if self.min_video_duration_seconds >= self.max_video_duration_seconds:
            raise ValueError(
                "min_video_duration_seconds must be below "
                "max_video_duration_seconds"
            )
        return self


#: Maximum caption length accepted by each platform, in characters.
_CAPTION_LIMITS: Final[dict[Platform, int]] = {
    Platform.INSTAGRAM: 2200,
    Platform.FACEBOOK: 2200,
    Platform.TIKTOK: 2200,
    Platform.YOUTUBE: 5000,
    Platform.LINKEDIN: 3000,
    Platform.X: 280,
}

#: Maximum hashtag count that reads as intentional rather than spam.
_HASHTAG_LIMITS: Final[dict[Platform, int]] = {
    Platform.INSTAGRAM: 15,
    Platform.FACEBOOK: 5,
    Platform.TIKTOK: 10,
    Platform.YOUTUBE: 8,
    Platform.LINKEDIN: 5,
    Platform.X: 3,
}

#: Unsupportable claims. These are regulatory and reputational risks, not
#: stylistic preferences, so they are errors rather than warnings.
_UNSUPPORTED_CLAIM_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (r"\b(guarantee[ds]?|guaranteed results?)\b", "guarantee"),
    (r"\b(cure|cures|heals?|treats?)\s+(all|any|every)\b", "medical_claim"),
    (r"\b(100\s*%|totally|completely)\s+(safe|risk[-\s]?free|effective)\b",
     "absolute_safety_claim"),
    (r"\bno\.?\s*1\b|(?<!\w)#\s?1(?!\d)|\bnumber one\b", "ranking_claim"),
    (r"\b(best|cheapest|fastest|only)\s+(in|on)\s+(the\s+)?"
     r"(world|country|market|city)\b", "superlative_claim"),
    (r"\b(free money|get rich|double your (money|income))\b",
     "financial_claim"),
    (r"\b(miracle|instantly? (?:cures?|fixes?))\b", "miracle_claim"),
)

#: Built-in review system prompt, used when no PromptRepository is injected.
_DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a brand-safety reviewer inside an automated marketing system. "
    "You receive a JSON object with a business subject, its positioning, "
    "and the finished copy for a week of content: headlines, captions, "
    "calls to action, hashtags, and video scripts.\n"
    "Review the copy for brand safety and tone only.\n"
    "Respond with exactly one JSON object of the form:\n"
    '{"findings": [{"item_id": "C1", "code": "off_brand_tone", '
    '"message": "...", "remediation": "..."}]}\n'
    "Rules:\n"
    "- Report only: offensive, discriminatory, or unsafe language; claims "
    "the copy cannot support; tone that contradicts the stated "
    "positioning; and copy that would embarrass the business.\n"
    "- Do not report spelling, formatting, hashtag counts, caption "
    "length, content counts, or scheduling — those are checked "
    "elsewhere and duplicate findings are discarded.\n"
    "- item_id must be an id present in the input, or omitted if the "
    "finding concerns the campaign as a whole.\n"
    "- code must be lower_snake_case.\n"
    "- Return an empty findings array if the copy is clean. Do not invent "
    "problems to appear thorough.\n"
    "- Output JSON only, with no surrounding text."
)


# ---------------------------------------------------------------------------
# Finding collection
# ---------------------------------------------------------------------------


class _FindingCollector:
    """Accumulates findings, assigning stable sequential ids.

    Kept internal: the report is the public artifact, and ids must be
    allocated by exactly one writer to stay unique and ordered.
    """

    def __init__(self) -> None:
        self._findings: list[Finding] = []

    def add(
        self,
        severity: Severity,
        category: CheckCategory,
        code: str,
        message: str,
        *,
        item_id: str | None = None,
        asset_id: str | None = None,
        remediation: str | None = None,
    ) -> None:
        """Record one finding under the next available id."""
        self._findings.append(
            Finding(
                id=f"Q{len(self._findings) + 1}",
                category=category,
                severity=severity,
                code=code,
                message=message,
                item_id=item_id,
                asset_id=asset_id,
                remediation=remediation,
            )
        )

    def count(self) -> int:
        """Return how many findings have been recorded so far."""
        return len(self._findings)

    def result(self) -> tuple[Finding, ...]:
        """Return every finding, in the order it was recorded."""
        return tuple(self._findings)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class QAAgent(BaseAgent[CampaignBundle, QAReport]):
    """Audits a finished campaign and issues the packaging gate verdict.

    Workflow:

    1. Run every deterministic check against the bundle, in a fixed order,
       recording a :class:`CheckResult` for each one whether it fires or
       not — so the report proves what was inspected, not only what broke.
    2. Read the budget position from the injected ledger and check it
       against the run's ceiling.
    3. Optionally ask a language model for an advisory brand-safety and
       tone review, narrowed to ``INFO`` and ``WARNING`` findings.
    4. Assemble the :class:`QAReport`, whose construction derives the
       status from the findings.

    The agent is read-only with respect to the campaign: it never edits,
    repairs or regenerates an artifact. Fixing findings is the responsible
    upstream agent's revise pass, driven by
    :attr:`QAReport.revision_feedback`.
    """

    def __init__(
        self,
        *,
        budget_ledger: BudgetLedgerPort | None = None,
        llm: LanguageModelPort | None = None,
        name: str | None = None,
        config: QAAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            budget_ledger: Read-only view of the run's spend. Required
                unless ``config.require_budget_audit`` is disabled.
            llm: Optional language model for the advisory brand-safety
                review. Without it the audit is fully deterministic.
            name: Logical agent name; defaults to the class name.
            config: QA-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository; overrides the built-in
                review prompt when it provides
                ``config.system_prompt_template``.
        """
        settings = config or QAAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._budget_ledger = budget_ledger
        self._llm = llm

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: CampaignBundle, *, run_id: str) -> QAReport:
        """Audit the campaign and return its verdict.

        Args:
            payload: Every artifact produced for one campaign.
            run_id: Identifier of this execution.

        Returns:
            The :class:`QAReport`, whose status is derived from its
            findings. A failing campaign is a successful execution.

        Raises:
            MissingBudgetLedgerError: If the budget audit is required but no
                ledger was injected.
            MalformedReviewError: If the optional model review returns
                output that cannot be parsed.
        """
        findings = _FindingCollector()
        checks: list[CheckResult] = [
            self._check(
                "content_mix",
                CheckCategory.CONTENT_MIX,
                findings,
                lambda: self._check_content_mix(payload, findings),
            ),
            self._check(
                "coverage",
                CheckCategory.COVERAGE,
                findings,
                lambda: self._check_coverage(payload, findings),
            ),
            self._check(
                "grounding",
                CheckCategory.GROUNDING,
                findings,
                lambda: self._check_grounding(payload, findings),
            ),
            self._check(
                "asset_integrity",
                CheckCategory.ASSET_INTEGRITY,
                findings,
                lambda: self._check_asset_integrity(payload, findings),
            ),
            self._check(
                "platform_fit",
                CheckCategory.PLATFORM_FIT,
                findings,
                lambda: self._check_platform_fit(payload, findings),
            ),
            self._check(
                "brand_safety",
                CheckCategory.BRAND_SAFETY,
                findings,
                lambda: self._check_brand_safety(payload, findings),
            ),
        ]

        budget_check, snapshot = await self._audit_budget(findings, run_id=run_id)
        checks.append(budget_check)

        review_check, reviewed = await self._model_review(
            payload, findings, run_id=run_id
        )
        checks.append(review_check)

        report = QAReport(
            run_id=run_id,
            subject=payload.week_plan.subject,
            source_context_run_id=payload.business_context.run_id,
            source_strategy_run_id=payload.strategy.run_id,
            source_plan_run_id=payload.week_plan.run_id,
            source_caption_run_id=payload.captions.run_id,
            source_creative_run_id=payload.creatives.run_id,
            source_video_run_id=payload.videos.run_id,
            status=derive_status(findings.result()),
            findings=findings.result(),
            checks=tuple(checks),
            budget=snapshot,
            model_reviewed=reviewed,
            created_at=datetime.now(UTC),
        )

        self._logger.bind(
            run_id=run_id,
            event="qa.audited",
            status=report.status.value,
            errors=len(report.errors),
            warnings=len(report.warnings),
            checks=len(report.checks),
        ).info("Campaign audited")
        return report

    # -- check harness -----------------------------------------------------------

    @staticmethod
    def _check(
        name: str,
        category: CheckCategory,
        findings: _FindingCollector,
        body: Callable[[], None],
    ) -> CheckResult:
        """Run one deterministic check and record what it produced."""
        before = findings.count()
        body()
        produced = findings.count() - before
        return CheckResult(
            name=name,
            category=category,
            outcome=CheckOutcome.FAILED if produced else CheckOutcome.PASSED,
            finding_count=produced,
        )

    # -- deterministic checks -----------------------------------------------------

    def _check_content_mix(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Re-verify the 5-post/2-video mix and the one-item-per-day spread.

        The upstream models already enforce this. QA checks it again on
        purpose: an audit that trusts its input cannot catch a regression
        in the thing it exists to guarantee.
        """
        items = bundle.week_plan.items
        posts = sum(1 for item in items if item.format is ContentFormat.POST)
        videos = sum(
            1 for item in items if item.format is ContentFormat.SHORT_FORM_VIDEO
        )
        if posts != REQUIRED_POSTS or videos != REQUIRED_VIDEOS:
            findings.add(
                Severity.ERROR,
                CheckCategory.CONTENT_MIX,
                "wrong_content_mix",
                f"Plan contains {posts} posts and {videos} short-form "
                f"videos; exactly {REQUIRED_POSTS} and {REQUIRED_VIDEOS} "
                "are required.",
                remediation="Regenerate the week plan with the required mix.",
            )
        days = sorted(item.day for item in items)
        if days != list(range(1, WEEK_DAYS + 1)):
            findings.add(
                Severity.ERROR,
                CheckCategory.CONTENT_MIX,
                "uneven_day_distribution",
                f"Plan must schedule exactly one item on each of days "
                f"1-{WEEK_DAYS}; got days {days}.",
                remediation="Reschedule so every day carries exactly one item.",
            )
        if len(bundle.creatives.creatives) != REQUIRED_POSTS:
            findings.add(
                Severity.ERROR,
                CheckCategory.CONTENT_MIX,
                "wrong_creative_count",
                f"Expected {REQUIRED_POSTS} post creatives, got "
                f"{len(bundle.creatives.creatives)}.",
            )
        if len(bundle.videos.videos) != REQUIRED_VIDEOS:
            findings.add(
                Severity.ERROR,
                CheckCategory.CONTENT_MIX,
                "wrong_video_count",
                f"Expected {REQUIRED_VIDEOS} videos, got "
                f"{len(bundle.videos.videos)}.",
            )

    def _check_coverage(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Every plan item must carry the deliverables its format requires."""
        plan_items = {item.id: item for item in bundle.week_plan.items}
        caption_ids = {caption.item_id for caption in bundle.captions.captions}
        creative_ids = {
            creative.item_id for creative in bundle.creatives.creatives
        }
        video_ids = {video.item_id for video in bundle.videos.videos}

        for missing in sorted(set(plan_items) - caption_ids):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "missing_caption",
                f"Plan item {missing} has no caption.",
                item_id=missing,
                remediation="Have the copywriter write copy for this item.",
            )
        for unknown in sorted(caption_ids - set(plan_items)):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "orphan_caption",
                f"Caption {unknown} does not correspond to any plan item.",
                item_id=unknown,
            )

        expected_creatives = {
            item_id
            for item_id, item in plan_items.items()
            if item.format is ContentFormat.POST
        }
        expected_videos = {
            item_id
            for item_id, item in plan_items.items()
            if item.format is ContentFormat.SHORT_FORM_VIDEO
        }
        for missing in sorted(expected_creatives - creative_ids):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "missing_creative",
                f"Post item {missing} has no creative.",
                item_id=missing,
            )
        for missing in sorted(expected_videos - video_ids):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "missing_video",
                f"Video item {missing} has no rendered video.",
                item_id=missing,
            )
        for wrong in sorted(creative_ids - expected_creatives):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "creative_for_non_post",
                f"Creative {wrong} is attached to an item that is not a post.",
                item_id=wrong,
            )
        for wrong in sorted(video_ids - expected_videos):
            findings.add(
                Severity.ERROR,
                CheckCategory.COVERAGE,
                "video_for_non_video_item",
                f"Video {wrong} is attached to an item that is not a "
                "short-form video.",
                item_id=wrong,
            )

    def _check_grounding(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Citations must resolve, and pillars must anchor to the strategy.

        This is the fact/assumption separation constraint expressed as an
        audit: a strategy may build on an assumption, but only on one that
        exists and is labelled as such in the business context.
        """
        fact_ids = {str(fact.id) for fact in bundle.business_context.observed_facts}
        assumption_ids = {str(a.id) for a in bundle.business_context.assumptions}
        known = fact_ids | assumption_ids

        cited: list[tuple[str, tuple[str, ...]]] = [
            (f"goal {index + 1}", goal.grounded_in)
            for index, goal in enumerate(bundle.strategy.goals)
        ]
        cited.append(("target audience", bundle.strategy.target_audience.grounded_in))
        cited.extend(
            (f"content pillar {pillar.name!r}", pillar.grounded_in)
            for pillar in bundle.strategy.content_pillars
        )
        for label, citations in cited:
            unknown = sorted(set(citations) - known)
            if unknown:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.GROUNDING,
                    "ungrounded_citation",
                    f"Strategy {label} cites context ids that do not exist: "
                    f"{unknown}.",
                    remediation="Re-derive the strategy citing only real "
                    "fact and assumption ids.",
                )

        pillar_names = {pillar.name for pillar in bundle.strategy.content_pillars}
        for item in bundle.week_plan.items:
            if item.content_pillar not in pillar_names:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.GROUNDING,
                    "unanchored_pillar",
                    f"Plan item {item.id} names content pillar "
                    f"{item.content_pillar!r}, which is not in the strategy.",
                    item_id=item.id,
                    remediation="Use one of: "
                    f"{sorted(pillar_names)}.",
                )

        if not fact_ids:
            findings.add(
                Severity.ERROR,
                CheckCategory.GROUNDING,
                "no_observed_facts",
                "Business context contains no observed facts; the entire "
                "campaign would rest on assumptions.",
            )
        elif len(assumption_ids) > len(fact_ids):
            findings.add(
                Severity.WARNING,
                CheckCategory.GROUNDING,
                "assumption_heavy_context",
                f"Business context carries {len(assumption_ids)} "
                f"assumptions against {len(fact_ids)} observed facts; the "
                "campaign leans more on inference than on evidence.",
                remediation="Broaden research before approving, or accept "
                "the inference risk explicitly.",
            )

    def _check_asset_integrity(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Assets must be uniquely identified, locatable and well formed."""
        creative_assets = {
            creative.asset.asset_id: creative.asset
            for creative in bundle.creatives.creatives
        }
        seen: set[str] = set()
        for creative in bundle.creatives.creatives:
            asset = creative.asset
            if asset.asset_id in seen:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.ASSET_INTEGRITY,
                    "duplicate_asset_id",
                    f"Asset id {asset.asset_id} is used by more than one "
                    "creative.",
                    item_id=creative.item_id,
                    asset_id=asset.asset_id,
                )
            seen.add(asset.asset_id)
            if len(creative.alt_text.strip()) < self._settings.min_alt_text_length:
                findings.add(
                    Severity.WARNING,
                    CheckCategory.ASSET_INTEGRITY,
                    "thin_alt_text",
                    f"Creative for {creative.item_id} has alt text shorter "
                    f"than {self._settings.min_alt_text_length} characters, "
                    "which is not usefully descriptive.",
                    item_id=creative.item_id,
                    asset_id=asset.asset_id,
                    remediation="Describe what is visible in the image, not "
                    "the headline.",
                )

        for video in bundle.videos.videos:
            asset = video.asset
            if asset.asset_id in seen:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.ASSET_INTEGRITY,
                    "duplicate_asset_id",
                    f"Asset id {asset.asset_id} is shared with another "
                    "asset in the campaign.",
                    item_id=video.item_id,
                    asset_id=asset.asset_id,
                )
            seen.add(asset.asset_id)

            unknown = sorted(
                set(video.direction.asset_references) - set(creative_assets)
            )
            if unknown:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.ASSET_INTEGRITY,
                    "unknown_asset_reference",
                    f"Video {video.item_id} reuses asset ids that are not "
                    f"in the approved creative package: {unknown}.",
                    item_id=video.item_id,
                )

            planned = video.direction.total_duration_seconds
            if not (
                self._settings.min_video_duration_seconds
                <= planned
                <= self._settings.max_video_duration_seconds
            ):
                findings.add(
                    Severity.ERROR,
                    CheckCategory.ASSET_INTEGRITY,
                    "video_duration_out_of_range",
                    f"Video {video.item_id} runs {planned}s, outside the "
                    f"{self._settings.min_video_duration_seconds}-"
                    f"{self._settings.max_video_duration_seconds}s window "
                    "for short-form content.",
                    item_id=video.item_id,
                    asset_id=asset.asset_id,
                )
            if asset.duration_seconds is not None:
                drift = abs(asset.duration_seconds - planned)
                if drift > self._settings.duration_tolerance_seconds:
                    findings.add(
                        Severity.WARNING,
                        CheckCategory.ASSET_INTEGRITY,
                        "duration_drift",
                        f"Video {video.item_id} was directed at {planned}s "
                        f"but the rendered asset reports "
                        f"{asset.duration_seconds}s.",
                        item_id=video.item_id,
                        asset_id=asset.asset_id,
                    )
            last_subtitle = max(
                line.end_seconds for line in video.direction.subtitles
            )
            if last_subtitle > planned + self._settings.duration_tolerance_seconds:
                findings.add(
                    Severity.WARNING,
                    CheckCategory.ASSET_INTEGRITY,
                    "subtitles_overrun",
                    f"Video {video.item_id} has subtitles running to "
                    f"{last_subtitle}s, past its {planned}s duration.",
                    item_id=video.item_id,
                )

    def _check_platform_fit(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Copy must fit the platform it is scheduled to publish on."""
        platforms = {item.id: item.platform for item in bundle.week_plan.items}
        for caption in bundle.captions.captions:
            platform = platforms.get(caption.item_id)
            if platform is None:
                continue  # already reported as an orphan caption
            limit = _CAPTION_LIMITS[platform]
            length = len(caption.caption)
            if length > limit:
                findings.add(
                    Severity.ERROR,
                    CheckCategory.PLATFORM_FIT,
                    "caption_too_long",
                    f"Caption for {caption.item_id} is {length} characters; "
                    f"{platform.value} allows {limit}.",
                    item_id=caption.item_id,
                    remediation=f"Cut roughly {length - limit} characters.",
                )
            hashtag_limit = _HASHTAG_LIMITS[platform]
            if len(caption.hashtags) > hashtag_limit:
                findings.add(
                    Severity.WARNING,
                    CheckCategory.PLATFORM_FIT,
                    "too_many_hashtags",
                    f"Caption for {caption.item_id} carries "
                    f"{len(caption.hashtags)} hashtags; more than "
                    f"{hashtag_limit} reads as spam on "
                    f"{platform.value}.",
                    item_id=caption.item_id,
                )

    def _check_brand_safety(
        self, bundle: CampaignBundle, findings: _FindingCollector
    ) -> None:
        """Scan every published surface for unsupportable or blocked copy.

        The scan covers what an audience would actually see: headlines,
        captions, calls to action, hashtags, scripts and subtitles. Image
        prompts are excluded — they are production inputs, not published
        text.
        """
        blocked = tuple(
            term.strip().lower()
            for term in self._settings.blocked_terms
            if term.strip()
        )
        for surface, item_id, text in self._published_text(bundle):
            lowered = text.lower()
            for pattern, code in _UNSUPPORTED_CLAIM_PATTERNS:
                match = re.search(pattern, lowered)
                if match:
                    findings.add(
                        Severity.ERROR,
                        CheckCategory.BRAND_SAFETY,
                        code,
                        f"{surface} for {item_id} makes an unsupportable "
                        f"claim ({match.group(0).strip()!r}).",
                        item_id=item_id,
                        remediation="Rewrite the claim so it states only "
                        "what the business context supports.",
                    )
            for term in blocked:
                if term in lowered:
                    findings.add(
                        Severity.ERROR,
                        CheckCategory.BRAND_SAFETY,
                        "blocked_term",
                        f"{surface} for {item_id} contains the blocked term "
                        f"{term!r}.",
                        item_id=item_id,
                        remediation=f"Remove {term!r} from the copy.",
                    )

    @staticmethod
    def _published_text(
        bundle: CampaignBundle,
    ) -> tuple[tuple[str, str, str], ...]:
        """Return every (surface, item id, text) an audience would read."""
        surfaces: list[tuple[str, str, str]] = []
        for caption in bundle.captions.captions:
            surfaces.extend(
                (
                    ("Headline", caption.item_id, caption.headline),
                    ("Caption", caption.item_id, caption.caption),
                    ("Call to action", caption.item_id, caption.call_to_action),
                    ("Hashtags", caption.item_id, " ".join(caption.hashtags)),
                )
            )
        for video in bundle.videos.videos:
            surfaces.append(("Script", video.item_id, video.direction.script))
            surfaces.append(
                (
                    "Subtitles",
                    video.item_id,
                    " ".join(line.text for line in video.direction.subtitles),
                )
            )
        return tuple(surfaces)

    # -- budget audit -----------------------------------------------------------

    async def _audit_budget(
        self, findings: _FindingCollector, *, run_id: str
    ) -> tuple[CheckResult, BudgetSnapshot | None]:
        """Compare recorded spend against the run's ceiling.

        Raises:
            MissingBudgetLedgerError: If the audit is required but no
                ledger was injected.
        """
        if self._budget_ledger is None:
            if self._settings.require_budget_audit:
                raise MissingBudgetLedgerError(
                    "QA requires a budget ledger to audit spend; inject one "
                    "or disable require_budget_audit explicitly.",
                    agent_name=self.name,
                    run_id=run_id,
                )
            return (
                CheckResult(
                    name="budget",
                    category=CheckCategory.BUDGET,
                    outcome=CheckOutcome.SKIPPED,
                    detail="No budget ledger injected and the audit is "
                    "not required.",
                ),
                None,
            )

        before = findings.count()
        snapshot = await self._budget_ledger.snapshot()
        if snapshot.total_spend > snapshot.max_budget:
            findings.add(
                Severity.ERROR,
                CheckCategory.BUDGET,
                "budget_exceeded",
                f"Run spent {snapshot.total_spend} {snapshot.currency} "
                f"against a ceiling of {snapshot.max_budget} "
                f"{snapshot.currency}.",
                remediation="Investigate the cost ledger before shipping; "
                "the cost guard should have refused the overspending call.",
            )
        elif (
            snapshot.warning_threshold is not None
            and snapshot.total_spend >= snapshot.warning_threshold
        ):
            findings.add(
                Severity.WARNING,
                CheckCategory.BUDGET,
                "budget_near_ceiling",
                f"Run spent {snapshot.total_spend} {snapshot.currency}, "
                f"leaving {snapshot.remaining} {snapshot.currency} of the "
                f"{snapshot.max_budget} {snapshot.currency} ceiling.",
            )
        produced = findings.count() - before
        return (
            CheckResult(
                name="budget",
                category=CheckCategory.BUDGET,
                outcome=CheckOutcome.FAILED if produced else CheckOutcome.PASSED,
                finding_count=produced,
            ),
            snapshot,
        )

    # -- optional model review ---------------------------------------------------

    async def _model_review(
        self, bundle: CampaignBundle, findings: _FindingCollector, *, run_id: str
    ) -> tuple[CheckResult, bool]:
        """Merge advisory brand-safety findings from a language model.

        The model's output is narrowed on the way in: severity is fixed at
        ``WARNING``, unknown item ids are dropped to campaign scope, and the
        number of accepted findings is capped. It cannot clear a
        deterministic error or change the derived status to ``PASSED``.

        Raises:
            MalformedReviewError: If the response cannot be parsed.
        """
        if self._llm is None or not self._settings.enable_model_review:
            return (
                CheckResult(
                    name="model_brand_review",
                    category=CheckCategory.BRAND_SAFETY,
                    outcome=CheckOutcome.SKIPPED,
                    detail="No language model injected."
                    if self._llm is None
                    else "Model review disabled by configuration.",
                ),
                False,
            )

        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(bundle),
        )
        try:
            data = extract_json_object(raw)
            entries = data.get("findings", [])
            if not isinstance(entries, list):
                raise ValueError("'findings' must be a JSON array")
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedReviewError(
                f"Brand-safety review returned unusable output: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc

        known_items = {item.id for item in bundle.week_plan.items}
        before = findings.count()
        accepted = 0
        for entry in entries:
            if accepted >= self._settings.max_model_findings:
                break
            if not isinstance(entry, dict):
                continue
            message = str(entry.get("message", "")).strip()
            if not message:
                continue
            code = str(entry.get("code", "")).strip().lower()
            code = re.sub(r"[^a-z0-9_]", "_", code).strip("_")
            if not code or not code[0].isalpha():
                code = "model_review_finding"
            item_id = entry.get("item_id")
            item_id = item_id if item_id in known_items else None
            remediation = str(entry.get("remediation", "")).strip() or None
            findings.add(
                Severity.WARNING,
                CheckCategory.BRAND_SAFETY,
                code,
                message[:1000],
                item_id=item_id,
                remediation=remediation,
            )
            accepted += 1

        produced = findings.count() - before
        self._logger.bind(
            run_id=run_id,
            event="qa.reviewed",
            proposed=len(entries),
            accepted=produced,
        ).debug("Advisory brand-safety review merged")
        return (
            CheckResult(
                name="model_brand_review",
                category=CheckCategory.BRAND_SAFETY,
                outcome=CheckOutcome.FAILED if produced else CheckOutcome.PASSED,
                finding_count=produced,
            ),
            True,
        )

    # -- prompt construction ------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt, or the built-in one."""
        if self.prompts is not None:
            return self.load_prompt(self._settings.system_prompt_template)
        return _DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _user_prompt(bundle: CampaignBundle) -> str:
        """Serialise the reviewable copy as compact JSON.

        Only published copy is sent: the review is about what an audience
        would read, and narrowing the payload keeps the call cheap and the
        model on task.
        """
        platforms = {item.id: item.platform.value for item in bundle.week_plan.items}
        scripts = {
            video.item_id: video.direction.script for video in bundle.videos.videos
        }
        return json.dumps(
            {
                "subject": bundle.week_plan.subject,
                "positioning": bundle.strategy.positioning,
                "brand_voice": bundle.captions.brand_voice,
                "items": [
                    {
                        "item_id": caption.item_id,
                        "platform": platforms.get(caption.item_id),
                        "headline": caption.headline,
                        "caption": caption.caption,
                        "call_to_action": caption.call_to_action,
                        "hashtags": list(caption.hashtags),
                        "script": scripts.get(caption.item_id),
                    }
                    for caption in bundle.captions.captions
                ],
            },
            ensure_ascii=False,
        )

    # -- cost -----------------------------------------------------------------

    def estimate_cost(self, payload: CampaignBundle, result: QAReport) -> float:
        """Return the flat configured cost; QA's spend is the review call.

        The deterministic audit costs nothing, so an execution with no model
        review is free.
        """
        if not result.model_reviewed:
            return 0.0
        return self._settings.base_cost_per_run_usd
