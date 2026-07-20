"""Tests for the packaging QA gate becoming advisory.

The behavioural change under test: a ``failed`` QA report no longer blocks
packaging — the report ships *with* the package instead — while the report's
own verdict is unchanged (errors still derive a ``failed`` status; QA stays
honest, it just stops being a hard gate). The structural consistency check
(the report must audit the bundled plan) is retained.

``PackagingRequest._validate_gate`` reads only ``qa_report.status``,
``qa_report.source_plan_run_id`` and ``bundle.week_plan.run_id``, so it is
exercised directly against a real :class:`QAReport` plus a lightweight
stand-in for the bundle. Building a fully valid :class:`CampaignBundle`
(~18 nested models with a strict provenance chain) would be disproportionate
for a validator that never inspects the rest of it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from marketingos.agents.packaging import PackagingRequest
from marketingos.agents.qa import (
    CheckCategory,
    CheckOutcome,
    CheckResult,
    Finding,
    QAReport,
    QAStatus,
    Severity,
    derive_status,
)

_PLAN_RUN_ID = "plan-run-1"


def _report(*, severity: Severity, plan_run_id: str = _PLAN_RUN_ID) -> QAReport:
    """A real QAReport carrying one finding of ``severity``."""
    finding = Finding(
        id="Q1",
        category=CheckCategory.BRAND_SAFETY,
        severity=severity,
        code="guarantee",
        message="Caption makes an unsupportable 'guaranteed' claim.",
    )
    return QAReport(
        run_id="qa-run-1",
        subject="Acme Coffee",
        source_context_run_id="ctx-1",
        source_strategy_run_id="strategy-1",
        source_plan_run_id=plan_run_id,
        source_caption_run_id="caption-1",
        source_creative_run_id="creative-1",
        source_video_run_id="video-1",
        status=derive_status((finding,)),
        findings=(finding,),
        checks=(
            CheckResult(
                name="brand_safety",
                category=CheckCategory.BRAND_SAFETY,
                outcome=CheckOutcome.FAILED,
            ),
        ),
        model_reviewed=True,
        created_at=datetime.now(UTC),
    )


def _request_stub(report: QAReport, *, week_plan_run_id: str) -> SimpleNamespace:
    """The minimal shape ``_validate_gate`` reads off a PackagingRequest."""
    return SimpleNamespace(
        qa_report=report,
        bundle=SimpleNamespace(week_plan=SimpleNamespace(run_id=week_plan_run_id)),
    )


def test_failed_qa_report_no_longer_blocks_packaging() -> None:
    report = _report(severity=Severity.ERROR)
    assert report.status is QAStatus.FAILED  # the verdict is still "failed"

    stub = _request_stub(report, week_plan_run_id=_PLAN_RUN_ID)

    # The advisory gate returns the request unchanged rather than raising.
    assert PackagingRequest._validate_gate(stub) is stub


def test_passed_with_warnings_still_packages() -> None:
    report = _report(severity=Severity.WARNING)
    assert report.status is QAStatus.PASSED_WITH_WARNINGS

    stub = _request_stub(report, week_plan_run_id=_PLAN_RUN_ID)

    assert PackagingRequest._validate_gate(stub) is stub


def test_report_auditing_a_different_plan_is_still_rejected() -> None:
    # Structural consistency is NOT relaxed: a report that audits a different
    # plan than the one bundled is still a wiring bug and must be rejected.
    report = _report(severity=Severity.ERROR, plan_run_id="some-other-plan")
    stub = _request_stub(report, week_plan_run_id=_PLAN_RUN_ID)

    with pytest.raises(ValueError, match="audits plan run"):
        PackagingRequest._validate_gate(stub)


def test_qa_verdict_derivation_is_unchanged() -> None:
    # Guards the honest-reporting invariant: making the gate advisory must not
    # quietly downgrade how severities map to a verdict.
    error = Finding(
        id="Q1",
        category=CheckCategory.BRAND_SAFETY,
        severity=Severity.ERROR,
        code="guarantee",
        message="Unsupportable claim.",
    )
    warning = Finding(
        id="Q2",
        category=CheckCategory.PLATFORM_FIT,
        severity=Severity.WARNING,
        code="too_many_hashtags",
        message="Too many hashtags.",
    )
    info = Finding(
        id="Q3",
        category=CheckCategory.GROUNDING,
        severity=Severity.INFO,
        code="note",
        message="Advisory note.",
    )

    assert derive_status((error, warning)) is QAStatus.FAILED
    assert derive_status((warning, info)) is QAStatus.PASSED_WITH_WARNINGS
    assert derive_status((info,)) is QAStatus.PASSED
    assert derive_status(()) is QAStatus.PASSED
