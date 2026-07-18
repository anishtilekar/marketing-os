"""PackagingAgent — final campaign assembly and delivery packaging.

The :class:`PackagingAgent` receives a :class:`PackagingRequest` — the fully
approved campaign artifacts together with their passing
:class:`~marketingos.agents.qa.QAReport` — and produces a
:class:`CampaignPackage`: the manifest, metadata, asset index with
checksums, README summary and the final archive reference, all organised in
the MarketingOS run structure.

Scope and guarantees
--------------------
* **QA gate.** Packaging is only reachable with a non-failed QA report: the
  :class:`PackagingRequest` model refuses to be constructed from a failed
  report, and the agent re-checks the gate at run time as defence in depth.
* **Coordination only.** All filesystem and compression work is delegated
  to an injected tool satisfying :class:`PackagingServicePort` (staging
  files into the run structure, computing checksums, producing the final
  archive). The agent composes *what* goes where; the service does the IO.
* **Deterministic run structure.** Every packaged file lives under
  ``{root_prefix}/{run_id}/`` in a fixed layout::

      manifest.json                     package manifest
      metadata.json                     package metadata
      README.md                         human-readable summary
      qa_report.json                    the QA verdict shipped with the run
      content/*.json                    approved campaign artifacts
      assets/images/{item_id}.{ext}     post creatives
      assets/videos/{item_id}.{ext}     short-form videos

* **Verifiable output.** Every index entry carries the checksum reported by
  the packaging service, and the :class:`CampaignPackage` model validates
  that the manifest's counts agree with the index.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PermanentAgentError,
    PromptRepository,
    ToolRegistry,
)
from marketingos.agents.planner import REQUIRED_POSTS, REQUIRED_VIDEOS
from marketingos.agents.qa import CampaignBundle, QAReport, QAStatus

__all__ = [
    "AssetIndexEntry",
    "AssetKind",
    "CampaignPackage",
    "Checksum",
    "PackageArchiveRef",
    "PackageManifest",
    "PackageMetadata",
    "PackagingAgent",
    "PackagingAgentConfig",
    "PackagingRequest",
    "PackagingServicePort",
    "QAGateNotPassedError",
    "StagedFile",
]


# ---------------------------------------------------------------------------
# Packaging service contract (port)
# ---------------------------------------------------------------------------


class Checksum(BaseModel):
    """A content checksum computed by the packaging service."""

    model_config = ConfigDict(frozen=True)

    algorithm: str = Field(default="sha256", min_length=1)
    value: str = Field(pattern=r"^[0-9a-fA-F]{16,128}$")


class StagedFile(BaseModel):
    """The packaging service's record of one file staged into the run."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1, description="Path inside the run structure.")
    size_bytes: int = Field(ge=0)
    checksum: Checksum
    media_type: str = Field(min_length=1)


class PackageArchiveRef(BaseModel):
    """Reference to the final, compressed campaign archive."""

    model_config = ConfigDict(frozen=True)

    uri: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    checksum: Checksum
    media_type: str = Field(default="application/zip", min_length=1)


@runtime_checkable
class PackagingServicePort(Protocol):
    """Structural contract for the packaging service.

    Satisfied by the packaging service in ``marketingos.services``. The
    service owns every filesystem and compression concern: fetching source
    assets, writing files into the run structure, computing checksums and
    producing the final archive. The agent depends only on this protocol.
    """

    async def stage_asset(self, *, source_uri: str, target_path: str) -> StagedFile:
        """Copy an existing asset into the run structure."""
        ...

    async def stage_text(
        self, *, content: str, target_path: str, media_type: str
    ) -> StagedFile:
        """Write generated text (manifest, metadata, README) into the run."""
        ...

    async def finalize(self, *, root_path: str) -> PackageArchiveRef:
        """Compress the completed run structure and return the archive."""
        ...


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class PackagingRequest(BaseModel):
    """Typed input pairing the approved campaign with its QA report.

    Construction enforces the QA gate: a failed report is rejected, as is a
    report that was produced for a different campaign.
    """

    model_config = ConfigDict(frozen=True)

    bundle: CampaignBundle
    qa_report: QAReport

    @model_validator(mode="after")
    def _validate_gate(self) -> "PackagingRequest":
        """Reject unapproved or mismatched packaging requests."""
        if self.qa_report.status is QAStatus.FAILED:
            raise ValueError(
                "campaign failed QA and cannot be packaged; resolve the "
                "report's errors first"
            )
        if self.qa_report.source_plan_run_id != self.bundle.week_plan.run_id:
            raise ValueError(
                "QA report audits plan run "
                f"{self.qa_report.source_plan_run_id!r}, not the bundled "
                f"plan {self.bundle.week_plan.run_id!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class AssetKind(StrEnum):
    """What an asset-index entry points at."""

    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"


class AssetIndexEntry(BaseModel):
    """One packaged file: identity, provenance, location and checksum."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    kind: AssetKind
    item_id: str | None = Field(
        default=None,
        description="Plan item the asset belongs to; None for documents.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Original asset location; None for generated documents.",
    )
    packaged_path: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    checksum: Checksum


class PackageManifest(BaseModel):
    """Machine-readable description of the package and its provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="1.0", min_length=1)
    package_run_id: str
    subject: str
    source_context_run_id: str
    source_strategy_run_id: str
    source_plan_run_id: str
    source_caption_run_id: str
    source_creative_run_id: str
    source_video_run_id: str
    qa_run_id: str
    media_asset_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    created_at: datetime


class PackageMetadata(BaseModel):
    """Descriptive metadata shipped alongside the manifest."""

    model_config = ConfigDict(frozen=True)

    generator: str = Field(default="MarketingOS", min_length=1)
    subject: str
    qa_status: QAStatus
    qa_run_id: str
    post_count: int = Field(ge=0)
    video_count: int = Field(ge=0)
    created_at: datetime


class CampaignPackage(BaseModel):
    """The final, delivery-ready campaign package.

    Construction validates internal consistency: unique packaged paths,
    unique asset ids and manifest counts that agree with the asset index.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    subject: str
    root_path: str = Field(min_length=1)
    manifest: PackageManifest
    metadata: PackageMetadata
    asset_index: tuple[AssetIndexEntry, ...] = Field(min_length=1)
    readme: str = Field(
        min_length=1, description="The README summary shipped in the package."
    )
    archive: PackageArchiveRef
    created_at: datetime

    @model_validator(mode="after")
    def _validate_consistency(self) -> "CampaignPackage":
        """The manifest, index and archive must describe the same package."""
        paths = [entry.packaged_path for entry in self.asset_index]
        if len(set(paths)) != len(paths):
            raise ValueError("packaged paths must be unique")
        asset_ids = [entry.asset_id for entry in self.asset_index]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("asset index ids must be unique")
        media = sum(
            1
            for entry in self.asset_index
            if entry.kind in (AssetKind.IMAGE, AssetKind.VIDEO)
        )
        documents = sum(
            1 for entry in self.asset_index if entry.kind is AssetKind.DOCUMENT
        )
        if self.manifest.media_asset_count != media:
            raise ValueError(
                f"manifest declares {self.manifest.media_asset_count} media "
                f"assets, index contains {media}"
            )
        if self.manifest.document_count != documents:
            raise ValueError(
                f"manifest declares {self.manifest.document_count} "
                f"documents, index contains {documents}"
            )
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QAGateNotPassedError(PermanentAgentError):
    """Packaging was invoked for a campaign whose QA report failed.

    Permanent: retrying cannot approve a failed campaign; the orchestration
    layer must resolve the QA findings and re-audit first.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PackagingAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`PackagingAgent`."""

    root_prefix: str = Field(
        default="runs",
        min_length=1,
        description="Top-level prefix of the MarketingOS run structure.",
    )
    manifest_schema_version: str = Field(default="1.0", min_length=1)


#: File extension by media type for packaged binary assets.
_EXTENSIONS: Final[dict[str, str]] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}
_FALLBACK_EXTENSION: Final[str] = ".bin"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PackagingAgent(BaseAgent[PackagingRequest, CampaignPackage]):
    """Assembles the approved campaign into a delivery-ready package.

    Workflow:

    1. Re-check the QA gate (defence in depth over the input validator).
    2. Stage every media asset (5 post images, 2 videos) into the run
       structure through the packaging service, concurrently.
    3. Stage the campaign documents: the approved artifacts as JSON, the QA
       report, the generated README and metadata.
    4. Build the manifest from the staged results and stage it last, so it
       describes the completed run.
    5. Ask the service to finalise (compress) the run and return the
       archive reference.
    6. Assemble the :class:`CampaignPackage`, whose construction verifies
       manifest/index consistency.

    Transient service failures (timeouts, connection errors) surface as
    retryable errors through the base agent's normalisation, so packaging
    retries as a whole and the run structure is always rebuilt completely.
    """

    def __init__(
        self,
        *,
        packaging_service: PackagingServicePort,
        name: str | None = None,
        config: PackagingAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            packaging_service: Service owning all filesystem and
                compression work.
            name: Logical agent name; defaults to the class name.
            config: Packaging-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository (unused by the built-in
                README composer; available to subclasses).
        """
        settings = config or PackagingAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._service = packaging_service

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: PackagingRequest, *, run_id: str) -> CampaignPackage:
        """Assemble, stage, index and archive the approved campaign.

        Args:
            payload: The approved campaign artifacts and their QA report.
            run_id: Identifier of this execution.

        Returns:
            The :class:`CampaignPackage` referencing every packaged artifact.

        Raises:
            QAGateNotPassedError: If the QA report is failed (only possible
                for requests built without validation).
        """
        if payload.qa_report.status is QAStatus.FAILED:
            raise QAGateNotPassedError(
                "Campaign failed QA and cannot be packaged.",
                agent_name=self.name,
                run_id=run_id,
            )

        bundle = payload.bundle
        root = f"{self._settings.root_prefix}/{run_id}"
        created_at = datetime.now(UTC)

        media_entries = await asyncio.gather(
            *(
                self._stage_media(
                    asset_id=creative.asset.asset_id,
                    kind=AssetKind.IMAGE,
                    item_id=creative.item_id,
                    source_uri=creative.asset.uri,
                    media_type=creative.asset.media_type,
                    target_dir=f"{root}/assets/images",
                )
                for creative in bundle.creatives.creatives
            ),
            *(
                self._stage_media(
                    asset_id=video.asset.asset_id,
                    kind=AssetKind.VIDEO,
                    item_id=video.item_id,
                    source_uri=video.asset.uri,
                    media_type=video.asset.media_type,
                    target_dir=f"{root}/assets/videos",
                )
                for video in bundle.videos.videos
            ),
        )

        readme = self._compose_readme(bundle, payload.qa_report, created_at)
        metadata = PackageMetadata(
            subject=bundle.week_plan.subject,
            qa_status=payload.qa_report.status,
            qa_run_id=payload.qa_report.run_id,
            post_count=REQUIRED_POSTS,
            video_count=REQUIRED_VIDEOS,
            created_at=created_at,
        )
        document_entries = list(
            await asyncio.gather(
                self._stage_document(
                    "doc-business-context",
                    content=bundle.business_context.model_dump_json(indent=2),
                    path=f"{root}/content/business_context.json",
                ),
                self._stage_document(
                    "doc-strategy",
                    content=bundle.strategy.model_dump_json(indent=2),
                    path=f"{root}/content/strategy.json",
                ),
                self._stage_document(
                    "doc-week-plan",
                    content=bundle.week_plan.model_dump_json(indent=2),
                    path=f"{root}/content/week_plan.json",
                ),
                self._stage_document(
                    "doc-captions",
                    content=bundle.captions.model_dump_json(indent=2),
                    path=f"{root}/content/captions.json",
                ),
                self._stage_document(
                    "doc-qa-report",
                    content=payload.qa_report.model_dump_json(indent=2),
                    path=f"{root}/qa_report.json",
                ),
                self._stage_document(
                    "doc-metadata",
                    content=metadata.model_dump_json(indent=2),
                    path=f"{root}/metadata.json",
                ),
                self._stage_document(
                    "doc-readme",
                    content=readme,
                    path=f"{root}/README.md",
                    media_type="text/markdown",
                ),
            )
        )

        manifest = PackageManifest(
            schema_version=self._settings.manifest_schema_version,
            package_run_id=run_id,
            subject=bundle.week_plan.subject,
            source_context_run_id=bundle.business_context.run_id,
            source_strategy_run_id=bundle.strategy.run_id,
            source_plan_run_id=bundle.week_plan.run_id,
            source_caption_run_id=bundle.captions.run_id,
            source_creative_run_id=bundle.creatives.run_id,
            source_video_run_id=bundle.videos.run_id,
            qa_run_id=payload.qa_report.run_id,
            media_asset_count=len(media_entries),
            document_count=len(document_entries) + 1,
            created_at=created_at,
        )
        document_entries.append(
            await self._stage_document(
                "doc-manifest",
                content=manifest.model_dump_json(indent=2),
                path=f"{root}/manifest.json",
            )
        )

        archive = await self._service.finalize(root_path=root)
        package = CampaignPackage(
            run_id=run_id,
            subject=bundle.week_plan.subject,
            root_path=root,
            manifest=manifest,
            metadata=metadata,
            asset_index=(*media_entries, *document_entries),
            readme=readme,
            archive=archive,
            created_at=created_at,
        )
        self._logger.bind(
            run_id=run_id,
            event="packaging.packaged",
            media_assets=manifest.media_asset_count,
            documents=manifest.document_count,
            archive_uri=archive.uri,
        ).info("Campaign packaged")
        return package

    # -- staging helpers -------------------------------------------------------------

    async def _stage_media(
        self,
        *,
        asset_id: str,
        kind: AssetKind,
        item_id: str,
        source_uri: str,
        media_type: str,
        target_dir: str,
    ) -> AssetIndexEntry:
        """Stage one media asset into the run structure and index it."""
        extension = _EXTENSIONS.get(media_type, _FALLBACK_EXTENSION)
        staged = await self._service.stage_asset(
            source_uri=source_uri,
            target_path=f"{target_dir}/{item_id}{extension}",
        )
        return AssetIndexEntry(
            asset_id=asset_id,
            kind=kind,
            item_id=item_id,
            source_uri=source_uri,
            packaged_path=staged.path,
            media_type=staged.media_type,
            size_bytes=staged.size_bytes,
            checksum=staged.checksum,
        )

    async def _stage_document(
        self,
        asset_id: str,
        *,
        content: str,
        path: str,
        media_type: str = "application/json",
    ) -> AssetIndexEntry:
        """Stage one generated document into the run structure and index it."""
        staged = await self._service.stage_text(
            content=content, target_path=path, media_type=media_type
        )
        return AssetIndexEntry(
            asset_id=asset_id,
            kind=AssetKind.DOCUMENT,
            packaged_path=staged.path,
            media_type=staged.media_type,
            size_bytes=staged.size_bytes,
            checksum=staged.checksum,
        )

    # -- README composition -----------------------------------------------------------

    @staticmethod
    def _compose_readme(
        bundle: CampaignBundle, qa_report: QAReport, created_at: datetime
    ) -> str:
        """Compose the human-readable package summary from approved content."""
        schedule_lines = [
            f"- Day {item.day} ({item.publish_time.isoformat(timespec='minutes')}, "
            f"{item.platform.value}): {item.format.value.replace('_', ' ')} — "
            f"{item.topic}"
            for item in sorted(
                bundle.week_plan.items, key=lambda item: (item.day, item.publish_time)
            )
        ]
        pillar_lines = [
            f"- {pillar.name}: {pillar.description}"
            for pillar in bundle.strategy.content_pillars
        ]
        return "\n".join(
            (
                f"# {bundle.week_plan.subject} — First-Week Campaign Package",
                "",
                f"Generated by MarketingOS on {created_at.date().isoformat()}.",
                f"QA status: {qa_report.status.value}.",
                "",
                "## Contents",
                "",
                f"- {REQUIRED_POSTS} social media posts (assets/images/)",
                f"- {REQUIRED_VIDEOS} short-form videos (assets/videos/)",
                "- Approved strategy, plan, captions, and business context "
                "(content/)",
                "- QA report (qa_report.json) and package manifest "
                "(manifest.json)",
                "",
                "## Positioning",
                "",
                bundle.strategy.positioning,
                "",
                "## Content pillars",
                "",
                *pillar_lines,
                "",
                "## Publishing schedule",
                "",
                *schedule_lines,
                "",
            )
        )