"""Typed data models for loaded prompt packages.

This module has a single responsibility: define the plain-data shapes that
:class:`~marketingos.prompts.loader.PromptLoader` and
:class:`~marketingos.prompts.versioning.PromptVersionResolver` read and
return. It contains **no** filesystem access, rendering, or orchestration
logic, and depends on nothing else in the prompts package, keeping it a true
leaf module both components can import without risking a cycle.

Shapes
------
* :class:`PromptAsset` — one loaded template file (``system.jinja`` or
  ``user.jinja``): its filename and raw, unrendered text.
* :class:`PromptMetadata` — the parsed, optional ``metadata.yaml`` sidecar.
  Every field is optional since the file itself is optional, and unrecognised
  keys are preserved rather than rejected, since template authors routinely
  extend it with agent-specific fields (tool requirements, I/O schemas,
  caching policy, and so on).
* :class:`PromptTemplate` — a fully loaded prompt package: whichever of the
  system/user assets are present, plus metadata. At least one asset must be
  present, mirroring :class:`~marketingos.prompts.loader.PromptLoader`'s own
  directory-structure check.
* :class:`PromptVersion` — one discovered, on-disk version directory for an
  agent, as found by :class:`~marketingos.prompts.versioning.PromptVersionResolver`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "PromptAsset",
    "PromptMetadata",
    "PromptTemplate",
    "PromptVersion",
]


class PromptAsset(BaseModel):
    """One loaded template file: its filename and raw, unrendered text."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1)
    content: str


class PromptMetadata(BaseModel):
    """Optional descriptive metadata for a prompt package.

    Parsed from the optional ``metadata.yaml`` sidecar. All fields are
    optional and unrecognised keys are kept (``extra="allow"``) rather than
    rejected, since richer templates attach fields this model does not name
    explicitly (see e.g. the ``research`` template's ``input_schema`` /
    ``output_schema`` / ``guardrails``).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    agent: str | None = None
    version: str | None = None
    description: str | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tags: tuple[str, ...] = ()
    recommended_model: str | None = None
    model_preferences: tuple[str, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    expected_input: str | None = None
    expected_output: str | None = None


class PromptTemplate(BaseModel):
    """A fully loaded prompt package: its assets and metadata.

    At least one of ``system`` or ``user`` must be present. This mirrors the
    check :class:`~marketingos.prompts.loader.PromptLoader` already performs
    against the directory structure, enforced again here as a model-level
    invariant, defence in depth against a future caller constructing a
    :class:`PromptTemplate` directly.
    """

    model_config = ConfigDict(frozen=True)

    system: PromptAsset | None = None
    user: PromptAsset | None = None
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)

    @model_validator(mode="after")
    def _require_at_least_one_asset(self) -> "PromptTemplate":
        if self.system is None and self.user is None:
            raise ValueError(
                "PromptTemplate requires at least one of 'system' or 'user'."
            )
        return self


class PromptVersion(BaseModel):
    """One discovered, on-disk prompt version directory for an agent."""

    model_config = ConfigDict(frozen=True)

    agent: str = Field(min_length=1)
    version: str = Field(min_length=1)
    path: Path
