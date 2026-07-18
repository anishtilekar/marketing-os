"""Central prompt registry for MarketingOS.

This module owns a single, focused responsibility: turning the on-disk prompt
library into rendered prompt strings. It performs **discovery, loading,
version resolution, rendering, and caching** — and nothing else. It contains
no orchestration logic and knows nothing about the agents that consume it,
keeping :class:`PromptRegistry` fully decoupled from agent implementations.

On-disk layout
--------------
The registry expects the following structure, rooted at the ``prompts``
package (the directory containing this file)::

    prompts/
        registry.py
        templates/
            <agent>/
                <version>/                # e.g. ``v1``, ``v2``
                    <kind>.jinja          # e.g. ``system.jinja``, ``user.jinja``
                    <kind>.<locale>.jinja # optional localized variant
        policies/
            <name>.jinja                  # shared, reusable prompt fragments

A concrete template is addressed by four coordinates:

* ``agent``   — the top-level directory (``strategist``, ``research``, ...).
* ``version`` — a version directory beneath the agent (``v1``, ``v2``, ...).
* ``kind``    — the file stem (``system``, ``user``, ...).
* ``locale``  — an optional, dotted filename suffix (``system.fr.jinja``).

Reference strings
-----------------
So that the registry satisfies the ``PromptRepository`` protocol declared in
:mod:`marketingos.agents.base` — ``render(template_name, /, **variables)`` —
templates can be addressed with a compact reference string parsed by
:meth:`PromptReference.parse`::

    "strategist/system"            # default (latest) version, base locale
    "strategist/v2/system"         # explicit version
    "copywriter/v1/user@fr"        # explicit version + locale
    "research/system@es"           # default version + locale

Design guarantees
-----------------
* **Read-through caching** — template sources are read from disk and compiled
  at most once per ``(agent, version, kind, locale)`` coordinate; subsequent
  requests are served from memory. :meth:`PromptRegistry.invalidate_cache`
  drops the caches without forgetting the discovered file set;
  :meth:`PromptRegistry.reload` performs a full re-discovery.
* **Thread-safe reads** — discovery and every cache mutation are serialised
  through a reentrant lock, and Jinja rendering of compiled templates is
  itself thread-safe, so a single shared registry can back many concurrent
  agent executions.
* **Strict variable validation** — missing template variables raise a precise
  :class:`MissingPromptVariableError` rather than silently rendering empty
  strings (the environment uses ``StrictUndefined``).
* **Extensibility** — the resolution seam (default-version selection, locale
  fallback, filesystem lookup) is expressed through small overridable methods,
  leaving room for future multi-tenant overrides and A/B experiments without
  changing the public API.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    Template,
    TemplateError,
    TemplateSyntaxError,
    Undefined,
    meta,
)
from jinja2.exceptions import UndefinedError
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from marketingos.exceptions import MarketingOSError

if TYPE_CHECKING:
    from loguru import Logger

__all__ = [
    "DEFAULT_POLICY_SUFFIXES",
    "DEFAULT_TEMPLATE_SUFFIXES",
    "InvalidPromptReferenceError",
    "MissingPromptVariableError",
    "PromptAgentNotFoundError",
    "PromptDirectoryNotFoundError",
    "PromptError",
    "PromptPolicyError",
    "PromptReference",
    "PromptRegistry",
    "PromptRenderError",
    "PromptTemplate",
    "PromptTemplateNotFoundError",
    "PromptVersionNotFoundError",
    "get_prompt_registry",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Filename suffixes recognised as renderable Jinja templates.
DEFAULT_TEMPLATE_SUFFIXES: Final[tuple[str, ...]] = (".jinja", ".j2")

#: Additional suffixes recognised as shared policy fragments (loaded verbatim).
DEFAULT_POLICY_SUFFIXES: Final[tuple[str, ...]] = (".txt", ".md")

#: The Jinja global under which loaded shared policies are exposed to templates.
POLICIES_GLOBAL: Final[str] = "policies"

_REFERENCE_SEPARATOR: Final[str] = "/"
_LOCALE_SEPARATOR: Final[str] = "@"
_LOCALE_STEM_SEPARATOR: Final[str] = "."
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^v(\d+)$")

_DEFAULT_BASE_DIR: Final[Path] = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class PromptError(MarketingOSError):
    """Base class for every error raised by the prompt registry."""


class PromptDirectoryNotFoundError(PromptError):
    """Raised when the templates directory is missing from the filesystem."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Prompt templates directory does not exist: {path}")
        self.path = Path(path)


class InvalidPromptReferenceError(PromptError):
    """Raised when a prompt reference string is syntactically invalid."""

    def __init__(self, reference: Any, reason: str) -> None:
        super().__init__(f"Invalid prompt reference {reference!r}: {reason}")
        self.reference = reference
        self.reason = reason


class PromptAgentNotFoundError(PromptError):
    """Raised when no templates are registered for the requested agent."""

    def __init__(self, agent: str, *, available: Iterable[str]) -> None:
        self.agent = agent
        self.available = tuple(available)
        known = ", ".join(self.available) or "<none>"
        super().__init__(
            f"No prompt templates registered for agent {agent!r}. "
            f"Known agents: {known}"
        )


class PromptVersionNotFoundError(PromptError):
    """Raised when the requested version does not exist for an agent."""

    def __init__(self, agent: str, version: str, *, available: Iterable[str]) -> None:
        self.agent = agent
        self.version = version
        self.available = tuple(available)
        known = ", ".join(self.available) or "<none>"
        super().__init__(
            f"Agent {agent!r} has no prompt version {version!r}. "
            f"Known versions: {known}"
        )


class PromptTemplateNotFoundError(PromptError):
    """Raised when no template matches the requested coordinates."""

    def __init__(
        self,
        agent: str,
        version: str,
        kind: str,
        *,
        locale: str | None = None,
        available: Iterable[str],
    ) -> None:
        self.agent = agent
        self.version = version
        self.kind = kind
        self.locale = locale
        self.available = tuple(available)
        known = ", ".join(self.available) or "<none>"
        super().__init__(
            f"No prompt template for agent={agent!r} version={version!r} "
            f"kind={kind!r} locale={locale!r}. Known kinds: {known}"
        )


class PromptRenderError(PromptError):
    """Raised when a template fails to compile or render."""

    def __init__(
        self,
        message: str,
        *,
        agent: str | None = None,
        version: str | None = None,
        kind: str | None = None,
        locale: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent = agent
        self.version = version
        self.kind = kind
        self.locale = locale


class MissingPromptVariableError(PromptRenderError):
    """Raised when required template variables are not supplied at render time."""

    def __init__(
        self,
        *,
        agent: str,
        version: str,
        kind: str,
        locale: str | None,
        missing: Iterable[str],
        required: Iterable[str],
    ) -> None:
        self.missing = tuple(sorted(missing))
        self.required = tuple(sorted(required))
        missing_str = ", ".join(self.missing) or "<none>"
        required_str = ", ".join(self.required) or "<none>"
        super().__init__(
            f"Missing variables for prompt {agent}/{version}/{kind}: "
            f"{missing_str}. Required variables: {required_str}",
            agent=agent,
            version=version,
            kind=kind,
            locale=locale,
        )


class PromptPolicyError(PromptError):
    """Raised when shared policy fragments cannot be loaded."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class PromptReference(BaseModel):
    """A parsed, validated address for a single prompt template.

    A reference is the string form agents pass through the ``PromptRepository``
    protocol. Its canonical grammar is::

        <agent> "/" [ <version> "/" ] <kind> [ "@" <locale> ]

    Attributes:
        agent: The owning agent (top-level template directory).
        kind: The template kind (file stem), e.g. ``system`` or ``user``.
        version: Explicit version, or ``None`` to defer to the registry's
            default-version resolution.
        locale: Explicit locale, or ``None`` for the base (unlocalized) variant.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    kind: str
    version: str | None = None
    locale: str | None = None

    @classmethod
    def parse(cls, reference: str) -> PromptReference:
        """Parse a reference string into a :class:`PromptReference`.

        Args:
            reference: A reference of the form ``"<agent>/<kind>"`` or
                ``"<agent>/<version>/<kind>"``, optionally suffixed with
                ``"@<locale>"``.

        Returns:
            The parsed, validated reference.

        Raises:
            InvalidPromptReferenceError: If ``reference`` is not a non-empty
                string, contains an unexpected number of path segments, or
                contains a segment that is not a safe identifier (guarding
                against path traversal and separator injection).
        """
        if not isinstance(reference, str):
            raise InvalidPromptReferenceError(
                reference, "reference must be a string"
            )
        body = reference.strip()
        if not body:
            raise InvalidPromptReferenceError(reference, "reference is empty")

        locale: str | None = None
        if _LOCALE_SEPARATOR in body:
            body, _, locale_part = body.partition(_LOCALE_SEPARATOR)
            locale = locale_part.strip()
            if not locale:
                raise InvalidPromptReferenceError(
                    reference, "locale suffix is empty"
                )

        segments = body.split(_REFERENCE_SEPARATOR)
        if len(segments) == 2:
            agent, kind = segments
            version: str | None = None
        elif len(segments) == 3:
            agent, version, kind = segments
        else:
            raise InvalidPromptReferenceError(
                reference,
                "expected '<agent>/<kind>' or '<agent>/<version>/<kind>'",
            )

        tokens = [agent, kind]
        if version is not None:
            tokens.append(version)
        if locale is not None:
            tokens.append(locale)
        for token in tokens:
            _validate_token(token, reference)

        return cls(agent=agent, kind=kind, version=version, locale=locale)

    def __str__(self) -> str:
        head = self.agent
        if self.version is not None:
            head = f"{head}{_REFERENCE_SEPARATOR}{self.version}"
        head = f"{head}{_REFERENCE_SEPARATOR}{self.kind}"
        if self.locale is not None:
            head = f"{head}{_LOCALE_SEPARATOR}{self.locale}"
        return head


class PromptTemplate(BaseModel):
    """An immutable, loaded prompt template with its metadata.

    Instances are pure data: they carry the raw template source together with
    everything needed to reason about it, but they do **not** render
    themselves. Rendering is owned by :class:`PromptRegistry`, which holds the
    corresponding compiled Jinja template and the shared policy globals.

    Attributes:
        agent: The owning agent.
        version: The resolved version this template was loaded from.
        kind: The template kind (file stem without locale suffix).
        locale: The resolved locale, or ``None`` for the base variant.
        path: Absolute filesystem path of the template file.
        source: The raw, unrendered template text.
        checksum: SHA-256 hex digest of ``source`` (for cache and audit use).
        required_variables: Variables the template references that must be
            supplied by the caller (Jinja globals such as ``policies`` are
            already excluded).
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    version: str
    kind: str
    locale: str | None = None
    path: Path
    source: str
    checksum: str
    required_variables: frozenset[str] = Field(default_factory=frozenset)

    @property
    def reference(self) -> str:
        """The fully-qualified, canonical reference string for this template."""
        head = f"{self.agent}{_REFERENCE_SEPARATOR}{self.version}{_REFERENCE_SEPARATOR}{self.kind}"
        if self.locale is not None:
            head = f"{head}{_LOCALE_SEPARATOR}{self.locale}"
        return head


# Internal coordinate types.
_Coordinate = tuple[str, str, str, str | None]  # (agent, version, kind, locale)
_KindKey = tuple[str, str | None]  # (kind, locale)
_Index = dict[str, dict[str, dict[_KindKey, Path]]]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """Discovers, loads, resolves, renders, and caches prompt templates.

    A registry is safe to share across threads and across concurrent agent
    executions. It is typically constructed once per process and injected into
    agents as the ``prompts`` collaborator; see :func:`get_prompt_registry`
    for the conventional process-wide instance.

    The registry satisfies the ``PromptRepository`` protocol declared in
    :mod:`marketingos.agents.base` through :meth:`render`, so an instance can
    be passed directly as ``BaseAgent(prompts=...)``.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        templates_subdir: str = "templates",
        policies_subdir: str = "policies",
        template_suffixes: Sequence[str] = DEFAULT_TEMPLATE_SUFFIXES,
        policy_suffixes: Sequence[str] = DEFAULT_POLICY_SUFFIXES,
        default_versions: Mapping[str, str] | None = None,
        strict_undefined: bool = True,
        auto_reload: bool = False,
    ) -> None:
        """Initialise the registry.

        Args:
            base_dir: Directory containing the ``templates`` and ``policies``
                subdirectories. Defaults to the directory of this module (the
                ``prompts`` package).
            templates_subdir: Name of the templates directory under ``base_dir``.
            policies_subdir: Name of the policies directory under ``base_dir``.
            template_suffixes: Filename suffixes treated as renderable
                templates during discovery.
            policy_suffixes: Extra suffixes (in addition to
                ``template_suffixes``) treated as shared policy fragments.
            default_versions: Optional per-agent version pins. When an agent is
                listed here, the pinned version is used whenever a caller omits
                an explicit version, overriding latest-version selection. A pin
                referencing a non-existent version surfaces as a
                :class:`PromptVersionNotFoundError` at resolution time.
            strict_undefined: When ``True`` (default), undefined variables raise
                at render time instead of rendering as empty strings.
            auto_reload: When ``True``, the Jinja environment re-checks template
                files for modifications on each access — convenient in
                development. Leave ``False`` in production and use
                :meth:`reload` to pick up changes explicitly.
        """
        self._base_dir: Final[Path] = Path(base_dir or _DEFAULT_BASE_DIR).resolve()
        self._templates_dir: Final[Path] = self._base_dir / templates_subdir
        self._policies_dir: Final[Path] = self._base_dir / policies_subdir
        self._template_suffixes: Final[tuple[str, ...]] = tuple(template_suffixes)
        self._policy_suffixes: Final[frozenset[str]] = frozenset(
            (*template_suffixes, *policy_suffixes)
        )
        self._default_versions: Final[Mapping[str, str]] = dict(
            default_versions or {}
        )

        self._lock = threading.RLock()
        self._index: _Index | None = None
        self._templates: dict[_Coordinate, PromptTemplate] = {}
        self._compiled: dict[_Coordinate, Template] = {}
        self._policies: dict[str, str] = {}
        self._policies_loaded: bool = False

        self._logger: Logger = logger.bind(component="PromptRegistry")

        self._environment = Environment(
            loader=FileSystemLoader(str(self._base_dir), followlinks=False),
            autoescape=False,  # prompts are plain text; HTML escaping corrupts them
            undefined=StrictUndefined if strict_undefined else Undefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            auto_reload=auto_reload,
            cache_size=400 if auto_reload else -1,
        )
        # Expose shared policies as an immutable global for every template.
        self._environment.globals[POLICIES_GLOBAL] = MappingProxyType(self._policies)

    # -- read-only accessors --------------------------------------------------

    @property
    def base_dir(self) -> Path:
        """The root directory holding ``templates`` and ``policies``."""
        return self._base_dir

    @property
    def templates_dir(self) -> Path:
        """The directory scanned for agent prompt templates."""
        return self._templates_dir

    @property
    def policies_dir(self) -> Path:
        """The directory scanned for shared policy fragments."""
        return self._policies_dir

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_dir={str(self._base_dir)!r})"

    # -- discovery ------------------------------------------------------------

    def list_agents(self) -> list[str]:
        """Return the sorted names of all agents that have templates."""
        return sorted(self._ensure_index())

    def list_versions(self, agent: str) -> list[str]:
        """Return the sorted versions available for ``agent``.

        Args:
            agent: The agent to inspect.

        Returns:
            The version directory names, sorted ascending.

        Raises:
            PromptAgentNotFoundError: If ``agent`` has no templates.
        """
        index = self._ensure_index()
        versions = index.get(agent)
        if versions is None:
            raise PromptAgentNotFoundError(agent, available=index)
        return sorted(versions)

    def list_kinds(self, agent: str, *, version: str | None = None) -> list[str]:
        """Return the sorted template kinds for an agent version.

        Args:
            agent: The agent to inspect.
            version: Explicit version, or ``None`` to use the default version.

        Returns:
            The distinct kinds (across all locales), sorted ascending.

        Raises:
            PromptAgentNotFoundError: If ``agent`` has no templates.
            PromptVersionNotFoundError: If the resolved version does not exist.
        """
        index = self._ensure_index()
        versions = index.get(agent)
        if versions is None:
            raise PromptAgentNotFoundError(agent, available=index)
        resolved = version or self._default_version(agent, versions)
        kinds = versions.get(resolved)
        if kinds is None:
            raise PromptVersionNotFoundError(agent, resolved, available=versions)
        return sorted({kind for (kind, _locale) in kinds})

    def exists(
        self,
        agent: str,
        version: str | None = None,
        *,
        kind: str | None = None,
        locale: str | None = None,
    ) -> bool:
        """Return whether templates exist for the given coordinates.

        This never raises: unknown coordinates simply yield ``False``.

        Args:
            agent: The agent to check.
            version: Optional version; ``None`` checks/uses the default version.
            kind: Optional kind; when omitted only agent/version presence is
                checked.
            locale: Optional locale; when supplied, a match on either the exact
                localized template or the base variant counts as present.

        Returns:
            ``True`` if a matching template (or, with ``kind`` omitted, the
            agent/version) exists.
        """
        try:
            index = self._ensure_index()
        except PromptDirectoryNotFoundError:
            return False

        versions = index.get(agent)
        if versions is None:
            return False
        if version is None and kind is None:
            return True

        try:
            resolved = version or self._default_version(agent, versions)
        except PromptVersionNotFoundError:
            return False
        kinds = versions.get(resolved)
        if kinds is None:
            return False
        if kind is None:
            return True
        if locale is not None:
            return (kind, locale) in kinds or (kind, None) in kinds
        return any(existing_kind == kind for (existing_kind, _locale) in kinds)

    # -- resolution and loading ----------------------------------------------

    def get_prompt(
        self,
        agent: str,
        kind: str,
        *,
        version: str | None = None,
        locale: str | None = None,
    ) -> PromptTemplate:
        """Resolve and load a single template, without rendering it.

        Args:
            agent: The owning agent.
            kind: The template kind (e.g. ``"system"`` or ``"user"``).
            version: Explicit version, or ``None`` for the default version.
            locale: Explicit locale, or ``None`` for the base variant. When a
                locale is requested but unavailable, the base variant is used
                as a fallback.

        Returns:
            The loaded :class:`PromptTemplate`, served from cache when possible.

        Raises:
            PromptAgentNotFoundError: If the agent has no templates.
            PromptVersionNotFoundError: If the resolved version is unknown.
            PromptTemplateNotFoundError: If no matching template exists.
            PromptRenderError: If the template source cannot be parsed.
        """
        reference = PromptReference(
            agent=agent, kind=kind, version=version, locale=locale
        )
        return self._resolve_and_load(reference)

    def render(self, template_name: str, /, **variables: Any) -> str:
        """Render a template addressed by a reference string.

        This is the ``PromptRepository`` protocol entrypoint used by agents via
        ``BaseAgent.load_prompt``.

        Args:
            template_name: A reference string (see :class:`PromptReference`).
            **variables: Substitution variables passed to the template.

        Returns:
            The rendered prompt text.

        Raises:
            InvalidPromptReferenceError: If ``template_name`` is malformed.
            PromptAgentNotFoundError: If the agent has no templates.
            PromptVersionNotFoundError: If the resolved version is unknown.
            PromptTemplateNotFoundError: If no matching template exists.
            MissingPromptVariableError: If a required variable is not supplied.
            PromptRenderError: If the template fails to compile or render.
        """
        reference = PromptReference.parse(template_name)
        template = self._resolve_and_load(reference)
        return self._render(template, variables)

    def get_policy(self, name: str) -> str:
        """Return the text of a shared policy fragment.

        Args:
            name: The policy file stem (filename without suffix).

        Returns:
            The policy text: rendered if the fragment carries a template
            suffix, verbatim otherwise.

        Raises:
            PromptPolicyError: If no policy with ``name`` is registered.
        """
        policies = self._ensure_policies()
        try:
            return policies[name]
        except KeyError as exc:
            known = ", ".join(sorted(policies)) or "<none>"
            raise PromptPolicyError(
                f"Unknown policy {name!r}. Known policies: {known}"
            ) from exc

    def list_policies(self) -> list[str]:
        """Return the sorted names of all registered shared policies."""
        return sorted(self._ensure_policies())

    # -- cache lifecycle ------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Drop cached template sources, compiled templates, and policies.

        The discovered file set (the index) is retained: the next request
        re-reads and re-compiles from disk using the same known paths. Use this
        after editing template *contents* in place. To pick up added or removed
        files, use :meth:`reload` instead.
        """
        with self._lock:
            self._templates.clear()
            self._compiled.clear()
            self._policies.clear()
            self._policies_loaded = False
            if self._environment.cache is not None:
                self._environment.cache.clear()
        self._logger.bind(event="prompt_registry.cache_invalidated").debug(
            "Prompt caches invalidated"
        )

    def reload(self) -> None:
        """Fully re-discover the prompt library and rebuild all caches.

        Clears the discovery index and every cache, then eagerly re-scans the
        filesystem and reloads shared policies. Use this after adding, removing,
        or renaming template or policy files.

        Raises:
            PromptDirectoryNotFoundError: If the templates directory is missing.
            PromptPolicyError: If shared policies cannot be loaded.
        """
        with self._lock:
            self.invalidate_cache()
            self._index = None
            index = self._ensure_index_locked()
            policies = self._ensure_policies_locked()
        self._logger.bind(
            event="prompt_registry.reloaded",
            agents=len(index),
            policies=len(policies),
        ).info("Prompt registry reloaded")

    # -- internal: index ------------------------------------------------------

    def _ensure_index(self) -> _Index:
        """Return the discovery index, building it once on first access."""
        index = self._index
        if index is not None:
            return index
        with self._lock:
            return self._ensure_index_locked()

    def _ensure_index_locked(self) -> _Index:
        """Return the index, building it under an already-held lock."""
        if self._index is None:
            self._index = self._discover()
        return self._index

    def _discover(self) -> _Index:
        """Scan the templates directory and build the coordinate index.

        Returns:
            A nested mapping ``{agent: {version: {(kind, locale): path}}}``
            containing only agents that have at least one template file.

        Raises:
            PromptDirectoryNotFoundError: If the templates directory is missing.
            PromptRenderError: If two files collapse to the same coordinate.
        """
        if not self._templates_dir.is_dir():
            raise PromptDirectoryNotFoundError(self._templates_dir)

        index: _Index = {}
        template_count = 0
        for agent_dir in self._iter_child_dirs(self._templates_dir):
            versions: dict[str, dict[_KindKey, Path]] = {}
            for version_dir in self._iter_child_dirs(agent_dir):
                kinds: dict[_KindKey, Path] = {}
                for file_path in sorted(version_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    if file_path.name.startswith("."):
                        continue
                    if file_path.suffix not in self._template_suffixes:
                        continue
                    key = self._parse_stem(file_path.stem)
                    if key in kinds:
                        raise PromptRenderError(
                            f"Duplicate prompt template for kind/locale {key} "
                            f"in {version_dir}: {kinds[key].name} and "
                            f"{file_path.name}",
                            agent=agent_dir.name,
                            version=version_dir.name,
                            kind=key[0],
                            locale=key[1],
                        )
                    kinds[key] = file_path
                    template_count += 1
                if kinds:
                    versions[version_dir.name] = kinds
            if versions:
                index[agent_dir.name] = versions

        self._logger.bind(
            event="prompt_registry.discovered",
            agents=len(index),
            templates=template_count,
            templates_dir=str(self._templates_dir),
        ).info("Discovered prompt templates")
        return index

    @staticmethod
    def _iter_child_dirs(directory: Path) -> list[Path]:
        """Return sorted, non-hidden child directories of ``directory``."""
        return sorted(
            child
            for child in directory.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )

    @staticmethod
    def _parse_stem(stem: str) -> _KindKey:
        """Split a filename stem into ``(kind, locale)``.

        ``"system"`` -> ``("system", None)`` and
        ``"system.fr"`` -> ``("system", "fr")``.
        """
        kind, separator, locale = stem.partition(_LOCALE_STEM_SEPARATOR)
        return (kind, locale if separator else None)

    # -- internal: resolution -------------------------------------------------

    def _resolve_and_load(self, reference: PromptReference) -> PromptTemplate:
        """Resolve a reference to concrete coordinates and load the template."""
        index = self._ensure_index()

        versions = index.get(reference.agent)
        if versions is None:
            raise PromptAgentNotFoundError(reference.agent, available=index)

        version = reference.version or self._default_version(
            reference.agent, versions
        )
        kinds = versions.get(version)
        if kinds is None:
            raise PromptVersionNotFoundError(
                reference.agent, version, available=versions
            )

        locale = self._resolve_locale(kinds, reference.kind, reference.locale)
        if locale is _NOT_FOUND:
            available = sorted({kind for (kind, _locale) in kinds})
            raise PromptTemplateNotFoundError(
                reference.agent,
                version,
                reference.kind,
                locale=reference.locale,
                available=available,
            )

        coordinate: _Coordinate = (
            reference.agent,
            version,
            reference.kind,
            locale,
        )
        return self._get_or_load(coordinate, kinds[(reference.kind, locale)])

    @staticmethod
    def _resolve_locale(
        kinds: Mapping[_KindKey, Path], kind: str, locale: str | None
    ) -> str | None | _NotFound:
        """Return the locale to use, applying base-variant fallback.

        Prefers an exact ``(kind, locale)`` match, then falls back to the base
        variant ``(kind, None)``. Returns the sentinel :data:`_NOT_FOUND` when
        neither exists.
        """
        if locale is not None and (kind, locale) in kinds:
            return locale
        if (kind, None) in kinds:
            return None
        return _NOT_FOUND

    def _default_version(
        self, agent: str, versions: Mapping[str, dict[_KindKey, Path]]
    ) -> str:
        """Resolve the default version for an agent when none is requested.

        A configured pin (``default_versions``) wins; otherwise the latest
        version is chosen by :meth:`_version_sort_key`.

        Raises:
            PromptVersionNotFoundError: If a configured pin references a version
                that does not exist.
        """
        pinned = self._default_versions.get(agent)
        if pinned is not None:
            if pinned not in versions:
                raise PromptVersionNotFoundError(
                    agent, pinned, available=versions
                )
            return pinned
        return max(versions, key=self._version_sort_key)

    @staticmethod
    def _version_sort_key(version: str) -> tuple[int, int, str]:
        """Order versions so ``vN`` sort numerically and newest is greatest.

        ``v2`` > ``v10``? No — the numeric component is compared as an integer,
        so ``v10`` > ``v9`` > ``v2``. Non-``vN`` names sort below all numbered
        versions and are ordered lexically among themselves.
        """
        match = _VERSION_PATTERN.match(version)
        if match is not None:
            return (1, int(match.group(1)), "")
        return (0, 0, version)

    # -- internal: loading and rendering -------------------------------------

    def _get_or_load(
        self, coordinate: _Coordinate, path: Path
    ) -> PromptTemplate:
        """Return the cached template for ``coordinate``, loading it if absent."""
        template = self._templates.get(coordinate)
        if template is not None:
            return template
        with self._lock:
            template = self._templates.get(coordinate)
            if template is None:
                template, compiled = self._load(coordinate, path)
                self._templates[coordinate] = template
                self._compiled[coordinate] = compiled
            return self._templates[coordinate]

    def _load(
        self, coordinate: _Coordinate, path: Path
    ) -> tuple[PromptTemplate, Template]:
        """Read, analyse, and compile a template file.

        Returns:
            The metadata-bearing :class:`PromptTemplate` and its compiled Jinja
            :class:`~jinja2.Template`.

        Raises:
            PromptRenderError: If the file cannot be read or has invalid syntax.
        """
        agent, version, kind, locale = coordinate
        # Ensure policy globals are present before analysing variables, so that
        # references to ``policies`` are not misreported as required.
        self._ensure_policies_locked()

        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromptRenderError(
                f"Failed to read prompt template {path}: {exc}",
                agent=agent,
                version=version,
                kind=kind,
                locale=locale,
            ) from exc

        try:
            ast = self._environment.parse(source, filename=str(path))
            compiled = self._environment.from_string(source)
        except TemplateSyntaxError as exc:
            raise PromptRenderError(
                f"Invalid Jinja syntax in {path}: {exc.message} "
                f"(line {exc.lineno})",
                agent=agent,
                version=version,
                kind=kind,
                locale=locale,
            ) from exc

        undeclared = meta.find_undeclared_variables(ast)
        required = frozenset(undeclared) - set(self._environment.globals)
        checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()

        template = PromptTemplate(
            agent=agent,
            version=version,
            kind=kind,
            locale=locale,
            path=path,
            source=source,
            checksum=checksum,
            required_variables=required,
        )
        self._logger.bind(
            event="prompt_registry.loaded",
            agent=agent,
            version=version,
            kind=kind,
            locale=locale,
            checksum=checksum,
            required_variables=sorted(required),
        ).debug("Loaded prompt template")
        return template, compiled

    def _render(self, template: PromptTemplate, variables: Mapping[str, Any]) -> str:
        """Render ``template`` with ``variables`` and shared policy globals.

        Raises:
            MissingPromptVariableError: If a required variable is absent.
            PromptRenderError: If rendering fails for any other reason.
        """
        coordinate: _Coordinate = (
            template.agent,
            template.version,
            template.kind,
            template.locale,
        )
        missing = template.required_variables - set(variables)
        if missing:
            raise MissingPromptVariableError(
                agent=template.agent,
                version=template.version,
                kind=template.kind,
                locale=template.locale,
                missing=missing,
                required=template.required_variables,
            )

        compiled = self._compiled.get(coordinate)
        if compiled is None:  # cache was invalidated between load and render
            _template, compiled = self._load(coordinate, template.path)
            with self._lock:
                self._templates[coordinate] = _template
                self._compiled[coordinate] = compiled

        try:
            rendered = compiled.render(dict(variables))
        except UndefinedError as exc:
            raise MissingPromptVariableError(
                agent=template.agent,
                version=template.version,
                kind=template.kind,
                locale=template.locale,
                missing={str(exc)},
                required=template.required_variables,
            ) from exc
        except TemplateError as exc:
            raise PromptRenderError(
                f"Failed to render prompt {template.reference}: {exc}",
                agent=template.agent,
                version=template.version,
                kind=template.kind,
                locale=template.locale,
            ) from exc

        self._logger.bind(
            event="prompt_registry.rendered",
            agent=template.agent,
            version=template.version,
            kind=template.kind,
            locale=template.locale,
            output_length=len(rendered),
        ).debug("Rendered prompt")
        return rendered

    # -- internal: policies ---------------------------------------------------

    def _ensure_policies(self) -> Mapping[str, str]:
        """Return shared policies, loading them once on first access."""
        if self._policies_loaded:
            return self._policies
        with self._lock:
            return self._ensure_policies_locked()

    def _ensure_policies_locked(self) -> Mapping[str, str]:
        """Load shared policies under an already-held lock, if not yet loaded."""
        if not self._policies_loaded:
            loaded = self._load_policies()
            self._policies.clear()
            self._policies.update(loaded)
            # The environment global is a live proxy over ``self._policies``;
            # updating in place keeps every template's view current.
            self._policies_loaded = True
        return self._policies

    def _load_policies(self) -> dict[str, str]:
        """Load every shared policy fragment into a name -> text mapping.

        Fragments carrying a template suffix are rendered through the Jinja
        environment, so one policy may compose another with ``{% include %}``
        — this is how a shared core fragment is reused by stricter or looser
        variants without duplicating its text. Fragments carrying a
        plain-text suffix are read verbatim.

        A policy may include other policies (resolved through the loader),
        but must not reference the ``policies`` global itself: that mapping
        is still being built here, so its contents are load-order dependent.

        Returns:
            A mapping of policy stem to its text. Missing the policies
            directory is not an error — it yields an empty mapping.

        Raises:
            PromptPolicyError: On duplicate policy names, unreadable files,
                or a fragment that fails to render.
        """
        policies: dict[str, str] = {}
        if not self._policies_dir.is_dir():
            self._logger.bind(
                event="prompt_registry.policies_absent",
                policies_dir=str(self._policies_dir),
            ).debug("No policies directory; continuing without shared policies")
            return policies

        for file_path in sorted(self._policies_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            if file_path.suffix not in self._policy_suffixes:
                continue
            name = file_path.stem
            if name in policies:
                raise PromptPolicyError(
                    f"Duplicate policy name {name!r} in {self._policies_dir}"
                )
            try:
                if file_path.suffix in self._template_suffixes:
                    relative = file_path.relative_to(self._base_dir).as_posix()
                    policies[name] = self._environment.get_template(relative).render()
                else:
                    policies[name] = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PromptPolicyError(
                    f"Failed to read policy {file_path}: {exc}"
                ) from exc
            except TemplateError as exc:
                raise PromptPolicyError(
                    f"Failed to render policy {file_path}: {exc}"
                ) from exc

        self._logger.bind(
            event="prompt_registry.policies_loaded",
            policies=len(policies),
            names=sorted(policies),
        ).debug("Loaded shared policies")
        return policies


# ---------------------------------------------------------------------------
# Sentinels and helpers
# ---------------------------------------------------------------------------


class _NotFound:
    """Sentinel type distinguishing "base locale" (``None``) from "no match"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<NOT_FOUND>"


_NOT_FOUND: Final[_NotFound] = _NotFound()


def _validate_token(token: str, reference: Any) -> None:
    """Validate a single reference segment, guarding against unsafe values.

    Raises:
        InvalidPromptReferenceError: If ``token`` is empty, a relative-path
            component (``.`` / ``..``), or contains characters outside the safe
            identifier set.
    """
    if not token:
        raise InvalidPromptReferenceError(reference, "empty path segment")
    if token in {".", ".."}:
        raise InvalidPromptReferenceError(
            reference, f"path traversal segment {token!r} is not allowed"
        )
    if _TOKEN_PATTERN.match(token) is None:
        raise InvalidPromptReferenceError(
            reference,
            f"segment {token!r} contains unsupported characters; allowed: "
            "letters, digits, '.', '_', '-'",
        )


# ---------------------------------------------------------------------------
# Process-wide instance
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """Return the process-wide :class:`PromptRegistry`.

    The instance is created lazily on first call and cached for the lifetime of
    the process, mirroring :func:`marketingos.config.loader.load_settings`.
    This is the conventional way to obtain the registry for dependency
    injection into agents. Call ``get_prompt_registry.cache_clear()`` to force
    a fresh instance (primarily useful in tests).
    """
    return PromptRegistry()
