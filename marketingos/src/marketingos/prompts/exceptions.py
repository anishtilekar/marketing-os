"""Exception hierarchy for the ``prompts`` package.

This module has a single responsibility: define the custom exceptions raised by
prompt components (loader, renderer, registry, and any future collaborators). It
contains **no** prompt loading, rendering, version resolution, caching, or other
business logic, and it depends on none of those components — only on the
application-wide base exception. That keeps it a true leaf module every prompt
component can import without risking a cycle.

Hierarchy
---------
Everything is rooted in :class:`PromptError`, which in turn extends the
application-wide :class:`~marketingos.exceptions.MarketingOSError`. Callers can
therefore catch at whatever granularity they need::

    MarketingOSError                     # anything wrong, anywhere in the app
    └── PromptError                      # anything wrong in the prompts package
        ├── PromptNotFoundError          # a requested prompt does not exist
        │   ├── PromptVersionNotFoundError
        │   └── PromptTemplateNotFoundError
        ├── PromptMetadataError          # metadata is missing or malformed
        ├── PromptRenderError            # a template failed to compile/render
        ├── PromptValidationError        # a prompt or its inputs failed validation
        ├── PromptDirectoryError         # a prompt directory is missing/unreadable
        ├── PromptCacheError             # a cache operation failed
        └── PromptConfigurationError     # the prompt subsystem is misconfigured

A component can ``except PromptError`` to treat any prompt failure uniformly,
or catch a specific subclass (or an intermediate base such as
:class:`PromptNotFoundError`) when it needs to react differently.

Design conventions
------------------
* **Meaningful defaults.** Every exception composes an informative message from
  whatever context it is given, and falls back to a sensible class-level default
  when raised with no arguments.
* **Optional context.** Constructors accept optional, keyword-friendly context
  (``name``, ``version``, ``kind``, ``path``, ``detail``, ...). Everything
  supplied is stored as a plain attribute for programmatic inspection, not just
  folded into the message string.
* **Exception chaining.** Constructors accept an optional ``cause``. When
  supplied it is recorded as the exception's ``__cause__``, so the triggering
  error is preserved even if a raise site forgets an explicit ``from``. Raise
  sites are still encouraged to write ``raise PromptRenderError(...) from exc``;
  that produces the identical chain and reads clearly at the call site.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from marketingos.exceptions import MarketingOSError

__all__ = [
    "PromptCacheError",
    "PromptConfigurationError",
    "PromptDirectoryError",
    "PromptError",
    "PromptMetadataError",
    "PromptNotFoundError",
    "PromptRenderError",
    "PromptTemplateNotFoundError",
    "PromptValidationError",
    "PromptVersionNotFoundError",
]


def _join(values: Iterable[str] | None) -> str:
    """Format an iterable of names for inclusion in an error message.

    Args:
        values: Names to join, or ``None``.

    Returns:
        A comma-separated string, or the literal ``"<none>"`` when there is
        nothing to list.
    """
    items = [str(value) for value in values] if values is not None else []
    return ", ".join(items) if items else "<none>"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class PromptError(MarketingOSError):
    """Base class for every error raised by the prompts package.

    Catch this to handle any prompt-related failure uniformly. It extends the
    application-wide :class:`~marketingos.exceptions.MarketingOSError`, so code
    that catches the latter also catches prompt errors.

    Args:
        message: The final error message. When omitted, the class-level
            :attr:`_default_message` is used.
        cause: The lower-level exception that triggered this one, if any. When
            supplied it is recorded as ``__cause__`` so the original error is
            preserved even without an explicit ``raise ... from`` clause.

    Attributes:
        message: The resolved, human-readable message.
        cause: The originating exception, or ``None``.
    """

    #: Fallback message used when neither a message nor context is supplied.
    _default_message: ClassVar[str] = "A prompt operation failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        resolved = message if message is not None else self._default_message
        super().__init__(resolved)
        self.message: str = resolved
        self.cause: BaseException | None = cause
        if cause is not None:
            # Preserve the triggering exception even when the raise site did not
            # use an explicit ``from``. An explicit ``raise ... from cause`` has
            # the identical effect and overrides this harmlessly.
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# Lookup failures
# ---------------------------------------------------------------------------


class PromptNotFoundError(PromptError):
    """Raised when a requested prompt cannot be located.

    This is the common base for the more specific lookup failures
    (:class:`PromptVersionNotFoundError` and
    :class:`PromptTemplateNotFoundError`). Catch :class:`PromptNotFoundError` to
    handle "what you asked for does not exist" uniformly, or a subclass to
    distinguish *what* was missing.

    Args:
        name: The prompt/agent name that could not be found.
        available: The names that *do* exist, for a helpful hint.
        message: An explicit message overriding the composed default.
        cause: The originating exception, if any.

    Attributes:
        name: The name that was looked up, or ``None``.
        available: The known alternatives, or ``None`` if not provided.
    """

    _default_message: ClassVar[str] = "The requested prompt could not be found."

    def __init__(
        self,
        name: str | None = None,
        *,
        available: Iterable[str] | None = None,
        message: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        available = (
            tuple(str(item) for item in available) if available is not None else None
        )
        self.name = name
        self.available: tuple[str, ...] | None = available
        super().__init__(message or self._compose(), cause=cause)

    def _compose(self) -> str:
        if not self.name:
            return self._default_message
        text = f"No prompt registered for {self.name!r}."
        if self.available:
            text += f" Known prompts: {_join(self.available)}."
        return text


class PromptVersionNotFoundError(PromptNotFoundError):
    """Raised when an agent exists but the requested version does not.

    Args:
        agent: The owning agent/prompt name.
        version: The version that could not be found.
        available: The versions that do exist for ``agent``.
        message: An explicit message overriding the composed default.
        cause: The originating exception, if any.

    Attributes:
        agent: The owning agent, or ``None``.
        version: The missing version, or ``None``.
        available: The known versions, or ``None``.
    """

    _default_message: ClassVar[str] = (
        "The requested prompt version could not be found."
    )

    def __init__(
        self,
        agent: str | None = None,
        version: str | None = None,
        *,
        available: Iterable[str] | None = None,
        message: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        available = (
            tuple(str(item) for item in available) if available is not None else None
        )
        self.agent = agent
        self.version = version
        super().__init__(
            name=agent,
            available=available,
            message=message or self._build(agent, version, available),
            cause=cause,
        )

    @staticmethod
    def _build(
        agent: str | None,
        version: str | None,
        available: tuple[str, ...] | None,
    ) -> str:
        if agent is None and version is None:
            return PromptVersionNotFoundError._default_message
        subject = f"version {version!r}" if version else "the requested version"
        owner = f" for prompt {agent!r}" if agent else ""
        text = f"No {subject}{owner}."
        if available:
            text += f" Known versions: {_join(available)}."
        return text


class PromptTemplateNotFoundError(PromptNotFoundError):
    """Raised when no template matches the requested coordinates.

    Args:
        agent: The owning agent.
        version: The resolved version searched.
        kind: The template kind (file stem), e.g. ``"system"`` or ``"user"``.
        locale: The requested locale, or ``None`` for the base variant.
        available: The kinds that do exist for this agent/version.
        message: An explicit message overriding the composed default.
        cause: The originating exception, if any.

    Attributes:
        agent: The owning agent, or ``None``.
        version: The version searched, or ``None``.
        kind: The requested kind, or ``None``.
        locale: The requested locale, or ``None``.
        available: The known kinds, or ``None``.
    """

    _default_message: ClassVar[str] = (
        "The requested prompt template could not be found."
    )

    def __init__(
        self,
        agent: str | None = None,
        version: str | None = None,
        kind: str | None = None,
        *,
        locale: str | None = None,
        available: Iterable[str] | None = None,
        message: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        available = (
            tuple(str(item) for item in available) if available is not None else None
        )
        self.agent = agent
        self.version = version
        self.kind = kind
        self.locale = locale
        super().__init__(
            name=agent,
            available=available,
            message=message or self._build(agent, version, kind, locale, available),
            cause=cause,
        )

    @staticmethod
    def _build(
        agent: str | None,
        version: str | None,
        kind: str | None,
        locale: str | None,
        available: tuple[str, ...] | None,
    ) -> str:
        coordinates = []
        if agent:
            coordinates.append(f"agent={agent!r}")
        if version:
            coordinates.append(f"version={version!r}")
        if kind:
            coordinates.append(f"kind={kind!r}")
        if locale:
            coordinates.append(f"locale={locale!r}")
        where = ", ".join(coordinates) if coordinates else "the requested coordinates"
        text = f"No prompt template for {where}."
        if available:
            text += f" Known kinds: {_join(available)}."
        return text


# ---------------------------------------------------------------------------
# Metadata / rendering / validation
# ---------------------------------------------------------------------------


class PromptMetadataError(PromptError):
    """Raised when a prompt's metadata is missing or malformed.

    Covers problems such as an unparsable version directory name, a bad or
    missing checksum, or invalid front-matter attached to a template.

    Args:
        message: An explicit message overriding the composed default.
        name: The prompt/reference the metadata belongs to.
        path: The file whose metadata is at fault.
        detail: A short human-readable explanation of what is wrong.
        cause: The originating exception, if any.

    Attributes:
        name: The associated prompt name, or ``None``.
        path: The associated file path, or ``None``.
        detail: The failure detail, or ``None``.
    """

    _default_message: ClassVar[str] = "Prompt metadata could not be processed."

    def __init__(
        self,
        message: str | None = None,
        *,
        name: str | None = None,
        path: object | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.detail = detail
        super().__init__(message or self._build(name, path, detail, cause), cause=cause)

    @staticmethod
    def _build(
        name: str | None,
        path: object | None,
        detail: str | None,
        cause: BaseException | None,
    ) -> str:
        subject = name or (str(path) if path is not None else None)
        text = (
            f"Invalid prompt metadata for {subject!r}."
            if subject
            else "Invalid prompt metadata."
        )
        reason = detail or (str(cause) if cause is not None else None)
        if reason:
            text += f" {reason}"
        return text


class PromptRenderError(PromptError):
    """Raised when a template fails to compile or render.

    Wrap the underlying templating error (for example a Jinja
    ``TemplateSyntaxError`` or a runtime failure) as ``cause`` so the original
    is preserved.

    Args:
        message: An explicit message overriding the composed default.
        reference: The prompt reference/name being rendered.
        detail: A short human-readable explanation (e.g. the syntax error text).
        cause: The originating templating exception, if any.

    Attributes:
        reference: The prompt being rendered, or ``None``.
        detail: The failure detail, or ``None``.
    """

    _default_message: ClassVar[str] = "The prompt template could not be rendered."

    def __init__(
        self,
        message: str | None = None,
        *,
        reference: str | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.reference = reference
        self.detail = detail
        super().__init__(
            message or self._build(reference, detail, cause), cause=cause
        )

    @staticmethod
    def _build(
        reference: str | None,
        detail: str | None,
        cause: BaseException | None,
    ) -> str:
        text = (
            f"Failed to render prompt {reference!r}."
            if reference
            else "Failed to render prompt template."
        )
        reason = detail or (str(cause) if cause is not None else None)
        if reason:
            text += f" {reason}"
        return text


class PromptValidationError(PromptError):
    """Raised when a prompt or the data supplied to it fails validation.

    Covers cases such as required template variables not being supplied, a
    rendered prompt failing a structural check, or a malformed prompt reference.

    Args:
        message: An explicit message overriding the composed default.
        name: The prompt/reference being validated.
        errors: Individual validation problems to enumerate in the message.
        detail: A short human-readable summary.
        cause: The originating exception, if any.

    Attributes:
        name: The associated prompt name, or ``None``.
        errors: A tuple of individual validation messages (possibly empty).
        detail: The summary detail, or ``None``.
    """

    _default_message: ClassVar[str] = "Prompt validation failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        name: str | None = None,
        errors: Iterable[str] | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.name = name
        self.errors: tuple[str, ...] = (
            tuple(str(error) for error in errors) if errors is not None else ()
        )
        self.detail = detail
        super().__init__(
            message or self._build(name, self.errors, detail), cause=cause
        )

    @staticmethod
    def _build(
        name: str | None,
        errors: tuple[str, ...],
        detail: str | None,
    ) -> str:
        subject = f" for prompt {name!r}" if name else ""
        text = f"Validation failed{subject}."
        if detail:
            text += f" {detail}"
        if errors:
            text += f" Issues: {_join(errors)}."
        return text


# ---------------------------------------------------------------------------
# Environment / infrastructure
# ---------------------------------------------------------------------------


class PromptDirectoryError(PromptError):
    """Raised when a prompt directory is missing or cannot be read.

    Args:
        message: An explicit message overriding the composed default.
        path: The directory at fault.
        detail: A short human-readable explanation.
        cause: The originating exception (e.g. an ``OSError``), if any.

    Attributes:
        path: The associated directory, or ``None``.
        detail: The failure detail, or ``None``.
    """

    _default_message: ClassVar[str] = "A prompt directory could not be accessed."

    def __init__(
        self,
        message: str | None = None,
        *,
        path: object | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.path = path
        self.detail = detail
        super().__init__(message or self._build(path, detail, cause), cause=cause)

    @staticmethod
    def _build(
        path: object | None,
        detail: str | None,
        cause: BaseException | None,
    ) -> str:
        text = (
            f"Prompt directory error at {str(path)!r}."
            if path is not None
            else "Prompt directory error."
        )
        reason = detail or (str(cause) if cause is not None else None)
        if reason:
            text += f" {reason}"
        return text


class PromptCacheError(PromptError):
    """Raised when a prompt cache operation fails.

    Args:
        message: An explicit message overriding the composed default.
        operation: The cache operation being performed (e.g. ``"get"``,
            ``"set"``, ``"invalidate"``).
        key: The cache key involved, if applicable.
        detail: A short human-readable explanation.
        cause: The originating exception, if any.

    Attributes:
        operation: The attempted operation, or ``None``.
        key: The cache key involved, or ``None``.
        detail: The failure detail, or ``None``.
    """

    _default_message: ClassVar[str] = "A prompt cache operation failed."

    def __init__(
        self,
        message: str | None = None,
        *,
        operation: str | None = None,
        key: object | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.operation = operation
        self.key = key
        self.detail = detail
        super().__init__(
            message or self._build(operation, key, detail, cause), cause=cause
        )

    @staticmethod
    def _build(
        operation: str | None,
        key: object | None,
        detail: str | None,
        cause: BaseException | None,
    ) -> str:
        text = f"Prompt cache {operation or 'operation'} failed"
        if key is not None:
            text += f" for key {key!r}"
        text += "."
        reason = detail or (str(cause) if cause is not None else None)
        if reason:
            text += f" {reason}"
        return text


class PromptConfigurationError(PromptError):
    """Raised when the prompt subsystem itself is misconfigured.

    Covers cases such as a default-version pin that points at a version that
    does not exist, an invalid suffix configuration, or a malformed reference
    grammar setting — problems that stem from configuration rather than from a
    specific missing prompt.

    Args:
        message: An explicit message overriding the composed default.
        setting: The name of the offending configuration setting.
        value: The offending value, if useful to report.
        detail: A short human-readable explanation.
        cause: The originating exception, if any.

    Attributes:
        setting: The associated setting name, or ``None``.
        value: The associated value, or ``None``.
        detail: The failure detail, or ``None``.
    """

    _default_message: ClassVar[str] = "The prompt subsystem is misconfigured."

    def __init__(
        self,
        message: str | None = None,
        *,
        setting: str | None = None,
        value: object | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.setting = setting
        self.value = value
        self.detail = detail
        super().__init__(
            message or self._build(setting, value, detail), cause=cause
        )

    @staticmethod
    def _build(
        setting: str | None,
        value: object | None,
        detail: str | None,
    ) -> str:
        if setting is not None:
            text = f"Invalid prompt configuration for {setting!r}"
            if value is not None:
                text += f" (got {value!r})"
            text += "."
        else:
            text = "Invalid prompt configuration."
        if detail:
            text += f" {detail}"
        return text