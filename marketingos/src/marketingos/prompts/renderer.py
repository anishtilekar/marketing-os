"""Render Jinja template *text* into final prompt strings.

This module has exactly one responsibility: take template **text** plus a set
of variables and return the rendered string. It is the low-level rendering
seam that sits underneath
:class:`marketingos.prompts.registry.PromptRegistry` and is equally usable by
any future component that has already obtained template text by some other
means.

What this module deliberately does **not** do:

* load templates from disk (it accepts text, never paths);
* discover prompt directories;
* resolve prompt versions or locales;
* cache templates or compiled artefacts;
* know anything about :class:`PromptRegistry` or its on-disk layout.

Those concerns belong to the registry (discovery, version resolution,
caching), and are intentionally kept out of here so that rendering stays a
pure, context-free function of ``(template_text, variables) -> str``.

Design notes
------------
* **StrictUndefined.** The environment is configured with
  :class:`jinja2.StrictUndefined`, so a template that references a variable the
  caller did not supply raises rather than silently rendering an empty string.
  Missing variables surface as :class:`MissingVariableError`.
* **Thread safety.** A :class:`Renderer` builds its Jinja
  :class:`~jinja2.Environment` once at construction and never mutates it
  afterwards. Jinja environments — and the templates they compile — are safe
  to use from multiple threads for compilation and rendering, so a single
  shared :class:`Renderer` can back many concurrent agent executions without
  any locking of its own.
* **No filesystem loader.** The environment is created without a loader, so
  ``{% include %}`` / ``{% extends %}`` are unsupported by construction. This
  is deliberate: it makes "text only, never a file path" a structural
  guarantee rather than a convention. Shared-fragment composition (policies,
  includes) is the registry's job, not the renderer's.
* **No static variable analysis.** Computing a template's *required* variables
  is left to the registry, which already does it at load time via
  :func:`jinja2.meta.find_undeclared_variables`. The renderer relies on
  ``StrictUndefined`` as its correctness guarantee, which keeps this module
  focused purely on rendering and avoids parsing every template twice.

Usage
-----
Functional entrypoint (backed by a shared default renderer)::

    from marketingos.prompts.renderer import render

    text = render("Hello {{ name }}", {"name": "world"})

Or an explicit, independently configured instance::

    renderer = Renderer(filters={"shout": str.upper})
    text = renderer.render("{{ msg | shout }}", {"msg": "hi"})
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, Final

from jinja2 import Environment, StrictUndefined, TemplateError, Undefined
from jinja2 import TemplateSyntaxError as JinjaTemplateSyntaxError
from jinja2.exceptions import UndefinedError
from loguru import logger

from marketingos.exceptions import MarketingOSError

__all__ = [
    "MissingVariableError",
    "RenderError",
    "Renderer",
    "TemplateRenderSyntaxError",
    "get_default_renderer",
    "render",
]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
#
# TODO(prompts-exceptions): These mirror the prompt exception hierarchy that is
# currently defined *inline* in ``marketingos.prompts.registry`` (``PromptError``
# / ``PromptRenderError`` / ``MissingPromptVariableError``). The intended home
# for both sets is a shared ``marketingos/prompts/exceptions.py``; once that
# module exists, move both hierarchies there and have the registry and the
# renderer import the *same* classes, so a caller's ``except PromptRenderError``
# catches errors originating from either component. They are kept renderer-local
# for now to avoid a ``renderer -> registry`` import cycle: the registry is
# meant to depend on the renderer, never the reverse. When consolidated,
# ``RenderError`` should be reparented under the shared ``PromptError`` base.


class RenderError(MarketingOSError):
    """Base class for every error raised while rendering a template.

    Catch this to handle any rendering failure uniformly, regardless of whether
    the underlying cause was invalid template syntax or a missing variable.
    """


class TemplateRenderSyntaxError(RenderError):
    """Raised when template text cannot be compiled because its Jinja syntax is
    invalid (an unclosed block, a malformed expression, and so on).

    This wraps :class:`jinja2.TemplateSyntaxError` so that callers never have to
    depend on Jinja's exception types directly. The original exception is always
    available via ``__cause__``.
    """


class MissingVariableError(RenderError):
    """Raised when a template references a variable that was not supplied.

    Because the environment uses :class:`jinja2.StrictUndefined`, referencing an
    absent variable fails at render time instead of rendering an empty string.
    The message identifies the offending variable as reported by Jinja; the
    original :class:`jinja2.UndefinedError` is available via ``__cause__``.
    """


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Render Jinja template text into prompt strings.

    A ``Renderer`` owns a single Jinja :class:`~jinja2.Environment`, configured
    once at construction and never mutated afterwards. That immutability is what
    makes an instance safe to share across threads and across concurrent agent
    executions without locking.

    A class (rather than a bare function) is used deliberately: the environment
    is the natural extension point for the features this system is likely to
    grow — custom ``filters``, injected ``globals``, and template macros. Those
    are configured up front and captured by the instance, keeping the per-render
    call path a simple ``render(text, variables) -> str``.

    The renderer accepts template **text only**. It never reads files, resolves
    versions, or caches templates; see the module docstring for the full list of
    non-responsibilities.

    Args:
        filters: Optional custom Jinja filters to register, keyed by the name
            used inside templates (e.g. ``{"shout": str.upper}``).
        globals: Optional variables exposed to every rendered template without
            being passed per call (e.g. a project name or a set of macros).
            Per-call variables take precedence over globals of the same name.
        strict_undefined: When ``True`` (default), an undefined variable raises
            :class:`MissingVariableError`. When ``False``, undefined variables
            render as empty strings — provided only as an escape hatch; the
            system's guarantee ("missing variables always raise") depends on the
            default.
        trim_blocks: Jinja ``trim_blocks`` setting. Defaults to ``True`` to match
            the registry's environment so prompts render identically whether they
            flow through the registry or the renderer directly.
        lstrip_blocks: Jinja ``lstrip_blocks`` setting; defaults to ``True`` for
            the same parity reason.
        keep_trailing_newline: Jinja ``keep_trailing_newline`` setting; defaults
            to ``True`` so a template's trailing newline is preserved.
    """

    def __init__(
        self,
        *,
        filters: Mapping[str, Callable[..., Any]] | None = None,
        globals: Mapping[str, Any] | None = None,
        strict_undefined: bool = True,
        trim_blocks: bool = True,
        lstrip_blocks: bool = True,
        keep_trailing_newline: bool = True,
    ) -> None:
        self._environment: Final[Environment] = Environment(
            # No loader: this renderer is text-only and never touches the
            # filesystem, which also disables include/extends by construction.
            autoescape=False,  # prompts are plain text; HTML escaping corrupts them
            undefined=StrictUndefined if strict_undefined else Undefined,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
            keep_trailing_newline=keep_trailing_newline,
        )
        # Configure extension points exactly once, before the environment is
        # shared; the instance is treated as immutable from here on.
        if globals:
            self._environment.globals.update(dict(globals))
        if filters:
            self._environment.filters.update(dict(filters))

        self._logger = logger.bind(component="Renderer")

    def render(self, template: str, variables: Mapping[str, Any]) -> str:
        """Render ``template`` text with ``variables`` and return the result.

        Args:
            template: The raw Jinja template *text*. Must be a ``str``; file
                paths are never accepted.
            variables: The substitution variables for this render. A read-only
                mapping is sufficient; it is not mutated.

        Returns:
            The rendered prompt string.

        Raises:
            RenderError: If ``template`` is not a string, or if rendering fails
                for a reason other than syntax or an undefined variable.
            TemplateRenderSyntaxError: If ``template`` cannot be compiled because
                its Jinja syntax is invalid.
            MissingVariableError: If the template references a variable that is
                not present in ``variables`` (or in the renderer's globals).
        """
        if not isinstance(template, str):
            raise RenderError(
                f"template must be a string, got {type(template).__name__!r}"
            )

        try:
            compiled = self._environment.from_string(template)
        except JinjaTemplateSyntaxError as exc:
            raise TemplateRenderSyntaxError(
                f"Invalid Jinja template syntax: {exc.message} (line {exc.lineno})"
            ) from exc

        try:
            rendered = compiled.render(dict(variables))
        except UndefinedError as exc:
            # StrictUndefined turns any reference to an unsupplied variable into
            # an UndefinedError at render time; surface it as a domain error.
            raise MissingVariableError(str(exc)) from exc
        except TemplateError as exc:
            # Any other Jinja-level failure (a filter raising, a bad expression
            # at runtime, ...). UndefinedError is handled above; it is a subclass
            # of TemplateError, so ordering matters here.
            raise RenderError(f"Failed to render template: {exc}") from exc

        self._logger.bind(
            event="renderer.rendered",
            template_length=len(template),
            output_length=len(rendered),
        ).debug("Rendered template")
        return rendered


# ---------------------------------------------------------------------------
# Process-wide convenience
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_default_renderer() -> Renderer:
    """Return the process-wide default :class:`Renderer`.

    The instance is created lazily on first call and reused thereafter, mirroring
    :func:`marketingos.prompts.registry.get_prompt_registry` and
    :func:`marketingos.config.loader.load_settings`. It carries no custom filters
    or globals; construct a :class:`Renderer` directly when those are needed.

    Call ``get_default_renderer.cache_clear()`` to force a fresh instance
    (primarily useful in tests).
    """
    return Renderer()


def render(template: str, variables: Mapping[str, Any]) -> str:
    """Render ``template`` text with ``variables`` using the default renderer.

    This is the small functional entrypoint most callers want. It delegates to
    the shared :func:`get_default_renderer` instance; see :meth:`Renderer.render`
    for the full contract and the exceptions it may raise.

    Args:
        template: The raw Jinja template text (never a file path).
        variables: The substitution variables for this render.

    Returns:
        The rendered prompt string.
    """
    return get_default_renderer().render(template, variables)