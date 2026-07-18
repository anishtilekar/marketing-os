"""Shared public-only compliance guard for the web/Instagram tools.

Per the architecture doc's Tool Abstraction Layer section: "Web/Instagram
tools are structurally restricted to public, unauthenticated read paths,
with a compliance wrapper that refuses outright if a login-wall or
private-data path is detected... implemented as decorators, not
duplicated inside each tool implementation." This module is that one
implementation, shared by :mod:`~marketingos.tools.web.website_scraper`
and :mod:`~marketingos.tools.web.instagram_public_reader`.

This is a hard refusal, not a configurable policy: there is no bypass
argument, because the point is that neither tool — nor a prompt that
happens to ask it to try harder — can talk its way past a login wall.
"""

from __future__ import annotations

import re
from typing import Final

from marketingos.exceptions.tool import ToolExecutionError

__all__ = ["refuse_if_login_walled"]

#: Case-insensitive substrings strongly associated with an authentication
#: gate rather than public page content. Intentionally coarse: a false
#: positive (refusing a public page that happens to mention "log in" in
#: passing) is a far smaller problem than a false negative (silently
#: reading past a login wall).
_LOGIN_WALL_MARKERS: Final[tuple[str, ...]] = (
    "log in to continue",
    "sign in to continue",
    "you must be logged in",
    "login required",
    "sign in to view this",
    "this account is private",
    "content isn't available right now",
    "page not found",
)

_MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(marker) for marker in _LOGIN_WALL_MARKERS), re.IGNORECASE
)


def refuse_if_login_walled(text: str, *, source: str) -> None:
    """Raise if ``text`` reads like an auth gate rather than public content.

    Args:
        text: Page text (HTML or extracted text) to inspect.
        source: The URL or identifier being fetched, used in the error.

    Raises:
        ToolExecutionError: If a login-wall marker is found.
    """
    if _MARKER_PATTERN.search(text):
        raise ToolExecutionError(
            f"Refusing to read {source!r}: content behind an apparent "
            "login wall, private-account gate, or removed page. This tool "
            "is restricted to public, unauthenticated read paths and does "
            "not bypass this — see marketingos.tools.web._compliance."
        )
