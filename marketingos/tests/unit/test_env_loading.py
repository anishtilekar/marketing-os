"""Tests for the API's ``.env`` loading at import time.

The bug this guards against: the app used to read API keys straight from
``os.environ`` with no ``.env`` loader, so the ``.env`` file users edited had
no effect and a stale/expired key in the shell environment silently persisted.
``_load_project_env`` now loads ``<root>/.env`` deterministically. No network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from marketingos.api.dependencies import _load_project_env

_LOADED_VAR = "MARKETINGOS_TEST_DOTENV_VAR"
_PRESET_VAR = "MARKETINGOS_TEST_PRESET_VAR"


@pytest.fixture(autouse=True)
def _clean_test_env() -> None:
    """Ensure the probe vars never leak between tests or into the session."""
    for name in (_LOADED_VAR, _PRESET_VAR):
        os.environ.pop(name, None)
    yield
    for name in (_LOADED_VAR, _PRESET_VAR):
        os.environ.pop(name, None)


def test_loads_values_from_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{_LOADED_VAR}=from_dotenv\n", encoding="utf-8")

    loaded = _load_project_env(tmp_path)

    assert loaded is True
    assert os.environ[_LOADED_VAR] == "from_dotenv"


def test_does_not_override_existing_environment(tmp_path: Path) -> None:
    # A value already in the real environment (e.g. an explicit shell export)
    # must win over the file — that precedence is what lets a shell key override
    # a stale one committed to .env.
    os.environ[_PRESET_VAR] = "from_shell"
    (tmp_path / ".env").write_text(f"{_PRESET_VAR}=from_file\n", encoding="utf-8")

    _load_project_env(tmp_path)

    assert os.environ[_PRESET_VAR] == "from_shell"


def test_missing_env_file_is_a_noop(tmp_path: Path) -> None:
    # No .env in the directory: loading must not raise and must report nothing
    # loaded, so a fresh checkout without a .env still starts.
    assert _load_project_env(tmp_path) is False
