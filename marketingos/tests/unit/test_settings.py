from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from marketingos.config import loader
from marketingos.config.loader import (
    ConfigFileNotFoundError,
    ConfigValidationError,
    load_settings,
)
from marketingos.config.settings import BudgetSettings
from marketingos.models.cost import CostLedger

VALID_BASE_YAML = """
project_name: MarketingOS
version: "0.1.0"
"""

VALID_BUDGET_YAML = """
max_budget: 100
warning_threshold: 80
currency: INR
"""

VALID_WORKFLOW_YAML = """
max_revisions: 2
enable_human_review: true
checkpoint_after_each_agent: true
"""

VALID_MODELS_YAML = """
default_llm: gpt-4.1-mini
fallback_llm: llama3
temperature: 0.3
max_tokens: 4000
image_quality: standard
"""

VALID_AGENTS_YAML = """
research:
  model: gpt-4.1-mini
  temperature: 0.2
  enabled: true

synthetic_source:
  model: gpt-4.1-mini
  temperature: 0.3
  enabled: true

business_analysis:
  model: gpt-4.1
  temperature: 0.2
  enabled: true

strategist:
  model: gpt-4.1
  temperature: 0.35
  enabled: true

planner:
  model: gpt-4.1
  temperature: 0.4
  enabled: true

copywriter:
  model: gpt-4.1
  temperature: 0.6
  enabled: true

designer:
  model: gpt-4.1-mini
  temperature: 0.5
  enabled: true

video_director:
  model: gpt-4.1-mini
  temperature: 0.5
  enabled: true

qa:
  model: gpt-4.1-mini
  temperature: 0.0
  enabled: true

packaging:
  model: gpt-4.1-mini
  temperature: 0.1
  enabled: true
"""

_DEFAULT_FILES = {
    "base.yaml": VALID_BASE_YAML,
    "budget.yaml": VALID_BUDGET_YAML,
    "workflow.yaml": VALID_WORKFLOW_YAML,
    "models.yaml": VALID_MODELS_YAML,
    "agents.yaml": VALID_AGENTS_YAML,
}


def _write_config_dir(
    directory: Path,
    overrides: dict[str, str] | None = None,
    omit: set[str] | None = None,
) -> Path:
    """Write a complete (or deliberately broken) set of config files."""
    files = dict(_DEFAULT_FILES)
    if overrides:
        files.update(overrides)
    if omit:
        for filename in omit:
            files.pop(filename, None)

    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure lru_cache does not leak state between tests."""
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = _write_config_dir(tmp_path / "configs")
    monkeypatch.setattr(loader, "_config_directory", lambda: directory)
    return directory


def test_config_directory_resolves_to_the_real_config_dir() -> None:
    """The loader must find the repository's actual config directory.

    Every other test here monkeypatches ``_config_directory``, which is
    exactly why the loader spent its life pointing at a ``configs``
    directory that has never existed. This one exercises the real lookup.
    """
    directory = loader._config_directory()  # noqa: SLF001 - the point of the test

    assert directory.is_dir()
    assert directory.name == "config"
    assert (directory / "budget.yaml").is_file()


def test_shipped_budget_yaml_is_populated_and_decimal() -> None:
    """config/budget.yaml carries the real ceiling, parsed as Decimal."""
    directory = loader._config_directory()  # noqa: SLF001 - real file on purpose
    budget = BudgetSettings.model_validate(
        loader._read_yaml(directory / "budget.yaml")  # noqa: SLF001
    )

    assert budget.max_budget == Decimal("100")
    assert isinstance(budget.max_budget, Decimal)
    assert budget.currency == "INR"


def test_budget_floats_are_converted_without_binary_error() -> None:
    """YAML floats become exact Decimals, not their binary approximations.

    ``Decimal(0.1)`` is ``0.1000000000000000055...``; the settings-boundary
    validator must route through ``str`` so money stays exact.
    """
    budget = BudgetSettings(
        max_budget=0.1, warning_threshold=0.2, currency="INR"
    )

    assert budget.max_budget == Decimal("0.1")
    assert budget.warning_threshold == Decimal("0.2")


def test_budget_ceiling_feeds_the_cost_ledger_directly() -> None:
    """The configured ceiling reaches CostLedger with no float hop."""
    budget = BudgetSettings(max_budget=100, warning_threshold=80, currency="INR")
    ledger = CostLedger(max_budget=budget.max_budget)

    assert ledger.max_budget == Decimal("100")


def test_valid_yaml_loads(config_dir: Path) -> None:
    settings = load_settings()

    assert settings.budget.max_budget == 100
    assert settings.workflow.max_revisions == 2
    assert settings.models.default_llm == "gpt-4.1-mini"
    assert settings.agents.research.model == "gpt-4.1-mini"
    assert settings.agents.planner.model == "gpt-4.1"


def test_missing_yaml_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _write_config_dir(tmp_path / "configs", omit={"budget.yaml"})
    monkeypatch.setattr(loader, "_config_directory", lambda: directory)

    with pytest.raises(ConfigFileNotFoundError) as exc_info:
        load_settings()

    assert "budget.yaml" in str(exc_info.value)


def test_invalid_value_raises_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_budget_yaml = """
max_budget: not_a_number
warning_threshold: 80
currency: INR
"""
    directory = _write_config_dir(
        tmp_path / "configs",
        overrides={"budget.yaml": invalid_budget_yaml},
    )
    monkeypatch.setattr(loader, "_config_directory", lambda: directory)

    with pytest.raises(ConfigValidationError) as exc_info:
        load_settings()

    assert "budget.yaml" in str(exc_info.value)


@pytest.mark.xfail(
    reason=(
        "Environment variable overrides are out of scope for Stage 1.2 "
        "(explicitly excluded) and are not yet wired through loader.py, "
        "since Settings() is currently constructed with explicit init "
        "kwargs, which take priority over env vars in pydantic-settings. "
        "This test documents the expected contract for a later stage."
    ),
    strict=False,
)
def test_environment_override_works(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "BUDGET",
        '{"max_budget": 500, "warning_threshold": 450, "currency": "USD"}',
    )
    load_settings.cache_clear()

    settings = load_settings()

    assert settings.budget.max_budget == 500
    assert settings.budget.currency == "USD"
