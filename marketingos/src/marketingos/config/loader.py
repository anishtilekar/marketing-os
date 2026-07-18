from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from marketingos.config.settings import (
    AgentSettings,
    BudgetSettings,
    ModelSettings,
    Settings,
    WorkflowSettings,
)

# TypeVar for preserving concrete type information through _build_section()
T = TypeVar('T', bound=BaseModel)


class ConfigError(Exception):
    """Base exception for configuration loading failures."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when a required configuration file is missing."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Configuration file not found: {path}")
        self.path = path


class ConfigParseError(ConfigError):
    """Raised when a configuration file contains invalid YAML."""

    def __init__(self, path: Path, original_error: Exception) -> None:
        super().__init__(f"Failed to parse YAML in configuration file: {path} ({original_error})")
        self.path = path
        self.original_error = original_error


class ConfigValidationError(ConfigError):
    """Raised when a configuration file fails schema validation."""

    def __init__(self, path: Path, original_error: Exception) -> None:
        super().__init__(f"Validation failed for configuration file: {path}\n{original_error}")
        self.path = path
        self.original_error = original_error


def _project_root() -> Path:
    """Locate the project root directory.

    The project root is determined as the directory containing the
    ``config`` folder, resolved relative to this file's location
    (``src/marketingos/config/loader.py``).
    """
    current_file = Path(__file__).resolve()
    return current_file.parents[3]


def _config_directory() -> Path:
    """Locate the ``config`` directory at the project root."""
    config_dir = _project_root() / "config"
    if not config_dir.is_dir():
        raise ConfigFileNotFoundError(config_dir)
    return config_dir


def _read_yaml(path: Path) -> dict[str, Any]:
    """Safely read and parse a single YAML configuration file.

    Args:
        path: Path to the YAML file to read.

    Returns:
        The parsed YAML content as a dictionary. An empty file yields
        an empty dictionary.

    Raises:
        ConfigFileNotFoundError: If the file does not exist.
        ConfigParseError: If the file contains invalid YAML.
    """
    if not path.is_file():
        raise ConfigFileNotFoundError(path)

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            content = yaml.safe_load(file_handle)
    except yaml.YAMLError as error:
        raise ConfigParseError(path, error) from error

    return content or {}


def _build_section(model: type[T], data: dict[str, Any], path: Path) -> T:
    """Validate a configuration section against its Pydantic model.

    Args:
        model: The Pydantic model class to validate against.
        data: The raw configuration data for this section.
        path: The source file path, used for error reporting.

    Returns:
        A validated instance of ``model``.

    Raises:
        ConfigValidationError: If validation fails.
    """
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigValidationError(path, error) from error


@lru_cache
def load_settings() -> Settings:
    """Load, validate, and cache the full MarketingOS configuration.

    Reads all YAML files from the ``config`` directory, validates each
    section against its corresponding Pydantic model, and assembles the
    root ``Settings`` object. The result is cached so subsequent calls
    return the same instance without re-reading files.

    Returns:
        A fully validated ``Settings`` instance.

    Raises:
        ConfigFileNotFoundError: If a required configuration file is missing.
        ConfigParseError: If a configuration file contains invalid YAML.
        ConfigValidationError: If a configuration file fails validation.
    """
    config_dir = _config_directory()

    base_path = config_dir / "base.yaml"
    budget_path = config_dir / "budget.yaml"
    workflow_path = config_dir / "workflow.yaml"
    models_path = config_dir / "models.yaml"
    agents_path = config_dir / "agents.yaml"

    _read_yaml(base_path)

    budget_data = _read_yaml(budget_path)
    workflow_data = _read_yaml(workflow_path)
    models_data = _read_yaml(models_path)
    agents_data = _read_yaml(agents_path)

    budget = _build_section(BudgetSettings, budget_data, budget_path)
    workflow = _build_section(WorkflowSettings, workflow_data, workflow_path)
    models = _build_section(ModelSettings, models_data, models_path)
    agents = _build_section(AgentSettings, agents_data, agents_path)

    try:
        return Settings(
            budget=budget,
            workflow=workflow,
            models=models,
            agents=agents,
        )
    except ValidationError as error:
        raise ConfigValidationError(config_dir, error) from error