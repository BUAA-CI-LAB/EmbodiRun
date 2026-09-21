"""Deployment metadata configuration and parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .validation import RESOURCE_ID, ConfigError, reject_unknown, string

_GIT_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    name: str
    deploy_commit: str


def parse_metadata(value: dict[str, Any]) -> MetadataConfig:
    reject_unknown(
        value,
        {"name", "deploy-commit"},
        "metadata",
    )
    name = string(value, "name", "metadata")
    if not RESOURCE_ID.fullmatch(name):
        raise ConfigError(
            "metadata.name must start with an alphanumeric character and contain "
            "only alphanumerics, dots, underscores, or hyphens"
        )
    return MetadataConfig(
        name=name,
        deploy_commit=_revision(value, "deploy-commit"),
    )


def _revision(value: dict[str, Any], key: str) -> str:
    revision = string(value, key, "metadata")
    if not _GIT_REVISION.fullmatch(revision) or ".." in revision or "@{" in revision:
        raise ConfigError(f"metadata.{key} is not a safe Git revision")
    return revision


__all__ = ["MetadataConfig"]
