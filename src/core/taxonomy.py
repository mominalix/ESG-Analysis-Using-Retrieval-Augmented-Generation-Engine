"""Configurable ESG taxonomy shared by the API and document pipeline."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from .config import settings
from .exceptions import ConfigurationError


class FrameworkDefinition(BaseModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    region: str = Field(min_length=1)
    type: str = Field(min_length=1)


class ESGTaxonomy(BaseModel):
    frameworks: list[FrameworkDefinition]
    categories: dict[str, list[str]]
    document_types: list[str]

    @property
    def framework_ids(self) -> list[str]:
        return [framework.id for framework in self.frameworks]

    @property
    def category_ids(self) -> list[str]:
        return list(self.categories)


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path | None = None) -> ESGTaxonomy:
    taxonomy_path = Path(path or settings.taxonomy_path)
    try:
        with taxonomy_path.open(encoding="utf-8") as taxonomy_file:
            return ESGTaxonomy.model_validate(json.load(taxonomy_file))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f"Unable to load ESG taxonomy from {taxonomy_path}",
            error_code="INVALID_TAXONOMY",
            details={"path": str(taxonomy_path), "reason": str(exc)},
        ) from exc


def get_taxonomy() -> ESGTaxonomy:
    return load_taxonomy(settings.taxonomy_path)
