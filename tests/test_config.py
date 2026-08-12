from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.taxonomy import ESGTaxonomy


def test_csv_and_extension_configuration_is_normalized():
    settings = Settings(
        _env_file=None,
        allowed_origins_env="https://one.example, https://two.example",
        supported_extensions_env="pdf, .md",
    )

    assert settings.allowed_origins == ["https://one.example", "https://two.example"]
    assert settings.supported_extensions == (".pdf", ".md")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)


def test_production_requires_admin_token():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            environment="production",
            admin_token=None,
            allowed_origins_env="https://app.example",
        )


def test_repository_and_packaged_taxonomies_stay_in_sync():
    repository = Settings(_env_file=None).taxonomy_path
    packaged = repository.parents[1] / "src" / "core" / "esg_taxonomy.json"
    assert ESGTaxonomy.model_validate_json(repository.read_text(encoding="utf-8")) == (
        ESGTaxonomy.model_validate_json(packaged.read_text(encoding="utf-8"))
    )
