# Contributing

## Setup

```bash
python -m venv .venv
pip install -e ".[dev,ui,openai]"
pre-commit install
```

Copy `.env.example` to `.env`. Tests do not require external services or provider keys.

## Before opening a change

```bash
ruff format src ui tests
ruff check src ui tests
pytest
```

Keep provider-specific imports inside lazy factories, put ESG taxonomy data in `config/esg_taxonomy.json`, and do not add fabricated fallbacks that could be confused with evidence-based analysis. New endpoints need typed request/response contracts and tests. Never include source documents, vector data, secrets, or `.env` files in a contribution.
