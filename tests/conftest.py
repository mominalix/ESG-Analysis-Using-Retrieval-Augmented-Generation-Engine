from __future__ import annotations

import os


os.environ.update(
    {
        "ENVIRONMENT": "test",
        "VECTOR_STORE_TYPE": "memory",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "ADMIN_TOKEN": "test-admin-token-with-safe-length",
        "LOG_FORMAT": "console",
    }
)
