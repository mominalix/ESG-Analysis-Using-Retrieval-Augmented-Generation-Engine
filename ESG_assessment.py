"""Backward-compatible Streamlit entry point.

Prefer ``streamlit run ui/main.py``. This module remains so existing deployment
commands do not execute the repository's obsolete prototype implementation.
"""

from ui.main import main


if __name__ == "__main__":
    main()
