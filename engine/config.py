"""Application configuration and shared constants."""
from __future__ import annotations

from pathlib import Path

# Default database location. data/ is gitignored — never commit it.
DEFAULT_DB_URL = "sqlite:///data/hypocxe.db"

# ROI harmonisation table used by engine/parser.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
ROI_ALIASES_PATH = REPO_ROOT / "reference" / "roi_aliases.yaml"
