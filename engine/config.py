"""Application configuration and shared constants."""
from __future__ import annotations

from pathlib import Path

from engine.schemas import DoseRegimen

# Default database location. data/ is gitignored — never commit it.
DEFAULT_DB_URL = "sqlite:///data/hypocxe.db"

# ROI harmonisation table used by engine/parser.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
ROI_ALIASES_PATH = REPO_ROOT / "reference" / "roi_aliases.yaml"

# Prescription (dose in cGy, fractions) fixed by protocol per dose regimen —
# the New Case form (Module 0) picks the regimen; rx_cgy/fractions are
# derived here, never typed in by hand.
DOSE_REGIMEN_RX: dict[DoseRegimen, tuple[float, int]] = {
    DoseRegimen.HYPO: (4400.0, 20),
    DoseRegimen.CONV: (4500.0, 25),
}


def rx_for_dose_regimen(regimen: DoseRegimen) -> tuple[float, int]:
    """(rx_cgy, fractions) for a dose regimen."""
    return DOSE_REGIMEN_RX[regimen]
