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


# The label a dose-regimen picker (pages/0_new_case.py, pages/1_patient_data.py's
# Edit dialog) shows for each choice, in display order — kept in one place
# so both pages build identical labels and neither hand-writes the rx/
# fractions text.
DOSE_REGIMEN_UNSET = "— not yet specified —"


def dose_regimen_choices() -> dict[str, DoseRegimen]:
    """{label: DoseRegimen} for every regimen, in display order. Doesn't
    include the "not yet specified" placeholder — callers add that
    themselves as the first option, since only the New Case/Edit widgets
    (not e.g. a filter) need it."""
    return {
        f"Hypo — {DOSE_REGIMEN_RX[DoseRegimen.HYPO][0]/100:g} Gy / "
        f"{DOSE_REGIMEN_RX[DoseRegimen.HYPO][1]} fx": DoseRegimen.HYPO,
        f"Conv — {DOSE_REGIMEN_RX[DoseRegimen.CONV][0]/100:g} Gy / "
        f"{DOSE_REGIMEN_RX[DoseRegimen.CONV][1]} fx": DoseRegimen.CONV,
    }
