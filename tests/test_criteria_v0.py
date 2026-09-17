"""Tests for engine/criteria_v0.py: the frozen rule tables and their
tamper-evident fingerprint.

CRITERIA_SHA256 is pinned to a literal, computed-once value: if this test
ever fails, either criteria_v0.py's rule data changed (and CLAUDE.md rule
2 says that needs the user's explicit "change criteria") or this test's
pin needs updating to match a deliberate, authorized change — never the
other way around (this test must never call criteria_v0._criteria_snapshot()
and compare a hash to itself; that would prove nothing).
"""
from __future__ import annotations

import pytest

from engine import criteria_v0 as V

# Computed once from the criteria data as implemented; pin, don't regenerate.
PINNED_CRITERIA_SHA256 = "92a3a36e76cab9e4b475ad66cac8dcea4bb2bc7646a7761e915b2b07e6d03b6e"


def test_criteria_sha256_matches_pinned_value():
    assert V.CRITERIA_SHA256 == PINNED_CRITERIA_SHA256


def test_criteria_sha256_is_deterministic():
    """Recomputing from the same tables gives the same hash — it's a pure
    function of the data, not e.g. dict-ordering-dependent."""
    second = V._canonical_json(V._criteria_snapshot())
    import hashlib
    assert hashlib.sha256(second.encode("utf-8")).hexdigest() == V.CRITERIA_SHA256


# --------------------------------------------------------------------------- #
# OAR_SCALE boundaries — task's required exact values: 2, 5, 10, -2, -5
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pct,expected_base", [
    (10.0000001, 3.0),   # just above 10 -> Excellent
    (10.0, 2.0),          # exactly 10 -> Moderate (">10%" is strict)
    (5.0, 2.0),            # exactly 5 -> Moderate (band's own lower bound)
    (4.9999999, 1.0),   # just below 5 -> Minor
    (2.0, 1.0),             # exactly 2 -> Minor (band's own lower bound)
    (1.9999999, 0.0),   # just below 2 -> Equivalent
    (0.0, 0.0),              # dead center -> Equivalent
    (-2.0, 0.0),            # exactly -2 -> the "-5..-2" closed band, not Equivalent
    (-1.9999999, 0.0),  # just above -2 -> Equivalent
    (-5.0, 0.0),            # exactly -5 -> still the closed "-5..-2" band
    (-5.0000001, -2.0),  # just below -5 -> Penalty
])
def test_oar_scale_boundaries(pct, expected_base):
    assert V.band_base_score(pct, V.OAR_SCALE) == pytest.approx(expected_base)


# --------------------------------------------------------------------------- #
# TARGET_SCALE boundaries — task's required exact values: 2, 1, -1, 0
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pct,expected_base", [
    (2.0000001, 3.0),   # just above 2 -> Excellent
    (2.0, 1.5),             # exactly 2 -> Minor (band's own upper bound)
    (1.0, 1.5),             # exactly 1 -> Minor (band's own lower bound)
    (0.9999999, 0.0),   # just below 1 -> Equivalent
    (0.0, 0.0),              # dead center -> Equivalent
    (-0.9999999, 0.0),  # just above -1 -> Equivalent
    (-1.0, -2.0),           # exactly -1 -> Penalty (fixed threshold is closed here)
    (-1.0000001, -2.0),  # just below -1 -> Penalty
])
def test_target_scale_boundaries(pct, expected_base):
    assert V.band_base_score(pct, V.TARGET_SCALE) == pytest.approx(expected_base)


def test_oar_scale_bands_are_a_complete_nonoverlapping_partition():
    """Every value from -100 to 100 (in fine steps) matches exactly one band."""
    import numpy as np
    for pct in np.arange(-100.0, 100.0, 0.1):
        matches = [b for b in V.OAR_SCALE if b.matches(pct)]
        assert len(matches) == 1, f"pct={pct} matched {len(matches)} bands"


def test_target_scale_bands_are_a_complete_nonoverlapping_partition():
    import numpy as np
    for pct in np.arange(-100.0, 100.0, 0.1):
        matches = [b for b in V.TARGET_SCALE if b.matches(pct)]
        assert len(matches) == 1, f"pct={pct} matched {len(matches)} bands"


def test_band_base_score_raises_if_somehow_unmatched():
    empty_scale = ()
    with pytest.raises(ValueError):
        V.band_base_score(0.0, empty_scale)


# --------------------------------------------------------------------------- #
# Multiplier tables
# --------------------------------------------------------------------------- #


def test_oar_hotspot_multiplier_table():
    assert dict(V.OAR_HOTSPOT_MULTIPLIER) == {1: 3.0, 2: 2.0, 3: 1.0}


def test_target_multiplier_table():
    assert dict(V.TARGET_MULTIPLIER) == {1: 5.0, 2: 3.0, 3: 1.0}


# --------------------------------------------------------------------------- #
# Structure groups (§3.7 radar chart)
# --------------------------------------------------------------------------- #


def test_structure_groups_cover_the_manuals_radar_categories():
    groups = set(V.STRUCTURE_GROUPS.values())
    assert groups == {"Bladder", "Rectum", "Bowel", "Marrow", "Femoral hd", "Kidneys"}


def test_bowel_group_includes_all_four_structures():
    bowel = {roi for roi, g in V.STRUCTURE_GROUPS.items() if g == "Bowel"}
    assert bowel == {"Small Bowel", "Bowel Bag", "Sigmoid", "Duodenum"}


def test_frozen_dataclasses_are_actually_frozen():
    band = V.OAR_SCALE[0]
    with pytest.raises(Exception):
        band.base = 999.0


def test_structure_groups_is_a_read_only_mapping():
    with pytest.raises(TypeError):
        V.STRUCTURE_GROUPS["New"] = "Nope"
