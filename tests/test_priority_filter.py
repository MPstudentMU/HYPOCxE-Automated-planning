import numpy as np
import pandas as pd
import pytest

from engine.priority_filter import (
    PrioritySelection, category_breakdown, cohort_summary,
    filter_goal_scores, paired_consistency, quality_index_table,
)


def _row(pt, plan, prio, cat, base, mult, max_base=3, scorable=True):
    return dict(patient=pt, plan=plan, priority=prio, category=cat,
                base_score=base, weighted_score=base * mult,
                max_weighted=max_base * mult, scorable=scorable,
                structure_group="Rectum")


@pytest.fixture
def goals():
    rows = [
        # Pt1 Manual: P1 OAR +3 (x3), P2 OAR +1 (x2), P3 OAR 0 (x1)
        _row("Pt1", "Manual", 1, "OAR Sparing", 3, 3),
        _row("Pt1", "Manual", 2, "OAR Sparing", 1, 2),
        _row("Pt1", "Manual", 3, "OAR Sparing", 0, 1),
        # Pt1 Auto: P1 Target +3 (x5), P1 zero-limit OAR 0 (x3, max base 1)
        _row("Pt1", "Auto", 1, "Target Coverage", 3, 5),
        _row("Pt1", "Auto", 1, "OAR Sparing", 0, 3, max_base=1),
        _row("Pt1", "Auto", 2, "OAR Sparing", -2, 2),
        # pending correction -> never counted
        _row("Pt1", "Auto", 2, "OAR Sparing", 3, 2, scorable=False),
    ]
    return pd.DataFrame(rows)


def _qi(tbl, plan):
    return tbl.loc[(tbl.patient == "Pt1") & (tbl.plan == plan),
                   "quality_index"].iloc[0]


def test_all_priorities_matches_original(goals):
    t = quality_index_table(goals, PrioritySelection.all())
    assert _qi(t, "Manual") == pytest.approx((9 + 2 + 0) / (9 + 6 + 3) * 100)
    assert _qi(t, "Auto") == pytest.approx((15 + 0 - 4) / (15 + 3 + 6) * 100)


def test_priority_1_only(goals):
    t = quality_index_table(goals, PrioritySelection((1,)))
    assert _qi(t, "Manual") == pytest.approx(9 / 9 * 100)
    assert _qi(t, "Auto") == pytest.approx(15 / 18 * 100)
    assert (t["priority_filter"] == "Priority 1").all()


def test_priority_2_only_excludes_unscorable(goals):
    t = quality_index_table(goals, PrioritySelection((2,)))
    assert _qi(t, "Auto") == pytest.approx(-4 / 6 * 100)
    assert t.loc[t.plan == "Auto", "n_goals"].iloc[0] == 1


def test_empty_subset_is_nan_not_zero(goals):
    t = quality_index_table(goals, PrioritySelection((3,)))
    assert np.isnan(_qi(t, "Auto"))
    assert t.loc[t.plan == "Auto", "n_goals"].iloc[0] == 0
    assert _qi(t, "Manual") == pytest.approx(0.0)


def test_multi_selection(goals):
    t = quality_index_table(goals, PrioritySelection((1, 2)))
    assert _qi(t, "Manual") == pytest.approx((9 + 2) / (9 + 6) * 100)


def test_selection_validation_and_labels():
    assert PrioritySelection((1, 2, 3)).is_all
    assert PrioritySelection((3, 1, 1)).levels == (1, 3)
    assert PrioritySelection((2, 1)).label == "Priority 1 + Priority 2"
    with pytest.raises(ValueError):
        PrioritySelection(())
    with pytest.raises(ValueError):
        PrioritySelection((4,))


def test_filter_and_breakdown(goals):
    f = filter_goal_scores(goals, PrioritySelection((1,)))
    assert set(f["priority"]) == {1}
    bd = category_breakdown(goals, PrioritySelection((1,)))
    auto_t = bd[(bd.plan == "Auto") & (bd.category == "Target Coverage")]
    assert auto_t["weighted_score"].iloc[0] == 15


def test_summary_and_paired_tests_run():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        for plan, spread in [("Manual", 3), ("Auto", 1), ("Auto+Manual", 1)]:
            base = int(rng.integers(-2, 4)) if spread == 3 else 2
            rows.append(_row(f"Pt{i}", plan, 1, "OAR Sparing", base, 3))
    df = pd.DataFrame(rows)
    qi = quality_index_table(df, PrioritySelection.all())
    s = cohort_summary(qi)
    assert list(s["plan"]) == ["Manual", "Auto", "Auto+Manual"]
    assert s.loc[s.plan == "Auto", "n"].iloc[0] == 6
    tests = paired_consistency(qi)
    assert list(tests["n_pairs"]) == [6, 6]
