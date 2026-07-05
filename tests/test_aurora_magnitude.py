from agents.aurora_processing import _assign_label


def test_big_when_move_exceeds_band():
    assert _assign_label(3.38, 0.01) == "big"    # +3.38% > 1%
    assert _assign_label(-2.0, 0.01) == "big"    # -2% magnitude > 1%


def test_small_when_move_within_band():
    assert _assign_label(0.6163, 0.01) == "small"  # 0.62% <= 1%
    assert _assign_label(-0.5, 0.01) == "small"


def test_band_scales_with_threshold():
    assert _assign_label(1.5, 0.02) == "small"   # 1.5% <= 2%
    assert _assign_label(2.5, 0.02) == "big"
