import numpy as np
from src.devig import implied_probabilities, remove_vig, overround


def test_implied_prob_of_evens():
    assert abs(implied_probabilities([2.0])[0] - 0.5) < 1e-12


def test_remove_vig_sums_to_one():
    assert abs(remove_vig([2.35, 3.20, 3.20]).sum() - 1.0) < 1e-12


def test_overround_positive_for_real_book():
    assert overround([2.35, 3.20, 3.20]) > 0


def test_fair_book_unchanged():
    # 1/2 + 1/4 + 1/4 = 1 already, so nothing should change.
    fair = remove_vig([2.0, 4.0, 4.0])
    assert np.allclose(fair, [0.5, 0.25, 0.25])
    assert abs(overround([2.0, 4.0, 4.0])) < 1e-12