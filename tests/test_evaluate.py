import numpy as np
from src.evaluate import rps, brier, log_loss, accuracy

# One match, home win, ordered [home, draw, away]
PERFECT = np.array([[1.0, 0.0, 0.0]])
UNIFORM = np.array([[1/3, 1/3, 1/3]])
HOME_WIN = np.array([0])


def test_perfect_forecast_scores_zero():
    assert rps(PERFECT, HOME_WIN) == 0.0
    assert brier(PERFECT, HOME_WIN) == 0.0


def test_accuracy_perfect():
    assert accuracy(PERFECT, HOME_WIN) == 1.0


def test_worse_forecast_scores_higher():
    confident_wrong = np.array([[0.0, 0.0, 1.0]])
    assert rps(confident_wrong, HOME_WIN) > rps(UNIFORM, HOME_WIN)
    assert brier(confident_wrong, HOME_WIN) > brier(UNIFORM, HOME_WIN)
    assert log_loss(confident_wrong, HOME_WIN) > log_loss(UNIFORM, HOME_WIN)


def test_rps_respects_outcome_ordering():
    # Forecast leans home-win; a draw is "closer" to that than an away win,
    # so the draw should be penalised less. Brier can't see this; RPS can.
    lean_home = np.array([[0.6, 0.25, 0.15]])
    draw = np.array([1])
    away = np.array([2])
    assert rps(lean_home, draw) < rps(lean_home, away)


def test_known_rps_value():
    # Hand-computed: probs [0.5, 0.3, 0.2], outcome home win.
    # cum probs [0.5, 0.8], cum actual [1, 1]
    # ((0.5-1)^2 + (0.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
    probs = np.array([[0.5, 0.3, 0.2]])
    assert abs(rps(probs, HOME_WIN) - 0.145) < 1e-9