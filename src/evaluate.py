"""Evaluation metrics for probabilistic match forecasts.

Predictions are arrays of shape (n_matches, 3) with columns
ordered [home_win, draw, away_win]. Outcomes are integers:
0 = home win, 1 = draw, 2 = away win.
"""
import numpy as np


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score, averaged over matches. Lower is better.

    Compares the *cumulative* forecast distribution to the cumulative
    actual outcome, so it respects the ordering home > draw > away.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(outcomes)
    actual = np.zeros((n, 3))
    actual[np.arange(n), outcomes] = 1.0
    cum_probs = np.cumsum(probs, axis=1)
    cum_actual = np.cumsum(actual, axis=1)
    # Sum over the first 2 cumulative steps (the 3rd is always 1-1=0)
    per_match = np.sum((cum_probs - cum_actual) ** 2, axis=1) / 2
    return float(np.mean(per_match))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score, averaged over matches. Lower is better."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(outcomes)
    actual = np.zeros((n, 3))
    actual[np.arange(n), outcomes] = 1.0
    return float(np.mean(np.sum((probs - actual) ** 2, axis=1)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the actual outcome. Lower is better."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    p_actual = np.clip(probs[np.arange(len(outcomes)), outcomes], eps, 1.0)
    return float(np.mean(-np.log(p_actual)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Fraction of matches where the highest-probability outcome happened."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))