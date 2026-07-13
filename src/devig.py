"""Convert bookmaker odds to fair (no-vig) probabilities.

A three-way book's implied probabilities (1 / decimal odds) sum to more
than 1. The excess is the 'overround' or vig — the bookmaker's margin.
Removing it (rescaling to sum to 1) recovers the market's fair estimate.
"""
import numpy as np


def implied_probabilities(decimal_odds):
    """Raw implied probability of each outcome: 1 / decimal odds.
    Sums to > 1 because of the bookmaker's margin."""
    odds = np.asarray(decimal_odds, dtype=float)
    return 1.0 / odds


def overround(decimal_odds):
    """The bookmaker's margin: how much implied probabilities exceed 1.
    e.g. 0.05 means a 5% built-in edge."""
    return float(implied_probabilities(decimal_odds).sum() - 1.0)


def remove_vig(decimal_odds):
    """Fair (no-vig) probabilities: implied probabilities rescaled to
    sum to exactly 1. This is the simple normalisation method."""
    raw = implied_probabilities(decimal_odds)
    return raw / raw.sum()