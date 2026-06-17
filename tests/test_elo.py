from src.elo import expected_score, update_ratings, compute_ratings
import pandas as pd


def test_equal_ratings_give_half():
    """Two equally-rated teams should each have a 50% chance."""
    assert expected_score(1500, 1500) == 0.5


def test_stronger_team_is_favoured():
    """A 400-point-stronger team should be roughly a 91% favourite."""
    result = expected_score(1900, 1500)
    assert 0.90 < result < 0.92


def test_probabilities_sum_to_one():
    """A beating B and B beating A must together account for 100%."""
    a_beats_b = expected_score(1600, 1400)
    b_beats_a = expected_score(1400, 1600)
    assert abs((a_beats_b + b_beats_a) - 1.0) < 0.0001

def test_favourite_winning_gains_little():
    new_a, new_b = update_ratings(1900, 1500, score_a=1.0)
    gain = new_a - 1900
    assert 0 < gain < 5
    

def test_underdog_winning_gains_a_lot():
    """A weak underdog who wins should gain a large chunk of points."""
    new_a, new_b = update_ratings(1500, 1900, score_a=1.0)
    gain = new_a - 1500
    assert gain > 25


def test_points_are_conserved():
    """Whatever one team gains, the other must lose — points only move."""
    start_a, start_b = 1600, 1400
    new_a, new_b = update_ratings(start_a, start_b, score_a=1.0)
    a_change = new_a - start_a
    b_change = new_b - start_b
    assert abs(a_change + b_change) < 0.0001

def test_winner_outrates_loser():
    """If one team always beats another, it should end up rated higher."""
    matches = pd.DataFrame({
        "home_team": ["A", "A", "A"],
        "away_team": ["B", "B", "B"],
        "home_score": [1, 1, 1],
        "away_score": [0, 0, 0],
    })
    ratings = compute_ratings(matches)
    assert ratings["A"] > ratings["B"]


def test_unplayed_fixtures_are_ignored():
    """A match with no score must not create or change any ratings."""
    matches = pd.DataFrame({
        "home_team": ["A"],
        "away_team": ["B"],
        "home_score": [None],
        "away_score": [None],
    })
    ratings = compute_ratings(matches)
    assert ratings == {}