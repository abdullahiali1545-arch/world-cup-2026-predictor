import pandas as pd

def expected_score(rating_a, rating_b):
    """Probability that team A beats team B, based on their Elo ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
def update_ratings(rating_a, rating_b, score_a, k=30):
    """
    Update both teams' ratings after a match.

    score_a is the actual result for team A: 1.0 win, 0.5 draw, 0.0 loss.
    Returns the new (rating_a, rating_b).
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    score_b = 1 - score_a

    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)

    return new_rating_a, new_rating_b

def compute_ratings(matches, k=30, base_rating=1500):
    """
    Walk every played match in date order and return each team's final Elo rating.

    matches: a DataFrame sorted by date, with columns
             home_team, away_team, home_score, away_score.
    Returns a dictionary mapping team name -> final rating.
    """
    ratings = {}

    for row in matches.itertuples(index=False):
        # Skip fixtures that haven't been played yet (no score).
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        # Look up current ratings; new teams start at base_rating.
        home_rating = ratings.get(row.home_team, base_rating)
        away_rating = ratings.get(row.away_team, base_rating)

        # Translate the score into a result for the home team.
        if row.home_score > row.away_score:
            home_result = 1.0
        elif row.home_score < row.away_score:
            home_result = 0.0
        else:
            home_result = 0.5

        # Update both teams and store the new ratings.
        new_home, new_away = update_ratings(
            home_rating, away_rating, score_a=home_result, k=k
        )
        ratings[row.home_team] = new_home
        ratings[row.away_team] = new_away

    return ratings

