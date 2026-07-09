import pandas as pd
from src.elo import update_ratings


def build_training_data(matches, k=30, base_rating=1500):
    """
    Walk every played match in date order and record one row of training
    data per match: each team's Elo rating BEFORE the match, whether it was
    on neutral ground, and the actual outcome (the label to learn).

    The Elo ratings are the values BEFORE the match is applied, so no future
    information leaks into the features.

    matches: DataFrame with columns
             date, home_team, away_team, home_score, away_score, neutral.
    Returns a DataFrame with one row per played match.
    """
    matches = matches.sort_values("date")
    ratings = {}
    rows = []
    for row in matches.itertuples(index=False):
        # Skip fixtures that haven't been played yet.
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue

        # Pre-match ratings: exactly what we would have known before kickoff.
        home_rating = ratings.get(row.home_team, base_rating)
        away_rating = ratings.get(row.away_team, base_rating)

        # The actual outcome, from the home team's point of view.
        if row.home_score > row.away_score:
            outcome = "home_win"
            home_result = 1.0
        elif row.home_score < row.away_score:
            outcome = "home_loss"
            home_result = 0.0
        else:
            outcome = "draw"
            home_result = 0.5

        # Record the training row using PRE-match information only.
        rows.append({
            "date": row.date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "elo_diff": home_rating - away_rating,
            "neutral": int(row.neutral),
            "outcome": outcome,
        })

        # Only now update the ratings, ready for the next match.
        new_home, new_away = update_ratings(
            home_rating, away_rating, score_a=home_result, k=k
        )
        ratings[row.home_team] = new_home
        ratings[row.away_team] = new_away

    return pd.DataFrame(rows)