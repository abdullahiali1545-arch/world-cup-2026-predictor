"""
Generate dated win/draw/loss predictions for the day's fixtures.

Reads historical results, computes current Elo ratings, then predicts
every fixture in fixtures.json dated today (UTC). Writes the output to
predictions/YYYY-MM-DD.json — the same format used since the project began.

Designed to run unattended (e.g. via GitHub Actions) as well as by hand.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.elo import compute_ratings, predict_match


def load_history(path="data/results.csv"):
    """Load the historical match data, parsed and sorted by date ascending."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_todays_fixtures(target_date, path="fixtures.json"):
    """Return the list of fixtures in fixtures.json matching target_date (YYYY-MM-DD)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [fx for fx in data["fixtures"] if fx["date"] == target_date]


def build_predictions(fixtures, ratings):
    """Turn each fixture into a prediction record using the Elo baseline."""
    predictions = []
    for fx in fixtures:
        home, away = fx["home_team"], fx["away_team"]
        # Fail loudly on an unknown team rather than silently defaulting.
        for team in (home, away):
            if team not in ratings:
                raise ValueError(
                    f"'{team}' not found in ratings — check spelling against the dataset."
                )
        probs = predict_match(ratings[home], ratings[away])
        predictions.append({
            "home_team": home,
            "away_team": away,
            "home_rating": round(ratings[home]),
            "away_rating": round(ratings[away]),
            "prob_home_win": round(probs["win"], 3),
            "prob_draw": round(probs["draw"], 3),
            "prob_away_win": round(probs["loss"], 3),
        })
    return predictions


def main():
    target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fixtures = load_todays_fixtures(target_date)
    if not fixtures:
        print(f"No fixtures dated {target_date} in fixtures.json — nothing to predict.")
        return

    df = load_history()
    ratings = compute_ratings(df)
    predictions = build_predictions(fixtures, ratings)

    output = {
        "match_date": target_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "elo-baseline-v1",
        "predictions": predictions,
    }

    out_path = Path("predictions") / f"{target_date}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()