"""Market benchmark for the World Cup semifinals.

De-vigs bookmaker odds and places the market's fair probabilities
alongside the model and baseline for the same fixtures. The market is
the benchmark, not a target: this shows the gap honestly.

Odds: FanDuel 90-minute money line, 12 July 2026 (American -> decimal).
"""
import json
import joblib
import pandas as pd

from src.elo import compute_ratings, predict_match
from src.devig import remove_vig, overround

FIXTURES = [
    {"home": "France", "away": "Spain", "odds": [2.35, 3.20, 3.20]},
    {"home": "England", "away": "Argentina", "odds": [2.60, 2.90, 3.10]},
]

matches = pd.read_csv("data/results.csv", parse_dates=["date"])
matches = matches.sort_values("date")
ratings = compute_ratings(matches)

bundle = joblib.load("src/model.joblib")
model = bundle["model"]
features = bundle["features"]
ci = {c: i for i, c in enumerate(model.classes_)}


def model_probs(home, away):
    elo_diff = ratings[home] - ratings[away]
    X = pd.DataFrame([{"elo_diff": elo_diff, "neutral": 1}])[features]
    raw = model.predict_proba(X)[0]
    return [raw[ci["home_win"]], raw[ci["draw"]], raw[ci["home_loss"]]]


def baseline_probs(home, away):
    p = predict_match(ratings[home], ratings[away])
    return [p["win"], p["draw"], p["loss"]]


def show(label, probs):
    print(f"{label:10}{probs[0]:>8.1%}{probs[1]:>8.1%}{probs[2]:>8.1%}")


records = []
for fx in FIXTURES:
    home = fx["home"]
    away = fx["away"]
    market = remove_vig(fx["odds"])
    mdl = model_probs(home, away)
    base = baseline_probs(home, away)
    vig = overround(fx["odds"])

    print("")
    print(f"{home} vs {away}   (bookmaker margin {vig:.1%})")
    print(f"{'':10}{'Home':>8}{'Draw':>8}{'Away':>8}")
    show("Market", market)
    show("Model", mdl)
    show("Baseline", base)

    records.append({
        "home_team": home,
        "away_team": away,
        "market": [round(float(x), 4) for x in market],
        "model": [round(float(x), 4) for x in mdl],
        "baseline": [round(float(x), 4) for x in base],
        "bookmaker_margin": round(vig, 4),
    })

out = {
    "note": "Probabilities ordered [home, draw, away]. Odds: FanDuel 90-min money line, 12 Jul 2026.",
    "fixtures": records,
}
with open("results/market_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("")
print("Saved results/market_benchmark.json")