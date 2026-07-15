"""
Monte Carlo simulation of the remaining World Cup 2026 bracket.

State as of 15 Jul 2026: Spain have won their semi-final (2-0 v France)
and are already in the final. England v Argentina is the remaining
semi-final; its winner meets Spain in the final.

The simulation therefore fixes Spain's final berth, plays the England v
Argentina semi-final, then plays the final, many thousands of times, and
counts how often each of the three live teams lifts the trophy.

Knockout matches cannot be drawn, so the model's draw probability is
resolved as extra-time/penalties: split evenly between the two sides
(a near-coin-flip). Because knockout games are on neutral ground, the
model is symmetrised by averaging both home/away assignments, so no team
gets an arbitrary 'home' edge.

With only three live teams the champion probabilities are also computable
in closed form; `analytic_champion_probs` provides that, and the test
suite checks the Monte Carlo estimate matches it within tolerance.
"""
from collections import Counter
import joblib
import numpy as np
import pandas as pd

from src.elo import compute_ratings

MODEL_PATH = "src/model.joblib"

# Team already through to the final.
FINALIST = "Spain"
# The remaining semi-final; winner meets FINALIST in the final.
SEMIFINAL = ("England", "Argentina")


def live_teams():
    return [FINALIST, *SEMIFINAL]


def load_ratings(history_path="data/results.csv"):
    df = pd.read_csv(history_path, parse_dates=["date"]).sort_values("date")
    return compute_ratings(df)


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    model = bundle["model"]
    return model, bundle["features"], list(model.classes_)


def _match_probs(model, features, classes, rating_a, rating_b):
    """Model's [P(A win), P(draw), P(B win)] for a neutral match,
    symmetrised by averaging both home/away assignments."""
    idx = {c: i for i, c in enumerate(classes)}

    def one(diff):
        X = pd.DataFrame([[diff, 1]], columns=features)  # neutral = 1
        p = model.predict_proba(X)[0]
        return p[idx["home_win"]], p[idx["draw"]], p[idx["home_loss"]]

    a1, d1, b1 = one(rating_a - rating_b)   # A as home
    b2, d2, a2 = one(rating_b - rating_a)   # B as home (flip back to A)
    return 0.5 * (a1 + a2), 0.5 * (d1 + d2), 0.5 * (b1 + b2)


def advance_prob(model, features, classes, ratings, team_a, team_b):
    """Probability team_a beats team_b in a knockout (draw split 50/50)."""
    a_win, draw, _ = _match_probs(
        model, features, classes, ratings[team_a], ratings[team_b]
    )
    return a_win + 0.5 * draw


def build_advance_table(model, features, classes, ratings):
    """Precompute P(a beats b) for every ordered pair of the live teams."""
    teams = live_teams()
    return {
        (a, b): advance_prob(model, features, classes, ratings, a, b)
        for a in teams for b in teams if a != b
    }


def simulate_once(adv, rng):
    """Play the remaining bracket once. Returns champion."""
    a, b = SEMIFINAL
    semi_winner = a if rng.random() < adv[(a, b)] else b
    # Final: semi_winner vs the team already through.
    if rng.random() < adv[(semi_winner, FINALIST)]:
        return semi_winner
    return FINALIST


def run_monte_carlo(adv, n=50000, seed=42):
    rng = np.random.default_rng(seed)
    counts = Counter()
    for _ in range(n):
        counts[simulate_once(adv, rng)] += 1
    return {team: counts[team] / n for team in live_teams()}


def analytic_champion_probs(adv):
    """Exact champion probabilities for the current three-team state."""
    a, b = SEMIFINAL
    f = FINALIST
    reach_a = adv[(a, b)]         # a wins the semi
    reach_b = adv[(b, a)]         # b wins the semi
    return {
        a: reach_a * adv[(a, f)],
        b: reach_b * adv[(b, f)],
        f: reach_a * adv[(f, a)] + reach_b * adv[(f, b)],
    }