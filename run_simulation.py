"""
Run the Monte Carlo tournament simulation and report each live team's
probability of winning the World Cup.

Prints the Monte Carlo estimate alongside the closed-form analytic answer
(they should agree closely), and writes results/simulation.json with a
timestamp so the forecast is a dated, pre-result artifact.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src.simulate import (
    FINALIST,
    SEMIFINAL,
    live_teams,
    load_ratings,
    load_model,
    build_advance_table,
    run_monte_carlo,
    analytic_champion_probs,
)

N_SIMS = 50000
SEED = 42


def main():
    ratings = load_ratings()
    model, features, classes = load_model()
    adv = build_advance_table(model, features, classes, ratings)

    mc = run_monte_carlo(adv, n=N_SIMS, seed=SEED)
    exact = analytic_champion_probs(adv)

    teams = sorted(live_teams(), key=lambda t: exact[t], reverse=True)

    print(f"World Cup 2026 - championship probabilities ({N_SIMS:,} simulations)")
    print(f"State: {FINALIST} in final; {SEMIFINAL[0]} v {SEMIFINAL[1]} to play\n")
    print(f"{'Team':12}{'MonteCarlo':>12}{'Analytic':>12}")
    for t in teams:
        print(f"{t:12}{mc[t]:>11.1%}{exact[t]:>12.1%}")

    max_gap = max(abs(mc[t] - exact[t]) for t in teams)
    print(f"\nLargest MC-vs-analytic gap: {max_gap:.4f}")

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_simulations": N_SIMS,
        "seed": SEED,
        "state": {
            "finalist": FINALIST,
            "remaining_semifinal": list(SEMIFINAL),
        },
        "champion_probabilities_monte_carlo": {t: round(mc[t], 4) for t in teams},
        "champion_probabilities_analytic": {t: round(exact[t], 4) for t in teams},
        "model": "logreg-v1",
    }
    out_path = Path("results") / "simulation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()