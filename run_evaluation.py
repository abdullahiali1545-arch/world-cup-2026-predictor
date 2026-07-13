"""Model vs baseline evaluation on held-out matches (after 2022-01-01).

Prints the comparison table and saves a reliability diagram to results/.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import build_training_data
from src.elo import predict_match
from src.evaluate import rps, brier, log_loss, accuracy

CUTOFF = "2022-01-01"
OUTCOME_TO_INT = {"home_win": 0, "draw": 1, "home_loss": 2}

# 1. Rebuild the leak-free training rows and take the held-out slice.
matches = pd.read_csv("data/results.csv", parse_dates=["date"])
data = build_training_data(matches)
test = data[data["date"] >= CUTOFF].reset_index(drop=True)
outcomes = test["outcome"].map(OUTCOME_TO_INT).to_numpy()
print(f"Held-out matches after {CUTOFF}: {len(test)}")

# 2. Model probabilities, columns re-ordered to [home_win, draw, home_loss].
bundle = joblib.load("src/model.joblib")
model = bundle["model"]
features = bundle["features"]
raw = model.predict_proba(test[features])
col = {label: i for i, label in enumerate(model.classes_)}
model_probs = raw[:, [col["home_win"], col["draw"], col["home_loss"]]]

# 3. Baseline probabilities. predict_match only depends on the rating
#    difference, so passing (elo_diff, 0) is exact.
base_probs = np.array([
    [p["win"], p["draw"], p["loss"]]
    for p in (predict_match(d, 0.0) for d in test["elo_diff"])
])

# 4. The table.
print(f"\n{'Metric':<10}{'Baseline':>12}{'Model':>12}")
for name, fn in [("RPS", rps), ("Brier", brier),
                 ("Log loss", log_loss), ("Accuracy", accuracy)]:
    print(f"{name:<10}{fn(base_probs, outcomes):>12.4f}"
          f"{fn(model_probs, outcomes):>12.4f}")

# 5. Reliability diagram: every (match, outcome) predicted probability
#    vs how often that outcome actually happened, in 10 bins.
actual = np.zeros_like(model_probs)
actual[np.arange(len(outcomes)), outcomes] = 1.0
fig, ax = plt.subplots(figsize=(6, 6))
bins = np.linspace(0, 1, 11)
for probs, label in [(base_probs, "Baseline"), (model_probs, "Model")]:
    p, a = probs.ravel(), actual.ravel()
    idx = np.digitize(p, bins) - 1
    centers = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    freqs = [a[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax.plot(centers, freqs, marker="o", label=label)
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Actual frequency")
ax.set_title(f"Reliability diagram (test matches after {CUTOFF})")
ax.legend()
fig.tight_layout()
fig.savefig("results/reliability.png", dpi=150)
print("\nSaved results/reliability.png")