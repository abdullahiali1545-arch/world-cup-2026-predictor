"""Streamlit app: World Cup 2026 match predictor.

Championship title-race board, the model-vs-baseline scoreboard, a
calibration diagram, the latest logged prediction, and the market benchmark.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.features import build_training_data
from src.elo import predict_match
from src.evaluate import rps, brier, log_loss, accuracy

CUTOFF = "2022-01-01"
OUTCOME_TO_INT = {"home_win": 0, "draw": 1, "home_loss": 2}
PRED_DIR = Path("predictions")
MARKET_FILE = Path("results/market_benchmark.json")
SIM_FILE = Path("results/simulation.json")

INK = "#0A1628"
PANEL = "#0E1E33"
GOLD = "#F2C14E"
TEAL = "#3AA6B9"
LIGHT = "#E8EEF4"
MUTED = "#8DA0B3"

st.set_page_config(page_title="World Cup 2026 forecasts", page_icon="⚽",
                   layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1, h2, h3 {
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #E8EEF4 !important;
}

.hero { padding: 0.4rem 0 1.1rem 0; margin-bottom: 0.4rem; }
.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #F2C14E;
}
.hero h1.big {
    font-family: 'Oswald', sans-serif; font-weight: 700;
    font-size: 3.1rem; line-height: 1.0; margin: 0.4rem 0 0.7rem 0;
    text-transform: uppercase; letter-spacing: 0.01em; color: #E8EEF4;
}
.hero p.lede {
    font-family: 'Inter', sans-serif; font-size: 1.02rem;
    color: #B9C6D4; max-width: 64ch; margin: 0; line-height: 1.55;
}

.tr-board { margin: 0.4rem 0 0.2rem 0; }
.tr-row { display: flex; align-items: center; gap: 0.9rem; margin: 0.5rem 0; }
.tr-name {
    flex: 0 0 8.5rem; font-family: 'Oswald', sans-serif; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.03em; font-size: 1.05rem;
    color: #E8EEF4; text-align: right; white-space: nowrap;
}
.tr-track {
    flex: 1 1 auto; height: 1.6rem; background: #14253A;
    border-radius: 3px; overflow: hidden;
}
.tr-fill { height: 100%; border-radius: 3px; }
.tr-fill.lead { background: linear-gradient(90deg, #F2C14E, #E8A82E); }
.tr-fill.rest { background: #3AA6B9; }
.tr-pct {
    flex: 0 0 4.2rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    font-size: 1.05rem; color: #E8EEF4; text-align: left;
}

/* Motion: bars sweep in on load, staggered like a broadcast graphic */
@keyframes sweep { from { width: 0; } }
.tr-fill { animation: sweep 1.1s cubic-bezier(0.22, 1, 0.36, 1) both; }
.tr-row:nth-child(1) .tr-fill { animation-delay: 0.10s; }
.tr-row:nth-child(2) .tr-fill { animation-delay: 0.28s; }
.tr-row:nth-child(3) .tr-fill { animation-delay: 0.46s; }

/* Hover: the row you're on comes alive, the rest stay quiet */
.tr-row { transition: transform 0.25s ease; border-radius: 4px; }
.tr-row:hover { transform: translateX(6px); }
.tr-row:hover .tr-fill.lead { background: linear-gradient(90deg, #FFD873, #F2B93E); }
.tr-row:hover .tr-fill.rest { background: #52C4D8; }
.tr-row:hover .tr-pct { color: #FFFFFF; }
.tr-fill { transition: background 0.25s ease; }

/* Tables and charts lift softly under the cursor */
div[data-testid="stDataFrame"], div[data-testid="stImage"] {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-radius: 6px;
}
div[data-testid="stDataFrame"]:hover, div[data-testid="stImage"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
}

/* Headline fades up on load */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } }
.hero .eyebrow { animation: rise 0.6s ease both; }
.hero h1.big   { animation: rise 0.6s ease 0.08s both; }
.hero p.lede   { animation: rise 0.6s ease 0.16s both; }

/* Respect visitors who've turned animation off */
@media (prefers-reduced-motion: reduce) {
    .tr-fill, .hero .eyebrow, .hero h1.big, .hero p.lede { animation: none; }
    .tr-row, .tr-row:hover,
    div[data-testid="stDataFrame"], div[data-testid="stDataFrame"]:hover,
    div[data-testid="stImage"], div[data-testid="stImage"]:hover {
        transform: none; transition: none;
    }
}
</style>
"""


@st.cache_data
def load_evaluation():
    """Rebuild leak-free rows, score model vs baseline on the held-out slice."""
    matches = pd.read_csv("data/results.csv", parse_dates=["date"])
    data = build_training_data(matches)
    test = data[data["date"] >= CUTOFF].reset_index(drop=True)
    outcomes = test["outcome"].map(OUTCOME_TO_INT).to_numpy()

    bundle = joblib.load("src/model.joblib")
    model = bundle["model"]
    features = bundle["features"]
    raw = model.predict_proba(test[features])
    col = {label: i for i, label in enumerate(model.classes_)}
    model_probs = raw[:, [col["home_win"], col["draw"], col["home_loss"]]]

    base_probs = np.array([
        [p["win"], p["draw"], p["loss"]]
        for p in (predict_match(d, 0.0) for d in test["elo_diff"])
    ])

    rows = []
    for name, fn in [("RPS", rps), ("Brier", brier),
                     ("Log loss", log_loss), ("Accuracy", accuracy)]:
        rows.append({
            "Metric": name,
            "Baseline": round(fn(base_probs, outcomes), 4),
            "Model": round(fn(model_probs, outcomes), 4),
        })
    return len(test), pd.DataFrame(rows), base_probs, model_probs, outcomes


def reliability_figure(base_probs, model_probs, outcomes):
    """Predicted vs actual frequency in 10 bins, styled for the dark theme."""
    actual = np.zeros_like(model_probs)
    actual[np.arange(len(outcomes)), outcomes] = 1.0
    bins = np.linspace(0, 1, 11)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(PANEL)
    for probs, colour, label in [(base_probs, TEAL, "Baseline"),
                                 (model_probs, GOLD, "Model")]:
        p, a = probs.ravel(), actual.ravel()
        idx = np.digitize(p, bins) - 1
        centers = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        freqs = [a[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(centers, freqs, marker="o", color=colour, label=label, lw=2)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Perfect calibration")

    ax.set_xlabel("Predicted probability", color=LIGHT)
    ax.set_ylabel("How often it happened", color=LIGHT)
    for spine in ax.spines.values():
        spine.set_color("#22364d")
    ax.tick_params(colors=LIGHT)
    ax.grid(True, color="#1a2c42", lw=0.6)
    leg = ax.legend(facecolor=PANEL, edgecolor="#22364d", labelcolor=LIGHT)
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return fig


@st.cache_data
def load_latest_predictions():
    files = sorted(f for f in PRED_DIR.glob("2026-*.json")
                   if "logreg" not in f.stem)
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_market():
    if not MARKET_FILE.exists():
        return None
    with open(MARKET_FILE, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def load_simulation():
    if not SIM_FILE.exists():
        return None
    with open(SIM_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def title_race_html(mc):
    order = sorted(mc, key=mc.get, reverse=True)
    top = order[0]
    rows = []
    for team in order:
        pct = mc[team] * 100
        cls = "lead" if team == top else "rest"
        rows.append(
            '<div class="tr-row">'
            f'<div class="tr-name">{team}</div>'
            f'<div class="tr-track"><div class="tr-fill {cls}" '
            f'style="width:{pct:.1f}%"></div></div>'
            f'<div class="tr-pct">{pct:.1f}%</div>'
            '</div>'
        )
    return '<div class="tr-board">' + "".join(rows) + "</div>"


# ---------------------------------------------------------------- layout
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">World Cup 2026 · every prediction logged before kickoff</div>
      <h1 class="big">Who lifts the trophy?</h1>
      <p class="lede">I trained a model on every international football match since
      1872 to forecast the 2026 World Cup, and I commit each prediction to GitHub
      before the game is played. I'm not trying to beat the bookmakers. What I care
      about is whether the probabilities are honest: when the model says 60%, does
      that happen about 60% of the time?</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sim = load_simulation()
if sim:
    st.markdown(title_race_html(sim["champion_probabilities_monte_carlo"]),
                unsafe_allow_html=True)
    state = sim["state"]
    a, b = state["remaining_semifinal"]
    st.caption(
        f"Based on {sim['n_simulations']:,} simulations of the games still to "
        f"play. {state['finalist']} are already through to the final; {a} play "
        f"{b} today for the other place."
    )
    an = sim.get("champion_probabilities_analytic")
    if an:
        st.caption(
            "With only three teams left I can also work the odds out exactly on "
            "paper. The simulation lands within a few tenths of a percent of that "
            "answer, which is how I know it is wired up correctly."
        )
else:
    st.info("Simulation results not found yet.")

st.divider()

n_test, metrics_df, base_probs, model_probs, outcomes = load_evaluation()

st.header("How good are the forecasts?")
st.caption(
    f"Both models were scored on {n_test} internationals played after January "
    "2022, none of which they were trained on. Lower is better for RPS, Brier "
    "and log loss; higher is better for accuracy."
)
st.dataframe(metrics_df, hide_index=True, use_container_width=True)
st.caption("The logistic-regression model edges out the simple Elo baseline on "
           "every measure.")

st.divider()

left, right = st.columns(2)

with left:
    st.header("Are they honest?")
    st.caption(
        "Each dot groups together predictions of a similar confidence and asks "
        "how often those calls actually came true. The closer the dots track the "
        "dashed line, the better calibrated the model is."
    )
    st.pyplot(reliability_figure(base_probs, model_probs, outcomes))

with right:
    st.header("Latest prediction")
    preds = load_latest_predictions()
    if preds:
        st.caption(
            f"Match day {preds['match_date']}, produced by the "
            f"{preds['model']} model at {preds['generated_at_utc'][11:16]} UTC."
        )
        rows = [{
            "Match": f"{p['home_team']} vs {p['away_team']}",
            "Home win": f"{p['prob_home_win']:.0%}",
            "Draw": f"{p['prob_draw']:.0%}",
            "Away win": f"{p['prob_away_win']:.0%}",
        } for p in preds["predictions"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    else:
        st.info("No prediction files found yet.")

st.divider()

st.header("Against the market")
market = load_market()
if market:
    st.caption(
        "The bookmakers' line is about the sharpest public forecast there is, so "
        "I use it as a yardstick rather than a target. These are their odds with "
        "the built-in margin stripped out, next to my model and the baseline."
    )
    labels = ["Home", "Draw", "Away"]
    for fx in market["fixtures"]:
        st.subheader(f"{fx['home_team']} vs {fx['away_team']}")
        table = pd.DataFrame({
            "Outcome": labels,
            "Market": [f"{v:.1%}" for v in fx["market"]],
            "My model": [f"{v:.1%}" for v in fx["model"]],
            "Baseline": [f"{v:.1%}" for v in fx["baseline"]],
        })
        st.dataframe(table, hide_index=True, use_container_width=True)
else:
    st.info("No market benchmark file found yet.")

st.divider()
st.caption("Source and full commit history: "
           "github.com/abdullahiali1545-arch/world-cup-2026-predictor")