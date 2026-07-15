"""Streamlit app: World Cup 2026 match predictor.

Four pages: an interactive head-to-head match predictor, the trophy race,
the model-vs-baseline scoreboard, and the bookmaker benchmark.

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
from src.elo import compute_ratings, predict_match
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


# ---------------------------------------------------------------- data
@st.cache_data
def load_matches():
    return pd.read_csv("data/results.csv", parse_dates=["date"])


@st.cache_data
def load_ratings():
    """Current Elo rating for every national team."""
    return compute_ratings(load_matches().sort_values("date"))


@st.cache_resource
def load_model_bundle():
    return joblib.load("src/model.joblib")


@st.cache_data
def load_evaluation():
    """Rebuild leak-free rows, score model vs baseline on the held-out slice."""
    data = build_training_data(load_matches())
    test = data[data["date"] >= CUTOFF].reset_index(drop=True)
    outcomes = test["outcome"].map(OUTCOME_TO_INT).to_numpy()

    bundle = load_model_bundle()
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


def model_predict(elo_diff: float, neutral: bool):
    """Three probabilities [team A win, draw, team B win] from the model."""
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]
    row = pd.DataFrame([{"elo_diff": elo_diff, "neutral": int(neutral)}])
    raw = model.predict_proba(row[features])[0]
    col = {label: i for i, label in enumerate(model.classes_)}
    return [raw[col["home_win"]], raw[col["draw"]], raw[col["home_loss"]]]


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


# ---------------------------------------------------------------- widgets
def bar_board(rows, animate=True):
    """rows: list of (label, fraction, is_lead). Renders themed bars."""
    parts = []
    for label, frac, lead in rows:
        cls = "lead" if lead else "rest"
        parts.append(
            '<div class="tr-row">'
            f'<div class="tr-name">{label}</div>'
            f'<div class="tr-track"><div class="tr-fill {cls}" '
            f'style="width:{frac * 100:.1f}%"></div></div>'
            f'<div class="tr-pct">{frac * 100:.1f}%</div>'
            '</div>'
        )
    return '<div class="tr-board">' + "".join(parts) + "</div>"


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


# ---------------------------------------------------------------- pages
def page_match_predictor():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Pick any two national teams</div>
          <h1 class="big">Match predictor</h1>
          <p class="lede">The same model that publishes the daily forecasts,
          pointed at any fixture you like. It knows two things about each
          match: the gap in Elo ratings and whether the venue is neutral.
          The simple Elo baseline is shown alongside so you can see where
          the trained model disagrees with it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ratings = load_ratings()
    teams = sorted(ratings)

    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        team_a = st.selectbox("Team A", teams,
                              index=teams.index("England") if "England" in teams else 0)
    with c2:
        st.markdown(
            "<div style='text-align:center; font-family:Oswald; font-size:1.6rem;"
            " padding-top:1.7rem;'>VS</div>", unsafe_allow_html=True)
    with c3:
        team_b = st.selectbox("Team B", teams,
                              index=teams.index("Brazil") if "Brazil" in teams else 1)

    neutral = st.toggle("Neutral venue", value=True,
                        help="Off means Team A is at home. This is one of the "
                             "model's two features, so it genuinely moves the "
                             "numbers. The Elo baseline has no venue term.")

    if team_a == team_b:
        st.warning("Pick two different teams.")
        return

    if st.button("⚽ Run prediction", use_container_width=True, type="primary"):
        ra, rb = ratings[team_a], ratings[team_b]
        diff = ra - rb
        m = model_predict(diff, neutral)
        b = predict_match(ra, rb)
        base = [b["win"], b["draw"], b["loss"]]

        st.caption(
            f"Current Elo ratings: {team_a} {ra:.0f}, {team_b} {rb:.0f} "
            f"(difference {diff:+.0f}"
            + ("" if neutral else f", {team_a} at home") + ")."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Model")
            st.markdown(bar_board([
                (f"{team_a}", m[0], m[0] == max(m)),
                ("Draw", m[1], m[1] == max(m)),
                (f"{team_b}", m[2], m[2] == max(m)),
            ]), unsafe_allow_html=True)
        with right:
            st.subheader("Elo baseline")
            st.markdown(bar_board([
                (f"{team_a}", base[0], base[0] == max(base)),
                ("Draw", base[1], base[1] == max(base)),
                (f"{team_b}", base[2], base[2] == max(base)),
            ]), unsafe_allow_html=True)

        st.caption(
            "Ratings come from replaying every international since 1872, so "
            "long-retired sides still have a number. Treat historical or "
            "rarely-active teams with a pinch of salt."
        )

    preds = load_latest_predictions()
    if preds:
        st.divider()
        st.subheader("The bot's latest pre-kickoff call")
        st.caption(
            f"Match day {preds['match_date']}, committed by the scheduled "
            f"workflow at {preds['generated_at_utc'][11:16]} UTC, before "
            "kickoff, using the "
            f"{preds['model']} model."
        )
        rows = [{
            "Match": f"{p['home_team']} vs {p['away_team']}",
            "Home win": f"{p['prob_home_win']:.0%}",
            "Draw": f"{p['prob_draw']:.0%}",
            "Away win": f"{p['prob_away_win']:.0%}",
        } for p in preds["predictions"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)


def page_trophy_race():
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
    if not sim:
        st.info("Simulation results not found yet.")
        return

    mc = sim["champion_probabilities_monte_carlo"]
    order = sorted(mc, key=mc.get, reverse=True)
    top = order[0]
    st.markdown(bar_board([(t, mc[t], t == top) for t in order]),
                unsafe_allow_html=True)

    state = sim["state"]
    a, b = state["remaining_semifinal"]
    st.caption(
        f"Based on {sim['n_simulations']:,} simulations of the games still to "
        f"play. {state['finalist']} are already through to the final; {a} play "
        f"{b} for the other place."
    )
    if sim.get("champion_probabilities_analytic"):
        st.caption(
            "With only three teams left I can also work the odds out exactly on "
            "paper. The simulation lands within a few tenths of a percent of that "
            "answer, which is how I know it is wired up correctly."
        )


def page_scoreboard():
    n_test, metrics_df, base_probs, model_probs, outcomes = load_evaluation()

    st.header("How good are the forecasts?")
    st.caption(
        f"Both models were scored on {n_test} internationals played after "
        "January 2022, none of which they were trained on. Lower is better for "
        "RPS, Brier and log loss; higher is better for accuracy."
    )
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    st.caption("The logistic-regression model edges out the simple Elo "
               "baseline on every measure.")

    st.divider()
    st.header("Are they honest?")
    st.caption(
        "Each dot groups together predictions of a similar confidence and asks "
        "how often those calls actually came true. The closer the dots track "
        "the dashed line, the better calibrated the model is."
    )
    left, _ = st.columns([1, 1])
    with left:
        st.pyplot(reliability_figure(base_probs, model_probs, outcomes))


def page_market():
    st.header("Against the market")
    market = load_market()
    if not market:
        st.info("No market benchmark file found yet.")
        return
    st.caption(
        "The bookmakers' line is about the sharpest public forecast there is, "
        "so I use it as a yardstick rather than a target. These are their odds "
        "with the built-in margin stripped out, next to my model and the "
        "baseline."
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


def page_teams():
    st.header("Every team in the tournament")
    st.caption(
        "All 48 sides that played at World Cup 2026, ranked by their current "
        "Elo rating, with their record in this tournament. Ratings include "
        "every international each team has ever played, so a side can rank "
        "above teams that went further in the bracket."
    )

    matches = load_matches()
    ratings = load_ratings()
    wc = matches[(matches["date"] >= "2026-06-01")
                 & (matches["tournament"] == "FIFA World Cup")]
    teams = sorted(set(wc["home_team"]) | set(wc["away_team"]))

    alive = set()
    sim = load_simulation()
    if sim:
        alive = {sim["state"]["finalist"], *sim["state"]["remaining_semifinal"]}

    rows = []
    for t in teams:
        home = wc[wc["home_team"] == t]
        away = wc[wc["away_team"] == t]
        w = int((home["home_score"] > home["away_score"]).sum()
                + (away["away_score"] > away["home_score"]).sum())
        d = int((home["home_score"] == home["away_score"]).sum()
                + (away["away_score"] == away["home_score"]).sum())
        played = len(home) + len(away)
        rows.append({
            "Team": ("🏆 " if t in alive else "") + t,
            "Elo rating": round(ratings.get(t, 0)),
            "Played": played,
            "Won": w,
            "Drawn": d,
            "Lost": played - w - d,
        })

    table = pd.DataFrame(rows).sort_values("Elo rating", ascending=False)
    table.insert(0, "Rank", range(1, len(table) + 1))
    st.dataframe(table, hide_index=True, use_container_width=True,
                 height=min(38 * len(table) + 40, 1200))
    st.caption("🏆 = still in the tournament. Draws include knockout games "
               "decided in extra time or on penalties, since the dataset "
               "records the 90-minute score.")


# ---------------------------------------------------------------- shell
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚽ World Cup 2026")
    page = st.radio("Pages", ["Match predictor", "Trophy race", "Teams",
                              "Scoreboard", "vs the market"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("Every prediction is committed to GitHub before kickoff by a "
               "scheduled bot, so the timestamps can't be faked.")
    st.caption("[Source & commit history](https://github.com/"
               "abdullahiali1545-arch/world-cup-2026-predictor)")

if page == "Match predictor":
    page_match_predictor()
elif page == "Trophy race":
    page_trophy_race()
elif page == "Teams":
    page_teams()
elif page == "Scoreboard":
    page_scoreboard()
else:
    page_market()