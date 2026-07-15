![CI](https://github.com/abdullahiali1545-arch/world-cup-2026-predictor/actions/workflows/ci.yml/badge.svg)

# World Cup 2026 Live Match Predictor

**Live app:** https://world-cup-2026-predictor-ypdwjsybuynesbypkmc5x5.streamlit.app

This project forecasts win/draw/loss probabilities for World Cup 2026 matches
and publishes every prediction to this repo before kickoff. The commits are
made by a scheduled GitHub Actions bot rather than by me, so the timestamps
in [`predictions/`](predictions/) can be checked and can't be backdated.

To clarify beforehand this project is not trying to beat the bookmakers.
What I wanted to test is whether the probabilities are honest. If the model
says 60%, that outcome should happen roughly 60% of the time.

Pick any two national teams on the live app and it predicts the match with both the trained model and the Elo baseline, side by side.

## Results

Both models were trained on matches before 2022, then scored on the 4,588
internationals played since then. None of the test matches were seen in
training.

| Metric   | Elo baseline | Logistic regression |
|----------|-------------:|--------------------:|
| RPS      |       0.1795 |          **0.1733** |
| Brier    |       0.5321 |          **0.5196** |
| Log loss |       0.9051 |          **0.8826** |
| Accuracy |       59.22% |          **59.85%** |

The regression wins on every metric, but the margins are small. Most of the
signal is already in the Elo ratings and the regression only squeezes a bit
more out of them. I believe that this is a fair reading of the table.

I used RPS (ranked probability score) as the headline metric because it
respects the order of the outcomes. Calling a home win when the game ends in
a draw is a smaller mistake than calling a home win when the away side wins.
Brier and log loss score those two mistakes the same.

The calibration curve is at [`results/reliability.png`](results/reliability.png)
and on the live app.

### Against the market

For the knockout rounds I also compared against bookmaker odds with the
margin stripped out (see [`results/market_benchmark.json`](results/market_benchmark.json)).
In the semifinals my model disagreed with the market on both matches. The
app shows the numbers side by side so you can judge for yourself.

## How it works

1. **Elo ratings.** Around 49,000 internationals going back to 1872 are
   replayed in date order, updating each team's rating after every match.
2. **Baseline.** The rating difference is turned directly into three
   probabilities, with the draw rate calibrated to how often draws actually
   happen in the data.
3. **Model.** A multinomial logistic regression on two features: the Elo
   difference and whether the venue is neutral. The train/test split is by
   date, not random. A random split would let the model use future matches
   to predict past ones, which inflates the scores and is impossible in real
   use.
4. **Tournament simulation.** The remaining bracket gets played out 50,000
   times to estimate each team's chance of lifting the trophy. Knockout
   games can't end in a draw, so a drawn result is settled as a coin flip
   for extra time and penalties. With three teams left the answer can also
   be computed exactly by hand, and the simulation lands within a few tenths
   of a percent of it. That agreement is my check that the simulation is
   wired up correctly.
5. **Automation.** A scheduled workflow generates and commits the day's
   predictions at 08:00 UTC, and the test suite (23 tests) runs on every
   push.

## Running it

```
pip install -r requirements.txt
python run_evaluation.py     # model vs baseline table, plus the reliability diagram
python run_simulation.py     # tournament simulation
streamlit run app.py         # the app, locally
```

The historical data is the Kaggle dataset "International football results
from 1872 to present", included at `data/results.csv`.

## Limitations

- Only two features. No squad strength, no injuries, no recent form. I kept
  it minimal on purpose because the point of the project is the evaluation
  pipeline rather than feature engineering, but it does cap how good the
  forecasts can be.
- The Elo K factor and the draw model were tuned by hand rather than fitted.
- The bookmaker comparison only covers the knockout matches I collected odds
  for, which is far too small a sample to score properly.
- International teams play few matches, so ratings move slowly and can lag
  behind real changes in team strength.

If I take this further the next steps would be recent form as a feature,
weighting old matches less, a Dixon-Coles scoreline model, and swapping the
regression for gradient boosting.
