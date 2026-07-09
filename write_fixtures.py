import json

fixtures = {
    "fixtures": [
        {"date": "2026-07-09", "home_team": "France",    "away_team": "Morocco"},
        {"date": "2026-07-10", "home_team": "Spain",     "away_team": "Belgium"},
        {"date": "2026-07-11", "home_team": "Norway",    "away_team": "England"},
        {"date": "2026-07-11", "home_team": "Argentina", "away_team": "Switzerland"},
    ]
}

with open("fixtures.json", "w", encoding="utf-8") as f:
    json.dump(fixtures, f, indent=2, ensure_ascii=False)

print("Wrote fixtures.json with", len(fixtures["fixtures"]), "fixtures")