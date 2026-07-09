import json

fixtures = {
    "fixtures": [
        {"date": "2026-07-05", "home_team": "Brazil", "away_team": "Norway"},
        {"date": "2026-07-05", "home_team": "Mexico", "away_team": "England"},
        {"date": "2026-07-06", "home_team": "Spain", "away_team": "Portugal"},
        {"date": "2026-07-06", "home_team": "Belgium", "away_team": "United States"},
        {"date": "2026-07-07", "home_team": "Egypt", "away_team": "Argentina"},
        {"date": "2026-07-07", "home_team": "Switzerland", "away_team": "Colombia"},
    ]
}

with open("fixtures.json", "w", encoding="utf-8") as f:
    json.dump(fixtures, f, indent=2, ensure_ascii=False)
print("Wrote fixtures.json")