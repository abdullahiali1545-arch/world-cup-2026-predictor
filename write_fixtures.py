import json
fixtures = {
    "fixtures": [
        {"date": "2026-07-14", "home_team": "France",  "away_team": "Spain"},
        {"date": "2026-07-15", "home_team": "England", "away_team": "Argentina"},
    ]
}
with open("fixtures.json", "w", encoding="utf-8") as f:
    json.dump(fixtures, f, indent=2, ensure_ascii=False)
print("Wrote fixtures.json with", len(fixtures["fixtures"]), "fixtures")