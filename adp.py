import requests
import csv

def fetch_adp():
    """
    Fetches ADP data from the FantasyFootballCalculator API.
    Hardcoded settings:
        - scoring: ppr
        - teams: 12
        - year: 2024
    """
    url = "https://fantasyfootballcalculator.com/api/v1/adp/ppr"
    params = {
        "teams": 12,
        "year": 2023
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    players = data.get("players", [])
    return players

def write_csv(players, filename="adp_export.csv"):
    """Writes player ADP data to a CSV file."""
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["player_name", "adp"])
        for p in players:
            name = (
                p.get("name") or
                p.get("player_name") or
                (p.get("player", {}) or {}).get("name")
            )
            adp = p.get("adp")
            writer.writerow([name, adp])

def main():
    print("Fetching ADP data (PPR, 12-team, 2024 season)...")
    players = fetch_adp()
    print(f"Fetched {len(players)} players.")
    write_csv(players)
    print("CSV file 'adp_export.csv' created successfully.")

if __name__ == "__main__":
    main()
