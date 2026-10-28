"""
Football Match Analytics 2026 - Import dati in SQLite.

Replica la pipeline di caricamento/join di generate_report.py
(stessa sorgente di verita', assert anti-duplicazione inclusi)
e salva df_total in data/dati.db, tabella "partite",
per alimentare dashboard.py.

Uso:  python import_to_sqlite.py
"""

import sqlite3
from pathlib import Path

import kagglehub
import pandas as pd

HANDLE = "dianatofficial/football-match-analytics-benchmark-2026"

print("1/3  Caricamento dataset (kagglehub)...")
base = kagglehub.dataset_download(HANDLE)
df_matches = pd.read_csv(f"{base}/data/matches.csv")
df_clubs = pd.read_csv(f"{base}/data/clubs.csv")
df_leagues = pd.read_csv(f"{base}/data/leagues.csv")

print("2/3  Join relazionali (chiavi univoche, shape check dopo ogni merge)...")
df_total = pd.merge(df_matches, df_clubs, left_on="home_club", right_on="club", how="left")
assert len(df_total) == len(df_matches), "Il primo merge ha duplicato righe!"
df_total = pd.merge(df_total, df_leagues, left_on="league_x", right_on="league", how="left")
assert len(df_total) == len(df_matches), "Il secondo merge ha duplicato righe!"
df_total = df_total.drop(columns=["league_y", "club", "league_clubs", "league", "country_y", "clubs"])
df_total = df_total.rename(columns={"league_x": "league"})

# Scarto gol (come nel notebook): utile per i KPI della dashboard
df_total["scarto"] = abs(df_total["goals_home"] - df_total["goals_away"])

print("3/3  Scrittura in SQLite...")
BASE_DIR = Path(__file__).parent
(BASE_DIR / "data").mkdir(exist_ok=True)

conn = sqlite3.connect(BASE_DIR / "data" / "dati.db")
df_total.to_sql("partite", conn, if_exists="replace", index=False)
check = pd.read_sql("SELECT COUNT(*) AS righe FROM partite", conn)
conn.close()

print(check)
print("OK - data/dati.db aggiornato, tabella 'partite'")
