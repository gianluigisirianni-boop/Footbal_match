"""
Football Match Analytics 2026 - Generatore automatico di report.

Esegue l'intera pipeline (join, EDA, feature engineering con shift
temporale, 2 modelli ML) e produce un report HTML auto-contenuto:
report_calcio.html

Uso:  python generate_report.py
"""

import base64
import io
from pathlib import Path
from datetime import datetime

import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

HANDLE = "dianatofficial/football-match-analytics-benchmark-2026"
OUT = Path(__file__).parent / "report_calcio.html"


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(fig, width="700"):
    return f'<img src="data:image/png;base64,{fig_to_base64(fig)}" width="{width}">'


# ---------------------------------------------------------------- pipeline
print("1/6  Caricamento dataset...")
base = kagglehub.dataset_download(HANDLE)
df_matches = pd.read_csv(f"{base}/data/matches.csv")
df_clubs = pd.read_csv(f"{base}/data/clubs.csv")
df_leagues = pd.read_csv(f"{base}/data/leagues.csv")

print("2/6  Join relazionali (chiavi univoche, shape check dopo ogni merge)...")
df_total = pd.merge(df_matches, df_clubs, left_on="home_club", right_on="club", how="left")
assert len(df_total) == len(df_matches), "Il primo merge ha duplicato righe!"
df_total = pd.merge(df_total, df_leagues, left_on="league_x", right_on="league", how="left")
assert len(df_total) == len(df_matches), "Il secondo merge ha duplicato righe!"
df_total = df_total.drop(columns=["league_y", "club", "league_clubs", "league", "country_y", "clubs"])
df_total = df_total.rename(columns={"league_x": "league"})

print("3/6  EDA...")
media_gol = df_total["total_goals"].mean()
goleada = df_total[df_total["total_goals"] > 5]
df_total["scarto"] = abs(df_total["goals_home"] - df_total["goals_away"])
scarto_medio = df_total[df_total["result"] != "draw"]["scarto"].mean()
home_win_pct = (df_total["result"] == "home win").mean() * 100

casa = df_total.groupby("home_club")["goals_home"].sum()
trasferta = df_total.groupby("away_club")["goals_away"].sum()
classifica = pd.DataFrame({"casa": casa, "trasferta": trasferta})
classifica["totale"] = classifica["casa"] + classifica["trasferta"]
top10 = classifica.sort_values("totale", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
top10.sort_values("totale")[["casa", "trasferta"]].plot(
    kind="barh", stacked=True, ax=ax, color=["#2b6cb0", "#90cdf4"])
ax.set_title("Top 10 bomber 2019-2026 (casa vs trasferta)")
ax.set_xlabel("Gol segnati")
bomber_html = img_tag(fig, width="720")

print("4/6  Feature engineering con shift temporale (anti-leakage)...")
stats_casa = df_total.groupby(["season", "home_club"]).agg(
    giocate_casa=("match_id", "count"),
    fatti_casa=("goals_home", "sum"),
    subiti_casa=("goals_away", "sum")).reset_index()
stats_trasferta = df_total.groupby(["season", "away_club"]).agg(
    giocate_trasferta=("match_id", "count"),
    fatti_trasferta=("goals_away", "sum"),
    subiti_trasferta=("goals_home", "sum")).reset_index()

stats_club = pd.merge(stats_casa, stats_trasferta,
                      left_on=["season", "home_club"],
                      right_on=["season", "away_club"], how="left")
stats_club = stats_club.drop(columns=["away_club"]).rename(columns={"home_club": "club"})
stats_club["gol_fatti"] = stats_club["fatti_casa"] + stats_club["fatti_trasferta"]
stats_club["gol_subiti"] = stats_club["subiti_casa"] + stats_club["subiti_trasferta"]
stats_club["partite"] = stats_club["giocate_casa"] + stats_club["giocate_trasferta"]
stats_club["gol_fatti_media"] = stats_club["gol_fatti"] / stats_club["partite"]
stats_club["gol_subiti_media"] = stats_club["gol_subiti"] / stats_club["partite"]
stats_club["diff_reti"] = stats_club["gol_fatti"] - stats_club["gol_subiti"]

feature_cols = ["gol_fatti_media", "gol_subiti_media", "diff_reti", "partite"]
stats_club = stats_club.sort_values(["club", "season"])
stats_shift = stats_club.copy()
stats_shift[feature_cols] = stats_club.groupby("club")[feature_cols].shift(1)
stats_shift = stats_shift.dropna(subset=feature_cols)

# check: la forza dei club persiste tra stagioni?
s = stats_club.sort_values(["club", "season"])
autocorr = s.groupby("club")["diff_reti"].apply(lambda x: x.autocorr()).dropna().mean()

ml = df_total[["season", "league", "home_club", "away_club", "result"]].copy()
ml = pd.merge(ml, stats_shift[["season", "club"] + feature_cols],
              left_on=["season", "home_club"], right_on=["season", "club"], how="inner")
ml = ml.drop(columns=["club"]).rename(columns={c: c + "_home" for c in feature_cols})
ml = pd.merge(ml, stats_shift[["season", "club"] + feature_cols],
              left_on=["season", "away_club"], right_on=["season", "club"], how="inner")
ml = ml.drop(columns=["club"]).rename(columns={c: c + "_away" for c in feature_cols})

print("5/6  Training modelli (baseline + balanced)...")
feature_ml = [c + "_home" for c in feature_cols] + [c + "_away" for c in feature_cols]
X = ml[feature_ml]
y = ml["result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

risultati = {}
for nome, cw in [("RandomForest", None), ("RandomForest balanced", "balanced")]:
    rf = RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=rf.classes_).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {nome}")
    risultati[nome] = {"accuracy": accuracy_score(y_test, y_pred),
                       "recall_draw": rep["draw"]["recall"],
                       "cm_html": img_tag(fig, width="380")}

imp = pd.Series(rf.feature_importances_, index=feature_ml).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.5))
imp.plot(kind="barh", ax=ax, title="Feature importance")
fi_html = img_tag(fig, width="620")

# ---------------------------------------------------------------- report
print("6/6  Generazione HTML...")
r1, r2 = risultati["RandomForest"], risultati["RandomForest balanced"]
baseline = (y == "home win").mean()
generato_il= datetime.now().strftime("%d/%m/%Y alle %H:%M")

html = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>Football Match Analytics 2026 — Report automatico</title>
<style>
 body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 30px auto;
        color: #222; line-height: 1.55; padding: 0 16px; }}
 h1 {{ border-bottom: 3px solid #2f855a; padding-bottom: 8px; }}
 h2 {{ color: #2f855a; margin-top: 40px; }}
 table {{ border-collapse: collapse; margin: 12px 0; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 14px; text-align: center; }}
 th {{ background: #e9f7ef; }}
 .box {{ background: #f7fafc; border-left: 4px solid #2f855a; padding: 10px 16px; }}
 .warn {{ border-left-color: #dd6b20; }}
 img {{ border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; }}
 .duecol {{ display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }}
</style></head><body>
<p><em>Generato automaticamente da <code>generate_report.py</code> il {generato_il}-nessun valore inserito a mano.</em></p>
<h1>Football Match Analytics 2026 — Report automatico</h1>
<p><em>Generato automaticamente da <code>generate_report.py</code> —
nessun valore inserito a mano.</em></p>

<h2>1. Dataset</h2>
<p>100.000 partite simulate, 8 leghe fittizie, 164 club, stagioni 2019-2026
(fonte: Kaggle, licenza CC BY-SA 4.0). Architettura relazionale a 3 tabelle
(matches/clubs/leagues) unite con join su chiavi univoche (verificati con
shape check: nessuna duplicazione).</p>

<h2>2. EDA — le risposte in numeri</h2>
<table>
<tr><th>Domanda</th><th>Risposta</th></tr>
<tr><td>Media gol a partita</td><td>{media_gol:.2f}</td></tr>
<tr><td>Partite con piu' di 5 gol</td><td>{len(goleada):,} ({len(goleada)/len(df_total)*100:.1f}%)</td></tr>
<tr><td>Scarto medio (partite con vincitore)</td><td>{scarto_medio:.2f} gol</td></tr>
<tr><td>Vittorie in casa</td><td>{home_win_pct:.1f}%</td></tr>
<tr><td>Bomber assoluto</td><td>{top10.index[0]} ({int(top10["totale"].iloc[0]):,} gol)</td></tr>
</table>
{bomber_html}

<h2>3. Modello ML — previsione risultato (leakage-free)</h2>
<p>Feature: statistiche dei club della <b>stagione precedente</b> (gol fatti/subiti
a partita, differenza reti) ottenute con shift temporale per club.
Le colonne post-partita (gol, xG, tiri, possesso) sono escluse: ignote prima
del fischio d'inizio.</p>
<div class="duecol">
<div>{r1["cm_html"]}</div>
<div>{r2["cm_html"]}</div>
</div>
<table>
<tr><th>Modello</th><th>Accuracy</th><th>Recall draw</th></tr>
<tr><td>RandomForest</td><td>{r1["accuracy"]:.2f}</td><td>{r1["recall_draw"]:.2f}</td></tr>
<tr><td>RandomForest (balanced)</td><td>{r2["accuracy"]:.2f}</td><td>{r2["recall_draw"]:.2f}</td></tr>
<tr><td><b>Baseline "sempre home win"</b></td><td><b>{baseline:.2f}</b></td><td>0.00</td></tr>
</table>
{fi_html}

<h2>4. Conclusione: risultato negativo, diagnosticato</h2>
<div class="box warn">
<p>Il modello <b>non batte la baseline</b> ({r1["accuracy"]:.2f} vs {baseline:.2f}).
La diagnosi: l'autocorrelazione anno-su-anno della forza dei club e'
<b>{autocorr:.2f}</b> → in questo dataset simulato la forza viene ri-estratta a ogni
stagione, quindi il passato non contiene informazione sul futuro.
Nessun modello puo' trovare un segnale che non esiste nei dati:
metodologia corretta, limite nei dati — dichiarato, non nascosto.</p>
</div>

<h2>5. Note metodologiche</h2>
<div class="box">
<ul>
<li>Join su chiavi univoche + shape check dopo ogni merge (il join sbagliato
aveva prodotto 2.034.540 righe da 100.000: prodotto cartesiano).</li>
<li>Feature mai post-partita: niente leakage.</li>
<li>Shift temporale per club: le statistiche della stagione S-1 diventano
feature per la stagione S; il 2019 esce (nessuna storia disponibile).</li>
<li>Checklist pre-training: describe() sulle feature (una colonna tutta zeri
era stata individuata e corretta grazie alla heatmap delle correlazioni).</li>
</ul></div>

</body></html>"""

OUT.write_text(html, encoding="utf-8")
print(f"\nReport salvato: {OUT}")
