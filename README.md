# Football Match Analytics Benchmark 2026 ⚽

Analisi di 100.000 partite simulate (8 leghe fittizie, 164 club, stagioni 2019-2026).
Dataset Kaggle: `dianatofficial/football-match-analytics-benchmark-2026` (CC BY-SA 4.0).
Architettura relazionale a 3 tabelle: `matches` (100.000 righe), `clubs` (164), `leagues` (8).

## Domande di ricerca

1. Quale club ha segnato più gol in assoluto? Split casa/trasferta, classifica top 10
2. Media gol per partita
3. Quante partite con più di 5 gol, e quali
4. Scarto medio di gol nelle partite con un vincitore (esclusi i pareggi)
5. Si può prevedere il risultato (home/draw/away) da statistiche pre-partita?

## Risultati EDA

- Media gol a partita: ~2.8 (coerente con `avg_goals_per_match` delle leghe)
- Scarto medio con vincitore: **1.76 gol**
- Bomber assoluto: **Granton Rovers** (2.990 gol), seguito da Thames FC (2.941) —
  i top club segnano di più in casa (fattore campo presente anche nei dati simulati)

## Modello ML — previsione risultato

**Approccio leakage-free:** feature = statistiche dei club della **stagione precedente**
(gol fatti/subiti a partita, differenza reti), ottenute con groupby + `shift(1)`
temporale per club. Le colonne post-partita (gol, xG, tiri, possesso) sono escluse
perché ignote prima del fischio d'inizio.

| Modello | Accuracy | Baseline "sempre home win" |
|---|---|---|
| RandomForest | 0.38 | **0.465** |
| RandomForest (class_weight=balanced) | 0.34 (recall draw 0.35) | 0.465 |

### Conclusione: risultato negativo, diagnosticato

Il modello **non batte la baseline**. La diagnosi: l'autocorrelazione anno-su-anno
della forza dei club (`diff_reti`) è **−0.14** → in questo dataset simulato la forza
viene ri-estratta a ogni stagione, quindi il passato non contiene informazione sul
futuro. Nessun modello può trovare un segnale che non esiste nei dati.
Metodologia corretta, limite nei dati — dichiarato, non nascosto.

## Bug memorabili (lezione del progetto)

1. **Join su chiave non univoca** (`league` tra matches e clubs): prodotto
   cartesiano, 100.000 → 2.034.540 righe. Regola: la chiave deve essere unica
   almeno da un lato; shape check dopo OGNI merge.
2. **Colonna morta**: `diff_reti = gol_subiti - gol_subiti` (stessa colonna due
   volte) → tutta zeri, std=0. Scoperta grazie alla riga bianca nella heatmap
   delle correlazioni. Checklist: `X.describe()` prima di allenare.
3. **Celle merge ri-eseguite**: i merge non sono idempotenti — ri-eseguirli
   moltiplica le righe. Restart & Run All è il test di accettazione.
4. **`_x`/`_y` automatici di pandas**: segnale di colonne omonime dopo un merge;
   rinominare subito con suffissi parlanti (`_home`/`_away`).

## File

- `Football_match.ipynb` — notebook completo (eseguire con Restart & Run All)
