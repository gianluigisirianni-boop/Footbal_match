"""
Football Match Analytics 2026 - Dashboard interattiva.

Legge data/dati.db (tabella "partite", creata da import_to_sqlite.py)
e mostra KPI e grafici con filtri a cascata stagione -> lega.

Uso:  streamlit run dashboard.py
"""

import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Match Dashboard", page_icon="⚽", layout="wide")

# Connessione al database
conn = sqlite3.connect("data/dati.db")
df = pd.read_sql("SELECT * FROM partite", conn)
conn.close()

st.title("Football Match 2026 — Dashboard")

# Filtri a cascata nella sidebar: stagione -> lega
stagioni = sorted(df["season"].unique())
scelta_stagione = st.sidebar.selectbox("Seleziona una stagione:", stagioni)
df_stagione = df[df["season"] == scelta_stagione]

leghe = sorted(df_stagione["league"].unique())
scelta_lega = st.sidebar.selectbox("Seleziona una lega:", leghe)
df_filtrato = df_stagione[df_stagione["league"] == scelta_lega]

# KPI (calcolati sul sottoinsieme filtrato)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Partite", len(df_filtrato))
col2.metric("Media gol a partita", round(df_filtrato["total_goals"].mean(), 2))
col3.metric("% vittorie in casa", round((df_filtrato["result"] == "home_win").mean() * 100, 2))
col4.metric("Scarto medio (gol)", round(df_filtrato["scarto"].mean(), 2))

# Grafico: esiti (casa / pareggio / trasferta)
st.subheader("Distribuzione esiti (stagione e lega selezionate)")
conteggi = df_filtrato["result"].value_counts()
st.bar_chart(conteggi)

st.caption(f"KPI calcolati su {len(df_filtrato)} partite — stagione {scelta_stagione}, lega {scelta_lega}")
