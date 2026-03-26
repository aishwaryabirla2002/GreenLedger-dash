# 🌿 Green Ledger Analytics Dashboard

A complete data-driven analytics dashboard for **Green Ledger** — a verified sustainability rewards platform for India.

## Features
- **Descriptive Analytics**: Demographics, distributions, Likert heatmaps
- **Diagnostic Analytics**: Adoption drivers, barrier decomposition, behavioral economics
- **Predictive Models**: Random Forest (Accuracy, Precision, Recall, F1, ROC-AUC), Linear Regression, Feature Importance
- **Association Rule Mining**: Support, Confidence, Lift with interactive filters
- **Customer Clustering**: K-Means (k=4) with elbow curve and radar profiles
- **Prescriptive Strategy**: Pricing tiers, acquisition channels, Pearson correlation matrix
- **Predict New Customer**: Live adoption probability scoring with gauge chart

## Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to GitHub (flat structure — no subfolders needed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point → Deploy

## Data
`data.csv` — 2,000 synthetic survey respondents anchored to:
- IAMAI-Kantar ICUBE 2024 (internet usage)
- UN World Population Prospects 2024 (age/gender)
- NSSO HCES 2022-23 (income distribution)

## Tech Stack
- Streamlit 1.35 · Plotly 5.18 · scikit-learn 1.3 · pandas · numpy · scipy
