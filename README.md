# Multi-Horizon Forecasting of Urban Cycling Demand

**Author:** Daniela de Sousa Silva

## Description

Forecasting daily cyclist counts at Munich bike counter stations using statistical and machine learning methods. Compares SARIMAX with Fourier terms, Prophet, and XGBoost hybrid models across three stations with distinct demand profiles.

**Data:** München Open Data Portal — bike counter network, daily counts 2013–2026  

---

## Repository Structure

```
├── data/
│   ├── raw/                  # Raw counter CSVs (download separately)
│   └── processed/            # master_bike_data.csv (all stations, daily)
├── notebooks/                # Analysis pipeline (run in order, 01 → 14)
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_and_sarima.ipynb
│   ├── 04_sarimax_fourier.ipynb
│   ├── 05_sarimax_weather.ipynb
│   ├── 06_olympia_sarimax.ipynb
│   ├── 07_erhardt_sarimax.ipynb
│   ├── 08_hirsch_prophet.ipynb
│   ├── 09_olympia_prophet.ipynb
│   ├── 10_erhardt_prophet.ipynb
│   ├── 11_xgboost.ipynb
│   ├── 12_sarimax_hybrid.ipynb
│   ├── 13_prophet_hybrid.ipynb
│   └── 14_additional_analysis.ipynb
├── src/
│   └── data_loader.py        # Shared data loading utilities
├── results/
│   ├── figures/              # PNG figure outputs
│   ├── models/               # Fitted model objects (.pkl)
│   ├── predictions/          # Forecast CSVs per station
│   └── tables/               # Performance metrics CSVs
└── outputs/
    └── figures/              # Thesis-ready figure exports
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn statsmodels prophet xgboost scikit-learn scipy holidays jupyter
```

## Notebooks

Each notebook is self-contained after `01_data_loading.ipynb`. Outputs (models, predictions, tables, figures) are saved to `results/` and read by later notebooks.

| Notebook | Content |
|----------|---------|
| 01 | Data loading and preprocessing |
| 02 | Exploratory data analysis |
| 03 | Baseline models and SARIMA |
| 04 | SARIMAX with Fourier terms |
| 05 | Weather regressor integration (Hirsch) |
| 06 | Cross-station validation (Olympia) |
| 07 | Structural break analysis (Erhardt) |
| 08–10 | Prophet models per station |
| 11 | Standalone XGBoost |
| 12 | SARIMAX + XGBoost hybrid |
| 13 | Prophet + XGBoost hybrid |
| 14 | Additional analysis and significance tests |

## Stations

| Station | Type |
|---------|------|
| Hirsch | Commuter |
| Olympia | Recreational |
| Erhardt | Mixed / structural break |

---

## Dataset Source

[München Open Data Portal](https://opendata.muenchen.de/dataset/daten-der-raddauerzaehlstellen-muenchen-jahreszahlen) — bike counter network, daily counts 2013–2026  

---

## License

This project is licensed under the MIT License. See the LICENSE.md file for details.
