# Reproduction code

**Paper:** Monitoring a distributed photovoltaic fleet without weather data: forecasting limits, daily operating regimes and peer-referenced fault detection
**Author:** Hafiz Atif Ali (ORCID 0009-0001-4366-729X)

## Data
Download "Solar Energy Production" (dataset ytdn-2qsp) from the City of Calgary open-data portal as CSV and save it as `data.csv` in this folder:
https://data.calgary.ca/Environment/Solar-Energy-Production/ytdn-2qsp

Rated capacities and coordinates come from dataset tbsv-89ps and are stored in `rated_capacity.csv`.

## Environment
Python 3.11. Install the packages with `pip install -r requirements.txt`. Random seeds are fixed in every script.

## Run order
| Step | Script | Produces |
|---|---|---|
| 1 | `prep.py` | Regular hourly grid, clear-sky and solar position, site statistics |
| 2 | `eda.py`, `audit.py` | Fig. 1–2, missing-day audit (Table 1), capacity stability, spatial correlation (Fig. 5) |
| 3 | `forecast.py`, `cliper.py`, `lstm_es.py`, `lstm_seed.py` (SEED=1..4), `metrics.py` | Table 3; 5-seed LSTM |
| 4 | `fc2.py`, `da_issue.py`, `sp_diag.py` | Skill-vs-horizon curve, ridge baseline, fixed issue-time day-ahead, smart-persistence diagnosis |
| 5 | `shap_run.py` | Fig. 3 |
| 6 | `unsup.py`, `regimes.py`, `reg2.py` | Regimes (Fig. 4), silhouette in fingerprint space, dip test, gap statistic, seasonal-null synchrony |
| 7 | `anomaly3.py` (env: R, LOO, ZX, KMIN, EPS, DAMP, W, SUF) | Fault-injection benchmark (Table 4, Fig. 6), all sensitivity runs |
| 8 | `chronic.py`, `figs3.py` | Long-term peer ratio (Fig. 8), case studies (Fig. 7) |
