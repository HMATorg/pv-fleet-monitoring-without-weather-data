# Publishing this code

The manuscript's Data availability section points to:

    https://github.com/HMATorg/pv-fleet-monitoring-without-weather-data

Create that repository and push this folder before you click **Submit** in Editorial Manager.

## Option A — git (from a terminal, in the folder that holds `code/`)

1. On github.com, signed in as **HMATorg**: **New repository** →
   Owner `HMATorg`, name `pv-fleet-monitoring-without-weather-data`, **Public**, do **not** add a README or .gitignore → **Create repository**.
2. Then:

```bash
git init
git add .
git commit -m "Reproduction code for PV fleet monitoring without weather data"
git branch -M main
git remote add origin https://github.com/HMATorg/pv-fleet-monitoring-without-weather-data.git
git push -u origin main
```

## Option B — no git

1. Create the repository exactly as in step 1 above.
2. On the empty repository page click **uploading an existing file**.
3. Drag in `code/` (all 24 files) and `CITATION.cff`, then **Commit changes**.

## Check

Open https://github.com/HMATorg/pv-fleet-monitoring-without-weather-data in a private window. If `code/README.md` and `code/requirements.txt` are visible without signing in, the link in the paper resolves.

## Contents

- `code/` — 24 files: the full pipeline (prep → features → forecasting → regimes → anomaly detection → figures), `requirements.txt` with pinned versions, `rated_capacity.csv`, and `README.md` with the run order.
- `CITATION.cff` — citation metadata (ORCID 0009-0001-4366-729X).
