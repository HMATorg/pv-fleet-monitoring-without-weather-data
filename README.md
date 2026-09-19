# Monitoring a distributed photovoltaic fleet without weather data

Reproduction code for the paper *Monitoring a distributed photovoltaic fleet without weather data: forecasting limits, daily operating regimes and peer-referenced fault detection*.

**Author:** Hafiz Atif Ali, HMAT Technologies — ORCID [0009-0001-4366-729X](https://orcid.org/0009-0001-4366-729X)

The study uses 7.5 years of hourly production from ten municipal PV systems in Calgary, Canada (City of Calgary open data, dataset `ytdn-2qsp`) and no on-site or reanalysis weather inputs. It covers three tasks:

- **Forecasting.** Hourly models benchmarked against CLIPER (optimal climatology–persistence) with rolling-origin folds and day-block bootstrap confidence intervals, plus a skill-versus-horizon curve and a fixed 11:00 issue-time day-ahead test.
- **Operating regimes.** Daily 24-hour fingerprints via PCA and an autoencoder, k-means regimes, and tests of whether the structure is genuinely multi-modal (Hartigan dip test, gap statistic, silhouette in a common space, bootstrap ARI) or a discretised amplitude family.
- **Fault detection.** A peer-referenced robust causal z-score with an optional alarm-masking mode and a silent-day rule, benchmarked on injected faults of six types against a single-site clear-sky index and Isolation Forest.

All results, tables and figures in the paper are produced by the scripts in [`code/`](code/), with fixed random seeds and pinned package versions. See [`code/README.md`](code/README.md) for the data download and run order, and [`PUSH_TO_GITHUB.md`](PUSH_TO_GITHUB.md) for how this repository was published.

## Quick start

```bash
pip install -r code/requirements.txt
# place the Calgary CSV as code/data.csv (see code/README.md)
cd code && python prep.py
```

## Licence

MIT — see [LICENSE](LICENSE). If you use the code, please cite the paper and [`CITATION.cff`](CITATION.cff).
