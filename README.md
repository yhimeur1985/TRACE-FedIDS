# TRACE: Trust-Aware Federated Intrusion Detection Under Non-IID and Byzantine IoT Clients

Code and complete experimental results for the paper

> Y. Himeur, S. Atalla, A. Amira, M. A. Ala'anzy, E. M. Bastaki,
> "Trust-Aware Federated Intrusion Detection Under Non-IID and Byzantine IoT Clients" (under review).

**TRACE** (Trust from Reliability, Agreement and Consistency Estimation) weights each client of a federated
intrusion detector by update agreement, security-aware validation reliability on a small trusted root set,
and historical consistency, each turned into a robust anomaly gate relative to the current client population.
**TRACE-C** adds a self-referenced gate that compares each client's update with a prediction from its own past.
Clients train **FT-T-G**, a grouped-token FT-Transformer of about 19k parameters.

Everything is implemented in NumPy (hand-written, gradient-checked backward pass) and runs on CPU.

## Repository layout

| Path | Content |
|---|---|
| `fl.py` | Federated simulator: partitions, local training, attacks, aggregation rules, evaluation |
| `model.py` | FT-T-G / FT-Transformer / MLP backbones (NumPy) |
| `prep_data.py` | Leakage-aware preprocessing of Edge-IIoTset and CIC-BCCC-NRC TabularIoTAttack-2024 |
| `runner.py`, `runner2.py` | Experiment grids of the first version and of the revision (one JSON per run, resumable) |
| `central.py`, `scalability.py` | Centralised backbone reference and server-cost benchmark (n = 20 ... 200) |
| `trigger_families.py`, `trig_patch.py` | Selection of the held-out backdoor trigger families (`data/triggers.json`) |
| `make_assets.py`, `make_assets2.py` | Generate every table, figure and number of the paper and its supplement from `results/` |
| `results/runs.zip` | 1,429 run files (`<tag>_<config-hash>.json`): config, metrics per round, final metrics, trust weights. Unzip into `results/runs/` before use (`cd results && unzip runs.zip`) |
| `results/scores/` | Saved test scores of clean models (operating-point analysis) |
| `results/*.json` | Centralised, scalability, speed and root-only results |
| `data/` | File manifests, excluded files, trigger definitions, capture provenance |
| `scripts/` | Auxiliary scripts (root-only baseline, trigger search, debugging helpers) |

Aggregation rules: FedAvg, FedProx, Krum, trimmed mean, median, FLTrust, FLAME, Zeno, FoolsGold, FLDetector,
FedAvg+root, bucketing+median, TRACE, TRACE-C.
Attacks: label flipping (all / one family), sign flipping, ALIE, Min-Max, boosted backdoor with five trigger
families (extreme, mid-range, distributed, semantic, clean-label), TRACE-aware adaptive backdoor / label flipping.

## Setup

```bash
python -m pip install -r requirements.txt   # Python >= 3.10
```

## Data

The datasets are public and are not redistributed here.

* Edge-IIoTset (ML file `ML-EdgeIIoT-dataset.csv`), Ferrag et al., IEEE Access 2022.
* CIC-BCCC-NRC TabularIoTAttack-2024 (University of New Brunswick), testbeds ACI-IoT-2023, Edge-IIoTSet-2022,
  IoMT-2024, IoT-HCRL-2019, MQTTIoT-IDS-2020, IoT-2022.

```bash
export EDGE_IIOT_CSV="/path/to/ML-EdgeIIoT-dataset.csv"
export CIC_BCCC_ROOT="/path/to/CIC-BCCC-NRC-TabularIoTAttacks-2024"   # folder with the CIC-BCCC-NRC-* testbed folders
python prep_data.py edge cic cic_full   # writes data/edge.npz, data/cic.npz (47 files), data/cic_full.npz (all 54 files)
```

`data/cic_manifest.json` and `data/cic_full_manifest.json` list every source file with its row counts.

## Reproducing the experiments

First unpack the run files: `cd results && unzip runs.zip && cd ..`. Every run is cached in `results/runs/` under a hash of its configuration, so existing runs are skipped.

```bash
python runner.py main hetero frac root ablation:edge ablation:cic backbone           # first version
python runner2.py newbase minmax noserver adaptive grouping triggers triggers2 tracec \
                  confirm rootstress lftarget prep ablation3 partial frac3 oppoints \
                  root3 hetero3 fulldata capdisj consdev envdev                          # revision
python central.py edge ftt-g 3000 0       # centralised references (dataset, model, steps, seed)
python scalability.py
```

A single run, e.g. TRACE-C under the backdoor on Edge-IIoTset:

```python
import fl
cfg = dict(dataset="edge", partition="category", rounds=40, eval_every=5, lr=1e-3, local_steps=20,
           agg="trace", trace=dict(consistency=True), attack="bd", seed=0)
print(fl.run(cfg)["final"])     # macro-F1, detection rate, FPR, attack success rate, ...
```

Runtime: about 2-5 CPU minutes per run (2 cores were used for all 1,429 runs).

## Regenerating the paper's tables and figures

```bash
python make_assets2.py            # tables/figures/numbers -> paper_assets/ (or $TRACE_PAPER_DIR)
python make_assets2.py --prov     # also the provenance table (needs data/*.npz)
```

Values in the tables are means over seeds of the metrics averaged over rounds 30, 35 and 40;
the supplementary tables also give 95 % confidence intervals and seed-level final-round values.

## Run-file tags

`main` (main comparison, seeds 0-2), `tracec*` (TRACE-C), `trigger` / `tracec-trigger` (held-out triggers),
`adaptive*` (adaptive attackers), `confirm*` (confirmatory seeds 3-4), `abl-*` (ablations), `frac`, `hetero`,
`root*`, `rootstress-*`, `rootdrop-*`, `lftarget*` (trusted-data tests), `partial-*`, `intermittent`, `latejoin`
(participation), `prep`, `fulldata`, `capdisj` (robustness checks), `grouping` (tokenisation control),
`dev-*`, `envdev-*` (development on the validation split, test set unused).

## Citation

```bibtex
@article{himeur2026trace,
  title   = {Trust-Aware Federated Intrusion Detection Under Non-IID and Byzantine IoT Clients},
  author  = {Himeur, Yassine and Atalla, Shadi and Amira, Abbes and Ala'anzy, Mohammed A. and Bastaki, Eesa M.},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The datasets are subject to their own licenses.
