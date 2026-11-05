# Do Network Intrusion Detectors Generalize?

CS 412 (Introduction to Machine Learning, Fall 2026, UIC) final project by Daniel Kolawole.

Trains logistic regression, a random forest, an MLP, and a small FT-Transformer style self-attention model on one NetFlow v2 dataset and tests them on the others (UNSW-NB15, ToN-IoT, CSE-CIC-IDS2018).

## Data

Download NF-UNSW-NB15-v2, NF-ToN-IoT-v2 and NF-CSE-CIC-IDS2018-v2 from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ and save the zips as `unsw.zip`, `ton.zip` and `cic.zip` in one folder. Cite Sarhan, Layeghy and Portmann (2022) when using them.

## Steps

```
pip install -r requirements.txt
python prepare_data.py DATA_DIR SAMPLE_DIR
python run_experiments.py --samples SAMPLE_DIR --out results/phase1_sklearn.json --models logreg rf
python run_experiments.py --samples SAMPLE_DIR --out results/phase1_mlp.json --models mlp
python run_experiments.py --samples SAMPLE_DIR --out results/multi_full.json --models logreg rf mlp --mode multi
python run_experiments.py --samples SAMPLE_DIR --out results/single_60k.json --models attn mlp --subsample 60000 --tag _60k
python run_experiments.py --samples SAMPLE_DIR --out results/multi_60k.json --models attn mlp --subsample 60000 --tag _60k --mode multi --seeds 0
python run_experiments.py --samples SAMPLE_DIR --out results/no_ttl.json --models logreg rf mlp --drop MIN_TTL MAX_TTL --tag _nottl
python feature_analysis.py SAMPLE_DIR results/features.json
python make_final_assets.py results assets
python gen_numbers.py assets
```

## Files

- `prepare_data.py` streams each dataset, drops IPs, ports and DNS_QUERY_ID, and keeps a 300,000 flow random sample
- `common.py` splits, preprocessing and metrics
- `models.py` the four models
- `run_experiments.py` single-source, multi-source, subsample and feature-drop runs
- `feature_analysis.py` permutation importance and benign feature shift
- `make_final_assets.py`, `gen_numbers.py` tables, figures and numbers for the report
- `check_numbers.py`, `voice_check.py` checks used while writing the reports
