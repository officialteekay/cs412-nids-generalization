#!/bin/bash
set -e
S="$1"
cd "$(dirname "$0")"
python3 run_experiments.py --samples "$S/samples" --out "$S/results/multi_full.json" --models logreg rf mlp --mode multi
python3 run_experiments.py --samples "$S/samples" --out "$S/results/single_60k.json" --models attn mlp --subsample 60000 --tag _60k
python3 run_experiments.py --samples "$S/samples" --out "$S/results/multi_60k.json" --models attn mlp --subsample 60000 --tag _60k --mode multi
echo PHASE2_DONE
