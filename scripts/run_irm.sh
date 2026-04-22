#!/bin/bash
# Train LightGCN + IRM
set -e

CONFIG=${1:-configs/ml1m.yaml}
LAMBDA=${2:-1.0}
N_ENVS=${3:-3}

python -m src.train --config "$CONFIG" --method irm --penalty_weight "$LAMBDA" --n_envs "$N_ENVS"
