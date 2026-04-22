#!/bin/bash
# Ablation sweep over E ∈ {2,3,5} and λ ∈ {0.1, 1, 10, 100}
set -e

CONFIG=${1:-configs/ml1m.yaml}

for METHOD in irm vrex; do
    for N_ENVS in 2 3 5; do
        for LAMBDA in 0.1 1.0 10.0 100.0; do
            echo "=== ${METHOD} | E=${N_ENVS} | λ=${LAMBDA} ==="
            python -m src.train --config "$CONFIG" \
                --method "$METHOD" \
                --n_envs "$N_ENVS" \
                --penalty_weight "$LAMBDA"
        done
    done
done
