#!/bin/bash
# Train vanilla LightGCN baseline
set -e

CONFIG=${1:-configs/ml1m.yaml}
python -m src.train --config "$CONFIG" --method baseline
