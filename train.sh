#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file:./mlruns}
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py --test-size 0.2 --C 1.0 --max-iter 200