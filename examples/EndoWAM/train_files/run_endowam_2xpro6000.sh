#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_2xpro6000}"
export EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-PRO 6000}"
export CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_4xpro6000.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_2xpro6000.yaml}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero2_accum4.json}"

exec bash "${SCRIPT_DIR}/run_endowam_4xh800.sh"
