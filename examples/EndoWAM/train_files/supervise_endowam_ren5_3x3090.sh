#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUNNER_SCRIPT="${RUNNER_SCRIPT:-run_endowam_ren5_3x3090.sh}"
export RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_ren5_3x3090}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-3}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export ENV_PREFIX="${ENV_PREFIX:-/mnt/data-hdd2/ljs/.conda/envs/endoguard}"
export CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_3x3090.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_ren5_zero2_3x3090.yaml}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero2_3x3090.json}"

exec bash "${SCRIPT_DIR}/supervise_endowam_ren5.sh"
