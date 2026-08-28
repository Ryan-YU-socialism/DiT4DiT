#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/data-hdd2/ljs/.conda/envs/dit4dit-ren5}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "ren5 environment is missing: ${ENV_PREFIX}" >&2
  echo "Run examples/EndoWAM/train_files/setup_ren5_env.sh first." >&2
  exit 1
fi
export PATH="${ENV_PREFIX}/bin:${PATH}"

# GPU 2 on ren5 currently reports an NVML/PCIe error. Keep it invisible to
# CUDA and DeepSpeed until that hardware fault has been repaired and verified.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_ren5_2x3090}"
export EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3090}"
export CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_2x3090.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_ren5_2x3090.yaml}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero3_2x3090.json}"

export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60}"
export BASE_MODEL="${BASE_MODEL:-/mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/data-hdd3/ljs/experiments/DiT4DiT}"
export HF_HOME="${HF_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/torch}"
export WANDB_MODE="${WANDB_MODE:-offline}"

exec bash "${SCRIPT_DIR}/run_endowam_4xh800.sh"
