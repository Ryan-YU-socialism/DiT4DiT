#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/data-hdd2/ljs/.conda/envs/endoguard}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "ren5 environment is missing: ${ENV_PREFIX}" >&2
  exit 1
fi
export PATH="${ENV_PREFIX}/bin:${PATH}"

# GPU 2 previously had a PCIe/NVML fault. This recipe may only be used after
# all three cards pass CUDA allocation and a three-rank NCCL collective test.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-3}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_ren5_3x3090}"
export EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3090}"
export CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_3x3090.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_ren5_zero2_3x3090.yaml}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero2_3x3090.json}"

# Keep the verified two-card CPUAdam setting unchanged for the requested
# three-card launch. Record benchmark timings before changing thread counts.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60}"
export BASE_MODEL="${BASE_MODEL:-/mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/data-hdd3/ljs/experiments/DiT4DiT}"
export HF_HOME="${HF_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/xdg}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/data-hdd2/ljs/.cache/dit4dit/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/data-hdd2/ljs/.cache/dit4dit/cuda}"
export TMPDIR="${TMPDIR:-/mnt/data-hdd2/ljs/.cache/dit4dit/tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# The two-card recipe needs a process-local NVML shim to hide the historically
# faulty third device. Do not load that partial shim here: this recipe uses all
# three cards and must expose the system NVML library unchanged to NCCL and
# diagnostics such as nvidia-smi.

mkdir -p \
  "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${TMPDIR}"

exec bash "${SCRIPT_DIR}/run_endowam_4xh800.sh"
