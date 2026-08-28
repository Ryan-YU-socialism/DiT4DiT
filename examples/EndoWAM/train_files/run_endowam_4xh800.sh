#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/root/autodl-tmp/datasets/endowam_pseudo_z60}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Cosmos-Predict2.5-2B}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/root/autodl-tmp/outputs/DiT4DiT}"
RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_4xh800}"
CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_4xh800.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_4xh800.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero2_h800.json}"
WANDB_MODE="${WANDB_MODE:-offline}"

CODE_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY_WORKTREE:-0}" != "1" ]]; then
  echo "The Git worktree is dirty. Commit the exact training code first, or set ALLOW_DIRTY_WORKTREE=1 explicitly." >&2
  exit 1
fi

for subset in ureter ercp esophagus; do
  for required in meta/info.json meta/stats_gr00t.json data/chunk-000 videos/chunk-000; do
    if [[ ! -e "${DATA_ROOT_DIR}/${subset}/${required}" ]]; then
      echo "Missing dataset entry: ${DATA_ROOT_DIR}/${subset}/${required}" >&2
      exit 1
    fi
  done
done

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Cosmos model directory does not exist: ${BASE_MODEL}" >&2
  exit 1
fi

python - "${NUM_PROCESSES}" <<'PY'
import sys
import torch

requested = int(sys.argv[1])
available = torch.cuda.device_count()
if available < requested:
    raise SystemExit(f"Requested {requested} GPUs, but PyTorch sees only {available}.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected CUDA device does not report bfloat16 support.")

print(f"torch={torch.__version__}, cuda={torch.version.cuda}, visible_gpus={available}")
for index in range(requested):
    props = torch.cuda.get_device_properties(index)
    print(
        f"gpu[{index}]={props.name}, "
        f"memory={props.total_memory / 1024**3:.1f} GiB"
    )
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU topology:"
  nvidia-smi topo -m || true
fi
echo "Dataset disk usage:"
du -sh "${DATA_ROOT_DIR}"
echo "Persistent-disk capacity:"
df -h "${DATA_ROOT_DIR}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIT4DIT_DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}"
export WANDB_MODE
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/cache/torch}"

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${TORCH_HOME}"
cp "$0" "${OUTPUT_DIR}/$(basename "$0")"

RESUME="${RESUME:-auto}"
if [[ "${RESUME}" == "auto" ]]; then
  if find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 \
      -name 'steps_*_pytorch_model.pt' -print -quit 2>/dev/null | grep -q .; then
    RESUME=true
  else
    RESUME=false
  fi
fi
if [[ "${RESUME}" != "true" && "${RESUME}" != "false" ]]; then
  echo "RESUME must be auto, true, or false; got ${RESUME}" >&2
  exit 1
fi

GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE * 2))
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Launching ${NUM_PROCESSES} GPUs with global batch ${GLOBAL_BATCH_SIZE}."
echo "Code commit: ${CODE_COMMIT}"
echo "Dataset: ${DATA_ROOT_DIR}"
echo "Base model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Resume: ${RESUME}"
echo "Max steps: ${MAX_TRAIN_STEPS}"
echo "Log: ${LOG_FILE}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  DiT4DiT/training/train.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.cosmos25.base_model "${BASE_MODEL}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT_DIR}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.is_resume "${RESUME}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  2>&1 | tee -a "${LOG_FILE}"
