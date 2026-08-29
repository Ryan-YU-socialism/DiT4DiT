#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/mnt/data-hdd3/ljs/experiments/DiT4DiT}"
RUN_ID="${RUN_ID:-dit4dit_endowam_pseudo_z60_ren5_2x3090}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
SAVE_CONSOLIDATED_CHECKPOINTS="${SAVE_CONSOLIDATED_CHECKPOINTS:-false}"
SAVE_FINAL_TRAINING_STATE="${SAVE_FINAL_TRAINING_STATE:-true}"
SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL:-true}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60}"
BASE_MODEL="${BASE_MODEL:-/mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B}"
CONFIG_YAML="${CONFIG_YAML:-DiT4DiT/config/endowam/dit4dit_endowam_pseudo_z60_ren5_2x3090.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-DiT4DiT/config/deepseeds/deepspeed_endowam_ren5_2x3090.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-DiT4DiT/config/deepseeds/endowam_zero3_2x3090.json}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/data-hdd2/ljs/.conda/envs/dit4dit-ren5}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-RTX 3090}"

if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer; got ${MAX_ATTEMPTS}" >&2
  exit 2
fi
if [[ ! "${RETRY_DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a non-negative integer; got ${RETRY_DELAY_SECONDS}" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate supervisors for the same run." >&2
  exit 2
fi

RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
SUPERVISOR_LOG_DIR="${RUN_DIR}/supervisor_logs"
mkdir -p "${SUPERVISOR_LOG_DIR}"
cd "${REPO_ROOT}"

# Keep a second reconnect from launching the same experiment concurrently.
exec 9>"${SUPERVISOR_LOG_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "$(date -Is) Another supervisor already owns RUN_ID=${RUN_ID}." >&2
  exit 75
fi

EXPECTED_CODE_COMMIT="${EXPECTED_CODE_COMMIT:-$(git rev-parse HEAD)}"
SUPERVISOR_LOG="${SUPERVISOR_LOG_DIR}/supervisor.log"
ATTEMPT_INDEX="${SUPERVISOR_LOG_DIR}/attempts.tsv"
LOCKED_LAUNCH_CONFIG="${SUPERVISOR_LOG_DIR}/launch_config.env"

# Resolve and lock every training-semantic input once. A later reconnect must
# provide the same values instead of silently falling back to runner defaults.
launch_config_candidate="$(mktemp "${SUPERVISOR_LOG_DIR}/launch_config.XXXXXX")"
{
  printf 'EXPECTED_CODE_COMMIT=%q\n' "${EXPECTED_CODE_COMMIT}"
  printf 'RUN_ROOT_DIR=%q\n' "${RUN_ROOT_DIR}"
  printf 'RUN_ID=%q\n' "${RUN_ID}"
  printf 'NUM_PROCESSES=%q\n' "${NUM_PROCESSES}"
  printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'PER_DEVICE_BATCH_SIZE=%q\n' "${PER_DEVICE_BATCH_SIZE}"
  printf 'GRADIENT_ACCUMULATION_STEPS=%q\n' "${GRADIENT_ACCUMULATION_STEPS}"
  printf 'NUM_WORKERS=%q\n' "${NUM_WORKERS}"
  printf 'MAX_TRAIN_STEPS=%q\n' "${MAX_TRAIN_STEPS}"
  printf 'SAVE_INTERVAL=%q\n' "${SAVE_INTERVAL}"
  printf 'EVAL_INTERVAL=%q\n' "${EVAL_INTERVAL}"
  printf 'LOGGING_FREQUENCY=%q\n' "${LOGGING_FREQUENCY}"
  printf 'SAVE_CONSOLIDATED_CHECKPOINTS=%q\n' "${SAVE_CONSOLIDATED_CHECKPOINTS}"
  printf 'SAVE_FINAL_TRAINING_STATE=%q\n' "${SAVE_FINAL_TRAINING_STATE}"
  printf 'SAVE_FINAL_MODEL=%q\n' "${SAVE_FINAL_MODEL}"
  printf 'DATA_ROOT_DIR=%q\n' "${DATA_ROOT_DIR}"
  printf 'BASE_MODEL=%q\n' "${BASE_MODEL}"
  printf 'CONFIG_YAML=%q\n' "${CONFIG_YAML}"
  printf 'ACCELERATE_CONFIG=%q\n' "${ACCELERATE_CONFIG}"
  printf 'DEEPSPEED_CONFIG=%q\n' "${DEEPSPEED_CONFIG}"
  printf 'ENV_PREFIX=%q\n' "${ENV_PREFIX}"
  printf 'EXPECTED_GPU_SUBSTRING=%q\n' "${EXPECTED_GPU_SUBSTRING}"
} > "${launch_config_candidate}"
if [[ -e "${LOCKED_LAUNCH_CONFIG}" ]]; then
  if ! cmp -s "${LOCKED_LAUNCH_CONFIG}" "${launch_config_candidate}"; then
    echo "The locked launch configuration differs from this invocation:" >&2
    diff -u "${LOCKED_LAUNCH_CONFIG}" "${launch_config_candidate}" >&2 || true
    rm -f "${launch_config_candidate}"
    exit 78
  fi
  rm -f "${launch_config_candidate}"
else
  mv "${launch_config_candidate}" "${LOCKED_LAUNCH_CONFIG}"
fi

matching_worker_pattern="DiT4DiT/training/train.py.*--run_id ${RUN_ID}"
if pgrep -f -- "${matching_worker_pattern}" >/dev/null 2>&1; then
  echo "A training worker for RUN_ID=${RUN_ID} is already running outside this supervisor." >&2
  exit 75
fi

printf '%s run_id=%s commit=%s max_attempts=%s retry_delay_seconds=%s\n' \
  "$(date -Is)" "${RUN_ID}" "${EXPECTED_CODE_COMMIT}" \
  "${MAX_ATTEMPTS}" "${RETRY_DELAY_SECONDS}" | tee -a "${SUPERVISOR_LOG}"

last_rc=1
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  current_commit="$(git rev-parse HEAD)"
  if [[ "${current_commit}" != "${EXPECTED_CODE_COMMIT}" ]]; then
    echo "$(date -Is) Code changed to ${current_commit}; refusing to continue." \
      | tee -a "${SUPERVISOR_LOG}"
    exit 78
  fi

  attempt_log="${SUPERVISOR_LOG_DIR}/attempt_${attempt}_$(date +%Y%m%dT%H%M%S).log"
  echo "$(date -Is) attempt=${attempt}/${MAX_ATTEMPTS} start resume=auto" \
    | tee -a "${SUPERVISOR_LOG}"

  set +e
  PYTHONUNBUFFERED=1 \
  RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
  RUN_ID="${RUN_ID}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
  SAVE_CONSOLIDATED_CHECKPOINTS="${SAVE_CONSOLIDATED_CHECKPOINTS}" \
  SAVE_FINAL_TRAINING_STATE="${SAVE_FINAL_TRAINING_STATE}" \
  SAVE_FINAL_MODEL="${SAVE_FINAL_MODEL}" \
  DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
  BASE_MODEL="${BASE_MODEL}" \
  CONFIG_YAML="${CONFIG_YAML}" \
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG}" \
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG}" \
  ENV_PREFIX="${ENV_PREFIX}" \
  EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING}" \
  RESUME=auto \
    bash "${SCRIPT_DIR}/run_endowam_ren5_2x3090.sh" \
      2>&1 | tee -a "${attempt_log}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  last_rc="${pipeline_status[0]}"
  tee_rc="${pipeline_status[1]}"

  printf '%s\tattempt=%s\trc=%s\tlog=%s\n' \
    "$(date -Is)" "${attempt}" "${last_rc}" "${attempt_log}" \
    >> "${ATTEMPT_INDEX}"

  if [[ "${tee_rc}" -ne 0 ]]; then
    echo "$(date -Is) Supervisor logging failed rc=${tee_rc}; retry disabled." \
      | tee -a "${SUPERVISOR_LOG}"
    exit 74
  fi

  if [[ "${last_rc}" -eq 0 ]]; then
    echo "$(date -Is) Training completed normally." | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi
  if [[ "${last_rc}" -eq 130 || "${last_rc}" -eq 143 ]]; then
    echo "$(date -Is) Training was interrupted; automatic retry disabled." \
      | tee -a "${SUPERVISOR_LOG}"
    exit "${last_rc}"
  fi
  if pgrep -f -- "${matching_worker_pattern}" >/dev/null 2>&1; then
    echo "$(date -Is) A worker survived runner exit; refusing an overlapping retry." \
      | tee -a "${SUPERVISOR_LOG}"
    exit 75
  fi
  if [[ "${last_rc}" -eq 126 || "${last_rc}" -eq 127 ]] || \
      grep -Eqi \
        'CUDA out of memory|OutOfMemoryError|ModuleNotFoundError|unrecognized arguments|worktree is dirty|environment is missing|Missing dataset entry|model directory does not exist|Requested [0-9]+ GPUs|does not report bfloat16|expected a name containing|must be a positive integer|must be true or false' \
        "${attempt_log}"; then
    echo "$(date -Is) Deterministic failure detected; automatic retry disabled." \
      | tee -a "${SUPERVISOR_LOG}"
    exit "${last_rc}"
  fi
  if [[ "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    break
  fi

  echo "$(date -Is) attempt=${attempt} failed rc=${last_rc}; retrying in ${RETRY_DELAY_SECONDS}s." \
    | tee -a "${SUPERVISOR_LOG}"
  sleep "${RETRY_DELAY_SECONDS}"
done

echo "$(date -Is) Exhausted ${MAX_ATTEMPTS} attempts; last_rc=${last_rc}." \
  | tee -a "${SUPERVISOR_LOG}"
exit "${last_rc}"
