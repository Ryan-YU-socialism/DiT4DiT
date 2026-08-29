#!/usr/bin/env bash
set -euo pipefail

HF_CLI="${HF_CLI:-/home/user/miniconda3/bin/huggingface-cli}"
HF_HOME="${HF_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/huggingface}"
MODEL_DIR="${MODEL_DIR:-/mnt/data-hdd2/ljs/models/Cosmos-Predict2.5-2B}"
MODEL_ID="${MODEL_ID:-nvidia/Cosmos-Predict2.5-2B}"
MODEL_REVISION="${MODEL_REVISION:-diffusers/base/post-trained}"
REN5_PROXY="${REN5_PROXY:-http://127.0.0.1:7890}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SECONDS="${RETRY_SECONDS:-15}"

if [[ ! -x "${HF_CLI}" ]]; then
  echo "huggingface-cli was not found: ${HF_CLI}" >&2
  exit 1
fi

mkdir -p "${HF_HOME}" "${MODEL_DIR}"

download_once() {
  local attempt="$1"
  local -a network_env=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
    -u http_proxy -u https_proxy -u all_proxy
  )

  # ren5's current Clash route occasionally drops Hugging Face TLS sessions.
  # Alternate direct and proxy attempts so either route can recover the run.
  if (( attempt % 2 == 0 )) && [[ -n "${REN5_PROXY}" ]]; then
    echo "network_route=clash proxy=${REN5_PROXY}"
    network_env=(
      env
      HTTP_PROXY="${REN5_PROXY}" HTTPS_PROXY="${REN5_PROXY}"
      http_proxy="${REN5_PROXY}" https_proxy="${REN5_PROXY}"
    )
  else
    echo "network_route=direct"
  fi

  "${network_env[@]}" \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    HF_HUB_ETAG_TIMEOUT=60 \
    HF_HOME="${HF_HOME}" \
    "${HF_CLI}" download "${MODEL_ID}" \
      --revision "${MODEL_REVISION}" \
      --local-dir "${MODEL_DIR}"
}

verify_model() {
  local required=(
    model_index.json
    text_encoder/config.json
    text_encoder/model.safetensors.index.json
    transformer/config.json
    transformer/diffusion_pytorch_model.safetensors
    vae/config.json
    vae/diffusion_pytorch_model.safetensors
  )
  local path
  for path in "${required[@]}"; do
    [[ -s "${MODEL_DIR}/${path}" ]] || return 1
  done
  [[ "$(find "${MODEL_DIR}/text_encoder" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)" -eq 4 ]] \
    || return 1
  ! find "${MODEL_DIR}" -name '*.incomplete' -print -quit | grep -q .
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  echo "[$(date -Is)] model_download_attempt=${attempt}/${MAX_ATTEMPTS}"
  if download_once "${attempt}" && verify_model; then
    echo "[$(date -Is)] Cosmos model verified: ${MODEL_DIR}"
    du -sh "${MODEL_DIR}"
    exit 0
  fi
  echo "[$(date -Is)] download incomplete; retrying in ${RETRY_SECONDS}s" >&2
  sleep "${RETRY_SECONDS}"
done

echo "Cosmos model download failed after ${MAX_ATTEMPTS} attempts." >&2
exit 1
