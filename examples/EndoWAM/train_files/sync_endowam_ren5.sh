#!/usr/bin/env bash
set -euo pipefail

RCLONE_BIN="${RCLONE_BIN:-rclone}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive:}"
DRIVE_FOLDER_ID="${DRIVE_FOLDER_ID:-10HpXkYo8FshgDsTSL1ki0i0rZSqE85V0}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/mnt/data-hdd3/ljs/datasets/endowam_pseudo_z60}"
REN5_PROXY="${REN5_PROXY:-http://127.0.0.1:7890}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_SECONDS="${RETRY_SECONDS:-15}"

if ! command -v "${RCLONE_BIN}" >/dev/null 2>&1; then
  echo "rclone was not found: ${RCLONE_BIN}" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT_DIR}"

sync_once() {
  local attempt="$1"
  local -a network_env=(
    env
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
    -u http_proxy -u https_proxy -u all_proxy
  )

  if (( attempt % 2 == 0 )) && [[ -n "${REN5_PROXY}" ]]; then
    echo "network_route=clash proxy=${REN5_PROXY}"
    network_env=(
      env
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
      -u http_proxy -u https_proxy -u all_proxy
      HTTP_PROXY="${REN5_PROXY}" HTTPS_PROXY="${REN5_PROXY}"
      http_proxy="${REN5_PROXY}" https_proxy="${REN5_PROXY}"
    )
  else
    echo "network_route=direct"
  fi

  "${network_env[@]}" \
    RCLONE_CONFIG_GDRIVE_ROOT_FOLDER_ID="${DRIVE_FOLDER_ID}" \
    "${RCLONE_BIN}" copy "${RCLONE_REMOTE}" "${DATA_ROOT_DIR}" \
      --fast-list \
      --transfers 8 \
      --checkers 16 \
      --drive-chunk-size 128M \
      --stats 30s \
      --stats-one-line
}

verify_dataset() {
  local subset required
  for subset in ureter ercp esophagus; do
    for required in meta/info.json meta/stats_gr00t.json data/chunk-000 videos/chunk-000; do
      [[ -e "${DATA_ROOT_DIR}/${subset}/${required}" ]] || return 1
    done
  done
}

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  echo "[$(date -Is)] dataset_sync_attempt=${attempt}/${MAX_ATTEMPTS}"
  if sync_once "${attempt}" && verify_dataset; then
    echo "[$(date -Is)] EndoWAM dataset verified: ${DATA_ROOT_DIR}"
    du -sh "${DATA_ROOT_DIR}"
    exit 0
  fi
  echo "[$(date -Is)] sync incomplete; retrying in ${RETRY_SECONDS}s" >&2
  sleep "${RETRY_SECONDS}"
done

echo "EndoWAM sync failed after ${MAX_ATTEMPTS} attempts." >&2
exit 1
