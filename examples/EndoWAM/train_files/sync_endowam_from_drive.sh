#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <authenticated-rclone-remote:path/to/endowam_pseudo_z60> [destination]" >&2
  exit 2
fi

REMOTE_PATH="$1"
DESTINATION="${2:-/root/autodl-tmp/datasets/endowam_pseudo_z60}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required. Configure Google Drive authentication outside the repository first." >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
rclone copy "${REMOTE_PATH}" "${DESTINATION}" \
  --fast-list \
  --transfers 8 \
  --checkers 16 \
  --drive-chunk-size 128M \
  --progress

for subset in ureter ercp esophagus; do
  for required in meta/info.json meta/stats_gr00t.json data/chunk-000 videos/chunk-000; do
    if [[ ! -e "${DESTINATION}/${subset}/${required}" ]]; then
      echo "Sync verification failed: ${DESTINATION}/${subset}/${required}" >&2
      exit 1
    fi
  done
done

echo "EndoWAM dataset verified at ${DESTINATION}"
