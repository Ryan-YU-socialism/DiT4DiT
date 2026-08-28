#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BIN="${CONDA_BIN:-/home/user/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/data-hdd2/ljs/.conda/envs/dit4dit-ren5}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/mnt/data-hdd2/ljs/.conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/data-hdd2/ljs/.cache/pip}"
export HF_HOME="${HF_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data-hdd2/ljs/.cache/dit4dit/xdg}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/data-hdd2/ljs/.cache/dit4dit/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/data-hdd2/ljs/.cache/dit4dit/cuda}"
export TMPDIR="${TMPDIR:-/mnt/data-hdd2/ljs/.cache/dit4dit/tmp}"
# ren5's shared user profile contains unrelated user-site packages and a stale
# NVIDIA extra index. Neither may leak into this isolated training environment.
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL || true

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

mkdir -p \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}" "${TMPDIR}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --prefix "${ENV_PREFIX}" python=3.10 pip -y
fi

"${CONDA_BIN}" install --prefix "${ENV_PREFIX}" \
  --channel nvidia/label/cuda-12.4.1 cuda-nvcc=12.4 -y
export CUDA_HOME="${ENV_PREFIX}"

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# requirements.txt targets Blackwell/PyTorch 2.7. Keep all project pins except
# the three framework packages replaced above for ren5's CUDA 12.4 driver.
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "${FILTERED_REQUIREMENTS}"' EXIT
grep -Ev '^(torch|torchvision|triton)==' requirements.txt > "${FILTERED_REQUIREMENTS}"
"${PYTHON}" -m pip install -r "${FILTERED_REQUIREMENTS}"
"${PYTHON}" -m pip install -e .

CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" - <<'PY'
import torch

assert torch.__version__.startswith("2.5.1"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert torch.cuda.is_bf16_supported()
for index in range(2):
    name = torch.cuda.get_device_name(index)
    assert "RTX 3090" in name, name
    print(index, name, torch.cuda.get_device_properties(index).total_memory / 1024**3)
PY

"${ENV_PREFIX}/bin/nvcc" --version

echo "ren5 environment ready: ${ENV_PREFIX}"
