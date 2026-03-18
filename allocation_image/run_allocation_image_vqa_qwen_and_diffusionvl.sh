#!/usr/bin/env bash
# =============================================================================
# run_allocation_image_vqa_qwen_and_diffusionvl.sh
#
# Attention allocation (system / visual / instruction) for image VQA tasks:
#   - mmmu
#   - mme
#   - mmvp
#
# Models:
#   - Qwen2.5-VL base
#   - DiffusionVL-QwenVL
#
# Usage:
#   bash run_allocation_image_vqa_qwen_and_diffusionvl.sh [dataset|all] [num_samples]
# Example:
#   bash run_allocation_image_vqa_qwen_and_diffusionvl.sh all 64
# =============================================================================
set -euo pipefail

# Required by user workflow: enable proxy before running.
if command -v clashon >/dev/null 2>&1; then
  clashon || true
fi
if command -v clashsub >/dev/null 2>&1; then
  clashsub use 2 || true
fi

source /data/liuyx/miniconda3/etc/profile.d/conda.sh
conda activate diffusionvl

export HF_HOME="${HF_HOME:-/data/liuyx/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
unset HF_ENDPOINT || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-all}"
NUM_SAMPLES="${2:-128}"
GPU_ID="${GPU_ID:-0}"

QWEN_BASE_MODEL_ID="${QWEN_BASE_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
DIFFUSIONVL_MODEL_ID="${DIFFUSIONVL_MODEL_ID:-hustvl/DiffusionVL-Qwen2.5VL-7B}"

resolve_model_ref() {
  local ref="$1"
  if [[ -d "$ref" ]]; then
    echo "$ref"
    return 0
  fi
  local cache_repo_dir="${HF_HUB_CACHE}/models--${ref//\//--}"
  if [[ -d "${cache_repo_dir}/snapshots" ]]; then
    local snap
    snap="$(find "${cache_repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)"
    if [[ -n "$snap" ]]; then
      echo "$snap"
      return 0
    fi
  fi
  echo "$ref"
}

QWEN_BASE_MODEL_ID_RESOLVED="$(resolve_model_ref "${QWEN_BASE_MODEL_ID}")"
DIFFUSIONVL_MODEL_ID_RESOLVED="$(resolve_model_ref "${DIFFUSIONVL_MODEL_ID}")"

run_one() {
  local model_type="$1"
  local task="$2"
  local model_id="$3"
  local out_dir="${WORK_DIR}/attn_alloc_${task}_${model_type}"
  echo "========================================"
  echo "[${model_type}] allocation avg — ${task} (samples=${NUM_SAMPLES})"
  echo "========================================"
  python "${WORK_DIR}/attention_allocation_image_vqa_qwen_diffusionvl.py" \
    --dataset "${task}" \
    --model_type "${model_type}" \
    --model_id "${model_id}" \
    --num_samples "${NUM_SAMPLES}" \
    --gpu_id "${GPU_ID}" \
    --output_dir "${out_dir}"
  echo "[${model_type}] done → ${out_dir}"
}

run_task() {
  local task="$1"
  run_one "qwen2_5_vl_base" "${task}" "${QWEN_BASE_MODEL_ID_RESOLVED}"
  run_one "diffusionvl_qwenvl" "${task}" "${DIFFUSIONVL_MODEL_ID_RESOLVED}"
}

case "${DATASET}" in
  mme|mmvp|mmmu)
    run_task "${DATASET}"
    ;;
  all)
    run_task mmmu
    run_task mme
    run_task mmvp
    ;;
  *)
    echo "Usage: $0 [mme|mmvp|mmmu|all] [num_samples]"
    exit 1
    ;;
esac

echo ""
echo "All Qwen-base and DiffusionVL image allocation runs complete."

