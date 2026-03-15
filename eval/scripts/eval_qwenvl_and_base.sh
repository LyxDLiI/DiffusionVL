#!/bin/bash

# Evaluate DiffusionVL-QwenVL and Qwen2.5-VL base model on the same task list.
#
# Usage:
#   cd eval
#   TASK_NAMES="mmmu_val,mme,mmvp,mathvision,mathvista" bash scripts/eval_qwenvl_and_base.sh
#
# Optional envs:
#   DIFFUSIONVL_MODEL_PATH (default: hustvl/DiffusionVL-Qwen2.5VL-7B)
#   BASE_MODEL_PATH        (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   OUTPUT_PATH            (default: ./eval_results/qwenvl_vs_base)
#   TASK_NAMES             (default: mmmu_val,mme,mmvp,mathvision,mathvista)
#   TOTAL_GPUS             (default: auto-detect)
#   AUTO_DOWNLOAD_MODELS   (default: 1)
#   BLOCK_SIZE             (default: 8)
#   STEPS                  (default: 8)
#   DRY_RUN                (default: 0; set 1 to verify scheduler without launching lmms_eval)

set -euo pipefail

DIFFUSIONVL_MODEL_PATH="${DIFFUSIONVL_MODEL_PATH:-hustvl/DiffusionVL-Qwen2.5VL-7B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/qwenvl_vs_base}"
TASK_NAMES="${TASK_NAMES:-mmmu_val,mme,mmvp,mathvision,mathvista}"
TOTAL_GPUS="${TOTAL_GPUS:-}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
DRY_RUN="${DRY_RUN:-0}"

BLOCK_SIZE="${BLOCK_SIZE:-8}"
STEPS="${STEPS:-8}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2_5}"
DIFFUSIONVL_MODEL="llava_onevision_diffusionvl_qwenvl"
BASE_MODEL="qwen2_5_vl"

count_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        python - <<'PY'
import os
s = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not s:
    print(0)
else:
    print(len([x for x in s.split(',') if x.strip()]))
PY
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return 0
    fi

    echo 1
}

AVAILABLE_GPUS="$(count_visible_gpus)"
if ! [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Failed to detect GPU count: $AVAILABLE_GPUS" >&2
    exit 1
fi

if [[ -z "$TOTAL_GPUS" ]]; then
    TOTAL_GPUS="$AVAILABLE_GPUS"
fi

if ! [[ "$TOTAL_GPUS" =~ ^[0-9]+$ ]] || [[ "$TOTAL_GPUS" -lt 1 ]]; then
    echo "[ERROR] TOTAL_GPUS must be a positive integer, got: $TOTAL_GPUS" >&2
    exit 1
fi

if [[ "$AVAILABLE_GPUS" -gt 0 && "$TOTAL_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
    echo "[WARN] Requested TOTAL_GPUS=$TOTAL_GPUS > detected GPUs=$AVAILABLE_GPUS. Capping to $AVAILABLE_GPUS." >&2
    TOTAL_GPUS="$AVAILABLE_GPUS"
fi

resolve_model_path() {
    local model_ref="$1"

    if [[ -d "$model_ref" ]]; then
        echo "$model_ref"
        return 0
    fi

    if [[ "$AUTO_DOWNLOAD_MODELS" != "1" ]]; then
        echo "[ERROR] Model path '$model_ref' does not exist and AUTO_DOWNLOAD_MODELS=$AUTO_DOWNLOAD_MODELS." >&2
        echo "[ERROR] Set AUTO_DOWNLOAD_MODELS=1 or provide a valid local directory." >&2
        exit 1
    fi

    echo "[INFO] Local path not found for '$model_ref'. Trying Hugging Face Hub pre-download..." >&2

    local local_path
    local status
    set +e
    local_path=$(python - "$model_ref" <<'PY' 2>/dev/null
import sys

model_ref = sys.argv[1]

try:
    from huggingface_hub import snapshot_download
except Exception:
    raise SystemExit(2)

try:
    p = snapshot_download(repo_id=model_ref)
except Exception:
    raise SystemExit(1)

print(p)
PY
)
    status=$?
    set -e

    if [[ $status -eq 0 ]]; then
        echo "$local_path"
        return 0
    fi

    if [[ $status -eq 2 ]]; then
        echo "[WARN] huggingface_hub is unavailable; passing repo id directly: $model_ref" >&2
        echo "$model_ref"
        return 0
    fi

    echo "[ERROR] Failed to pre-download '$model_ref'. Check repo id/token/network." >&2
    exit 1
}

trim() {
    local s="$1"
    # shellcheck disable=SC2001
    s="$(echo "$s" | sed 's/^ *//;s/ *$//')"
    echo "$s"
}

mkdir -p "$OUTPUT_PATH"

DIFFUSIONVL_MODEL_PATH_RESOLVED="$(resolve_model_path "$DIFFUSIONVL_MODEL_PATH")"
BASE_MODEL_PATH_RESOLVED="$(resolve_model_path "$BASE_MODEL_PATH")"

IFS=',' read -ra TASKS_RAW <<< "$TASK_NAMES"
TASKS=()
for task in "${TASKS_RAW[@]}"; do
    task_trimmed="$(trim "$task")"
    if [[ -n "$task_trimmed" ]]; then
        TASKS+=("$task_trimmed")
    fi
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "[ERROR] No valid tasks found in TASK_NAMES='$TASK_NAMES'." >&2
    exit 1
fi

TASK_QUEUE=()
for task in "${TASKS[@]}"; do
    if [[ "$task" == "chartqa" ]]; then
        DIFF_GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"stopping_criteria\":[\"\\n\"], \"remasking_strategy\": \"low_confidence_static\"}"
    else
        DIFF_GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"remasking_strategy\": \"low_confidence_static\"}"
    fi
    DIFF_MODEL_ARGS="pretrained=$DIFFUSIONVL_MODEL_PATH_RESOLVED,conv_template=$CONV_TEMPLATE,model_name=diffusionvl_qwenvl,enable_bd3lm=True,bd3lm_block_size=$BLOCK_SIZE"
    TASK_QUEUE+=("$DIFFUSIONVL_MODEL|$DIFFUSIONVL_MODEL_PATH_RESOLVED|$task|$DIFF_GEN_KWARGS|$DIFF_MODEL_ARGS|DiffusionVL-QwenVL")

    BASE_GEN_KWARGS='{"temperature":0,"max_new_tokens":128}'
    BASE_MODEL_ARGS="pretrained=$BASE_MODEL_PATH_RESOLVED"
    TASK_QUEUE+=("$BASE_MODEL|$BASE_MODEL_PATH_RESOLVED|$task|$BASE_GEN_KWARGS|$BASE_MODEL_ARGS|Qwen2.5-VL-Base")
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
if [[ "$TOTAL_TASKS" -eq 0 ]]; then
    echo "[ERROR] Task queue is empty; check TASK_NAMES and model arguments." >&2
    exit 1
fi

DISPATCHED_TASKS=0
FINISHED_TASKS=0
FAILED_TASKS=0

declare -A GPU_STATUS
declare -A GPU_PIDS
declare -A GPU_TASK_DESC
for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

echo "=========================================="
echo "DiffusionVL model ref: $DIFFUSIONVL_MODEL_PATH"
echo "DiffusionVL model path/ref used: $DIFFUSIONVL_MODEL_PATH_RESOLVED"
echo "Base model ref: $BASE_MODEL_PATH"
echo "Base model path/ref used: $BASE_MODEL_PATH_RESOLVED"
echo "Auto download models: $AUTO_DOWNLOAD_MODELS"
echo "Dry run: $DRY_RUN"
echo "Output path: $OUTPUT_PATH"
echo "Tasks: ${TASKS[*]}"
echo "Detected GPUs: $AVAILABLE_GPUS"
echo "Using GPUs: $TOTAL_GPUS"
echo "BD3-LM Block Size: $BLOCK_SIZE"
echo "BD3-LM Steps: $STEPS"
echo "Total jobs: $TOTAL_TASKS"
echo "=========================================="

while [[ $FINISHED_TASKS -lt $TOTAL_TASKS ]]; do
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]:-}" ]]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                pid="${GPU_PIDS[$gpu]}"
                if wait "$pid"; then
                    echo "GPU $gpu job succeeded: ${GPU_TASK_DESC[$gpu]}"
                else
                    FAILED_TASKS=$((FAILED_TASKS + 1))
                    echo "[ERROR] GPU $gpu job failed: ${GPU_TASK_DESC[$gpu]}"
                fi
                GPU_STATUS[$gpu]=0
                unset GPU_PIDS[$gpu]
                unset GPU_TASK_DESC[$gpu]
                FINISHED_TASKS=$((FINISHED_TASKS + 1))
                echo "GPU $gpu released. Finished: $FINISHED_TASKS / $TOTAL_TASKS"
            fi
        fi
    done

    if [[ $DISPATCHED_TASKS -lt $TOTAL_TASKS ]]; then
        for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
            if [[ ${GPU_STATUS[$gpu]} -eq 0 && $DISPATCHED_TASKS -lt $TOTAL_TASKS ]]; then
                IFS='|' read -r EVAL_MODEL MODEL_PATH TASK_NAME GEN_KWARGS MODEL_ARGS RUN_NAME <<< "${TASK_QUEUE[$DISPATCHED_TASKS]}"

                GPU_STATUS[$gpu]=1
                MODEL_PATH_LAST=$(basename "$MODEL_PATH")
                CURRENT_OUTPUT_PATH="$OUTPUT_PATH/$RUN_NAME/$MODEL_PATH_LAST"
                mkdir -p "$CURRENT_OUTPUT_PATH"
                LOG_FILE="$CURRENT_OUTPUT_PATH/${TASK_NAME}.log"

                TASK_DESC="$RUN_NAME | $TASK_NAME | $MODEL_PATH_LAST"
                GPU_TASK_DESC[$gpu]="$TASK_DESC"

                echo "Task: $TASK_NAME, Run: $RUN_NAME, Model: $MODEL_PATH_LAST, GPU: $gpu" > "$LOG_FILE"
                echo "Starting ($((DISPATCHED_TASKS + 1))/$TOTAL_TASKS): $TASK_DESC using GPU $gpu"

                if [[ "$DRY_RUN" == "1" ]]; then
                    (
                        echo "[DRY_RUN] simulate eval: $TASK_DESC" >> "$LOG_FILE"
                        sleep 0.1
                    ) &
                else
                    (
                        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 accelerate launch --num_processes=1 -m lmms_eval \
                            --model "$EVAL_MODEL" \
                            --gen_kwargs "$GEN_KWARGS" \
                            --model_args "$MODEL_ARGS" \
                            --tasks "$TASK_NAME" \
                            --batch_size 1 \
                            --log_samples \
                            --log_samples_suffix "$TASK_NAME" \
                            --output_path "$CURRENT_OUTPUT_PATH" >> "$LOG_FILE" 2>&1
                    ) &
                fi

                GPU_PIDS[$gpu]=$!
                DISPATCHED_TASKS=$((DISPATCHED_TASKS + 1))
            fi
        done
    fi

    if [[ $FINISHED_TASKS -lt $TOTAL_TASKS ]]; then
        sleep 1
    fi
done

wait
if [[ $FAILED_TASKS -gt 0 ]]; then
    echo "Completed with failures: $FAILED_TASKS / $TOTAL_TASKS jobs failed."
    exit 1
fi

echo "All $TOTAL_TASKS jobs completed successfully."
