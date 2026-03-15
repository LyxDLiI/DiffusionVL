#!/bin/bash

# Evaluate DiffusionVL-QwenVL and Qwen2.5-VL base model on the same task list.
# If model paths are not local directories, this script will try to download
# from Hugging Face Hub into the local cache automatically.
#
# Usage:
#   cd eval
#   DIFFUSIONVL_MODEL_PATH=hustvl/DiffusionVL-Qwen2.5VL-7B \
#   BASE_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
#   TASK_NAMES="mmmu_val,mme,mmvp,mathvision,mathvista" \
#   TOTAL_GPUS=8 \
#   bash scripts/eval_qwenvl_and_base.sh

set -euo pipefail

DIFFUSIONVL_MODEL_PATH="${DIFFUSIONVL_MODEL_PATH:-hustvl/DiffusionVL-Qwen2.5VL-7B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/qwenvl_vs_base}"
TASK_NAMES="${TASK_NAMES:-mmmu_val,mme,mmvp,mathvision,mathvista}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

# DiffusionVL BD3-LM arguments
BLOCK_SIZE="${BLOCK_SIZE:-8}"
STEPS="${STEPS:-8}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2_5}"
DIFFUSIONVL_MODEL="llava_onevision_diffusionvl_qwenvl"
BASE_MODEL="qwen2_5_vl"

resolve_model_path() {
    local model_ref="$1"

    if [[ -d "$model_ref" ]]; then
        echo "$model_ref"
        return 0
    fi

    echo "[INFO] Local path not found for '$model_ref'. Trying Hugging Face Hub download..." >&2

    python - "$model_ref" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_ref = sys.argv[1]

try:
    local_path = snapshot_download(repo_id=model_ref)
except Exception as e:
    print(f"[ERROR] Failed to download '{model_ref}' from Hugging Face Hub: {e}", file=sys.stderr)
    print("[ERROR] Provide an existing local path or a valid Hugging Face repo id.", file=sys.stderr)
    raise SystemExit(1)

print(local_path)
PY
}

mkdir -p "$OUTPUT_PATH"

# Resolve model refs (local path or HF repo id) to usable local checkpoint paths.
DIFFUSIONVL_MODEL_PATH_RESOLVED="$(resolve_model_path "$DIFFUSIONVL_MODEL_PATH")"
BASE_MODEL_PATH_RESOLVED="$(resolve_model_path "$BASE_MODEL_PATH")"

IFS=',' read -ra TASKS <<< "$TASK_NAMES"

# queue item format:
# eval_model|model_path|task|gen_kwargs|model_args|run_name
TASK_QUEUE=()

for task in "${TASKS[@]}"; do
    # DiffusionVL generation settings
    if [[ "$task" == "chartqa" ]]; then
        DIFF_GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"stopping_criteria\":[\"\\n\"], \"remasking_strategy\": \"low_confidence_static\"}"
    else
        DIFF_GEN_KWARGS="{\"temperature\":0, \"gen_length\":128, \"steps\":$STEPS, \"max_new_tokens\":128, \"remasking_strategy\": \"low_confidence_static\"}"
    fi
    DIFF_MODEL_ARGS="pretrained=$DIFFUSIONVL_MODEL_PATH_RESOLVED,conv_template=$CONV_TEMPLATE,model_name=diffusionvl_qwenvl,enable_bd3lm=True,bd3lm_block_size=$BLOCK_SIZE"
    TASK_QUEUE+=("$DIFFUSIONVL_MODEL|$DIFFUSIONVL_MODEL_PATH_RESOLVED|$task|$DIFF_GEN_KWARGS|$DIFF_MODEL_ARGS|DiffusionVL-QwenVL")

    # Base model generation settings (no BD3-LM args)
    BASE_GEN_KWARGS='{"temperature":0,"max_new_tokens":128}'
    BASE_MODEL_ARGS="pretrained=$BASE_MODEL_PATH_RESOLVED"
    TASK_QUEUE+=("$BASE_MODEL|$BASE_MODEL_PATH_RESOLVED|$task|$BASE_GEN_KWARGS|$BASE_MODEL_ARGS|Qwen2.5-VL-Base")
done

TOTAL_TASKS=${#TASK_QUEUE[@]}
DISPATCHED_TASKS=0
FINISHED_TASKS=0

declare -A GPU_STATUS
declare -A GPU_PIDS
for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
    GPU_STATUS[$gpu]=0
done

echo "=========================================="
echo "DiffusionVL model ref: $DIFFUSIONVL_MODEL_PATH"
echo "DiffusionVL model path: $DIFFUSIONVL_MODEL_PATH_RESOLVED"
echo "Base model ref: $BASE_MODEL_PATH"
echo "Base model path: $BASE_MODEL_PATH_RESOLVED"
echo "Output path: $OUTPUT_PATH"
echo "Tasks: $TASK_NAMES"
echo "GPUs: $TOTAL_GPUS"
echo "BD3-LM Block Size: $BLOCK_SIZE"
echo "BD3-LM Steps: $STEPS"
echo "Total jobs: $TOTAL_TASKS"
echo "=========================================="

while [[ $FINISHED_TASKS -lt $TOTAL_TASKS ]]; do
    for ((gpu=0; gpu<TOTAL_GPUS; gpu++)); do
        if [[ ${GPU_STATUS[$gpu]} -eq 1 && -n "${GPU_PIDS[$gpu]:-}" ]]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                GPU_STATUS[$gpu]=0
                unset GPU_PIDS[$gpu]
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

                echo "Task: $TASK_NAME, Run: $RUN_NAME, Model: $MODEL_PATH_LAST, GPU: $gpu" > "$LOG_FILE"
                echo "Starting ($((DISPATCHED_TASKS + 1))/$TOTAL_TASKS): $RUN_NAME on $TASK_NAME using GPU $gpu"

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

                GPU_PIDS[$gpu]=$!
                DISPATCHED_TASKS=$((DISPATCHED_TASKS + 1))
            fi
        done
    fi

    if [[ $FINISHED_TASKS -lt $TOTAL_TASKS ]]; then
        sleep 10
    fi
done

wait
echo "All $TOTAL_TASKS jobs completed."
