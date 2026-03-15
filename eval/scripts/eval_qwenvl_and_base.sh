#!/bin/bash

# Evaluate DiffusionVL-QwenVL and Qwen2.5-VL base model on the same task list.

#   TASK_NAMES="mmmu_val,mme,mmvp,mathvision,mathvista" \
#   TOTAL_GPUS=8 \
#   bash scripts/eval_qwenvl_and_base.sh

set -euo pipefail


OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/qwenvl_vs_base}"
TASK_NAMES="${TASK_NAMES:-mmmu_val,mme,mmvp,mathvision,mathvista}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

# DiffusionVL BD3-LM arguments
BLOCK_SIZE="${BLOCK_SIZE:-8}"
STEPS="${STEPS:-8}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2_5}"
DIFFUSIONVL_MODEL="llava_onevision_diffusionvl_qwenvl"
BASE_MODEL="qwen2_5_vl"


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
