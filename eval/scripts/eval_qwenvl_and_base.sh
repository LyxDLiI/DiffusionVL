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
#   TASK_NAMES             (default: mmmu_val,mme,chartqa)
#   TOTAL_GPUS             (default: auto-detect)
#   AUTO_DOWNLOAD_MODELS   (default: 1)
#   AUTO_DOWNLOAD_DATASETS (default: 1)
#   SKIP_UNKNOWN_TASKS     (default: 1; when 0, unknown task name exits immediately)
#   AUTO_INSTALL_DEPS      (default: 1; install missing qwen-vl-utils for base model)
#   AUTO_SYNC_TASK_DEFS    (default: 1; auto-sync missing task definitions from upstream lmms-eval)
#   TASK_DEFS_REPO         (default: https://github.com/EvolvingLMMs-Lab/lmms-eval.git)
#   HF_HOME                (default: /data/liuyx/.cache/huggingface)
#   HF_HUB_CACHE           (default: ${HF_HOME}/hub)
#   HF_ENDPOINT            (default: https://hf-mirror.com)
#   HUGGINGFACE_HUB_TOKEN  (default: from current shell env only; not hardcoded in script)
#   HF_TOKEN               (default: ${HUGGINGFACE_HUB_TOKEN})
#   PROXY_URL              (default: empty; only used when ENABLE_PROXY=1)
#   ENABLE_PROXY           (default: 0; when 1, export http/https proxy before downloads)
#   BLOCK_SIZE             (default: 8)
#   STEPS                  (default: 8)
#   DRY_RUN                (default: 0; set 1 to verify scheduler without launching lmms_eval)

set -euo pipefail
# ======================
# 1) HuggingFace env
# ======================
export HF_HOME="/data/liuyx/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
unset HF_ENDPOINT || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-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}"
export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"
: "${HUGGINGFACE_HUB_TOKEN:?Please export HUGGINGFACE_HUB_TOKEN before running.}"

DIFFUSIONVL_MODEL_PATH="${DIFFUSIONVL_MODEL_PATH:-hustvl/DiffusionVL-Qwen2.5VL-7B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/qwenvl_vs_base}"
TASK_NAMES="${TASK_NAMES:-mmmu_val,mme,chartqa}"
TOTAL_GPUS="${TOTAL_GPUS:-}"
AUTO_DOWNLOAD_MODELS="${AUTO_DOWNLOAD_MODELS:-1}"
AUTO_DOWNLOAD_DATASETS="${AUTO_DOWNLOAD_DATASETS:-1}"
SKIP_UNKNOWN_TASKS="${SKIP_UNKNOWN_TASKS:-1}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
AUTO_SYNC_TASK_DEFS="${AUTO_SYNC_TASK_DEFS:-1}"
TASK_DEFS_REPO="${TASK_DEFS_REPO:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
PROXY_URL="${PROXY_URL:-}"
ENABLE_PROXY="${ENABLE_PROXY:-0}"
DRY_RUN="${DRY_RUN:-0}"
HF_HOME="${HF_HOME:-/data/liuyx/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
unset HF_ENDPOINT || true
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-$HUGGINGFACE_HUB_TOKEN}"

BLOCK_SIZE="${BLOCK_SIZE:-8}"
STEPS="${STEPS:-8}"

CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_2_5}"
DIFFUSIONVL_MODEL="llava_onevision_diffusionvl_qwenvl"
BASE_MODEL="qwen2_5_vl"
TASKS_ROOT="${TASKS_ROOT:-./lmms-eval/lmms_eval/tasks}"
DATASETS_LOCAL_ROOT="${DATASETS_LOCAL_ROOT:-./lmms-lab}"
TASK_DEFS_CACHE_DIR="${TASK_DEFS_CACHE_DIR:-./.cache/lmms-eval-task-defs}"

setup_proxy() {
    if [[ "$ENABLE_PROXY" == "1" && -n "$PROXY_URL" ]]; then
        export https_proxy="$PROXY_URL"
        export http_proxy="$PROXY_URL"
        export HTTPS_PROXY="$PROXY_URL"
        export HTTP_PROXY="$PROXY_URL"
        echo "[INFO] Proxy enabled: $PROXY_URL"
    elif [[ "$ENABLE_PROXY" == "1" && -z "$PROXY_URL" ]]; then
        echo "[WARN] ENABLE_PROXY=1 but PROXY_URL is empty; skip exporting proxy."
    fi
}

setup_huggingface_env() {
    export HF_HOME
    export HF_HUB_CACHE
    export HF_ENDPOINT
    if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
        export HUGGINGFACE_HUB_TOKEN
    fi
    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN
    fi
}

setup_proxy
setup_huggingface_env

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

    local cache_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    local cache_repo_dir="$cache_root/models--${model_ref//\//--}"
    if [[ -d "$cache_repo_dir/snapshots" ]]; then
        local cached_snapshot
        cached_snapshot="$(find "$cache_repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)"
        if [[ -n "$cached_snapshot" ]]; then
            echo "[WARN] Local path not found for '$model_ref', but found cached snapshot: $cached_snapshot" >&2
            echo "$cached_snapshot"
            return 0
        fi
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

    if [[ -d "$cache_repo_dir/snapshots" ]]; then
        local cached_snapshot_after_fail
        cached_snapshot_after_fail="$(find "$cache_repo_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)"
        if [[ -n "$cached_snapshot_after_fail" ]]; then
            echo "[WARN] Pre-download failed for '$model_ref'; falling back to cached snapshot: $cached_snapshot_after_fail" >&2
            echo "$cached_snapshot_after_fail"
            return 0
        fi
    fi

    echo "[ERROR] Failed to pre-download '$model_ref'. Check repo id/token/network/cache." >&2
    exit 1
}

trim() {
    local s="$1"
    # shellcheck disable=SC2001
    s="$(echo "$s" | sed 's/^ *//;s/ *$//')"
    echo "$s"
}

find_task_yaml() {
    local task_name="$1"
    if [[ ! -d "$TASKS_ROOT" ]]; then
        echo ""
        return 0
    fi
    local yaml_path
    yaml_path="$(grep -RIl --include='*.yaml' -E "^task:[[:space:]]*\"?${task_name}\"?[[:space:]]*$" "$TASKS_ROOT" | head -n1 || true)"
    echo "$yaml_path"
}

extract_dataset_path_from_yaml() {
    local yaml_path="$1"
    if [[ -z "$yaml_path" || ! -f "$yaml_path" ]]; then
        echo ""
        return 0
    fi
    local dataset_path
    dataset_path="$(grep -E "^dataset_path:" "$yaml_path" | head -n1 | sed -E 's/^dataset_path:[[:space:]]*//; s/[[:space:]]+$//; s/^"//; s/"$//')"
    echo "$dataset_path"
}

task_requires_hf_token() {
    local task_name="$1"
    local yaml_path="$2"

    if [[ -z "$yaml_path" || ! -f "$yaml_path" ]]; then
        return 1
    fi

    # Match task dataset configs that explicitly set token: True
    # in dataset_kwargs.
    python - "$yaml_path" <<'PY'
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

m = re.search(r"dataset_kwargs:\s*(?:\n[ \t]+.*)*", text, flags=re.MULTILINE)
if not m:
    raise SystemExit(1)
block = m.group(0)
if re.search(r"^\s*token:\s*true\s*$", block, flags=re.IGNORECASE | re.MULTILINE):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

ensure_hf_token_for_task() {
    local task_name="$1"
    local yaml_path
    yaml_path="$(find_task_yaml "$task_name")"
    if ! task_requires_hf_token "$task_name" "$yaml_path"; then
        return 0
    fi
    if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] Task '$task_name' requires Hugging Face token (dataset_kwargs.token=True), but neither HUGGINGFACE_HUB_TOKEN nor HF_TOKEN is set." >&2
        echo "[ERROR] Please export token in shell before running." >&2
        exit 1
    fi
}

task_exists_locally() {
    local task_name="$1"
    local yaml_path
    yaml_path="$(find_task_yaml "$task_name")"
    [[ -n "$yaml_path" ]]
}

clone_task_defs_repo() {
    if [[ "$AUTO_SYNC_TASK_DEFS" != "1" ]]; then
        return 1
    fi

    if [[ -d "$TASK_DEFS_CACHE_DIR/lmms_eval/tasks" ]]; then
        return 0
    fi

    mkdir -p "$TASK_DEFS_CACHE_DIR"
    echo "[INFO] Cloning upstream task definitions from: $TASK_DEFS_REPO" >&2
    set +e
    git clone --depth 1 "$TASK_DEFS_REPO" "$TASK_DEFS_CACHE_DIR" >/dev/null 2>&1
    local status=$?
    set -e
    if [[ $status -ne 0 ]]; then
        echo "[WARN] Failed to clone task definitions repo: $TASK_DEFS_REPO" >&2
        return 1
    fi
    return 0
}

sync_task_definition_from_upstream() {
    local task_name="$1"
    if task_exists_locally "$task_name"; then
        return 0
    fi
    if ! clone_task_defs_repo; then
        return 1
    fi

    local remote_tasks_root="$TASK_DEFS_CACHE_DIR/lmms_eval/tasks"
    if [[ ! -d "$remote_tasks_root" ]]; then
        return 1
    fi

    local remote_yaml
    remote_yaml="$(grep -RIl --include='*.yaml' -E "^task:[[:space:]]*\"?${task_name}\"?[[:space:]]*$" "$remote_tasks_root" | head -n1 || true)"
    if [[ -z "$remote_yaml" ]]; then
        return 1
    fi

    local remote_task_dir
    remote_task_dir="$(dirname "$remote_yaml")"
    local task_dir_name
    task_dir_name="$(basename "$remote_task_dir")"
    local local_task_dir="$TASKS_ROOT/$task_dir_name"

    if [[ ! -d "$local_task_dir" ]]; then
        echo "[INFO] Syncing task definition directory: $task_dir_name" >&2
        cp -r "$remote_task_dir" "$local_task_dir"
    fi

    task_exists_locally "$task_name"
}

normalize_task_name() {
    local requested="$1"

    if task_exists_locally "$requested"; then
        echo "$requested"
        return 0
    fi

    # Try syncing exact name first.
    if sync_task_definition_from_upstream "$requested"; then
        echo "$requested"
        return 0
    fi

    # Fallback aliases for commonly renamed tasks across lmms-eval versions.
    local candidates=()
    case "$requested" in
        mathvista)
            candidates=("mathvista_testmini" "mathvista_test" "mathvista_val")
            ;;
        mathvision)
            candidates=("mathvision_test" "mathvision_testmini" "mathvision_val")
            ;;
        mmvp)
            candidates=("mmvp" "mmvp_val")
            ;;
    esac

    local candidate
    for candidate in "${candidates[@]}"; do
        if task_exists_locally "$candidate"; then
            echo "$candidate"
            return 0
        fi
        if sync_task_definition_from_upstream "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done

    echo ""
    return 0
}

ensure_dataset_for_task() {
    local task_name="$1"
    if [[ "$AUTO_DOWNLOAD_DATASETS" != "1" ]]; then
        return 0
    fi

    local yaml_path dataset_path
    yaml_path="$(find_task_yaml "$task_name")"
    dataset_path="$(extract_dataset_path_from_yaml "$yaml_path")"
    if [[ -z "$dataset_path" ]]; then
        return 0
    fi

    # Only handle local-style dataset paths like lmms-lab/MME/data.
    # For standard HF paths like lmms-lab/MMMU, datasets lib can stream/download itself.
    IFS='/' read -r ds_org ds_repo ds_subpath <<< "$dataset_path"
    if [[ -z "$ds_org" || -z "$ds_repo" || -z "$ds_subpath" ]]; then
        return 0
    fi

    local repo_id="${ds_org}/${ds_repo}"
    local local_repo_dir="${DATASETS_LOCAL_ROOT}/${ds_repo}"
    local expected_local_path="${local_repo_dir}/${ds_subpath}"
    if [[ -d "$expected_local_path" ]]; then
        return 0
    fi

    mkdir -p "$local_repo_dir"
    echo "[INFO] Dataset for task '$task_name' is missing: $expected_local_path"
    echo "[INFO] Downloading dataset repo '$repo_id' to '$local_repo_dir' ..."

    set +e
    python - "$repo_id" "$local_repo_dir" <<'PY'
import os
import sys

repo_id = sys.argv[1]
local_dir = sys.argv[2]

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f"[ERROR] huggingface_hub unavailable: {e}", file=sys.stderr)
    raise SystemExit(2)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(local_dir)
PY
    local status=$?
    set -e

    if [[ $status -ne 0 ]]; then
        echo "[ERROR] Failed to download dataset repo '$repo_id' for task '$task_name'." >&2
        return 1
    fi

    if [[ ! -d "$expected_local_path" ]]; then
        echo "[ERROR] Downloaded '$repo_id' but expected path still missing: $expected_local_path" >&2
        return 1
    fi

    echo "[INFO] Dataset ready for task '$task_name': $expected_local_path"
    return 0
}

check_and_install_runtime_deps() {
    if [[ "$AUTO_INSTALL_DEPS" != "1" ]]; then
        return 0
    fi

    ensure_python_module_installed() {
        local import_name="$1"
        local pip_name="$2"
        local hint="$3"

        set +e
        python - "$import_name" <<'PY'
import importlib.util
import sys
module_name = sys.argv[1]
ok = importlib.util.find_spec(module_name) is not None
sys.exit(0 if ok else 1)
PY
        local exists=$?
        set -e
        if [[ $exists -eq 0 ]]; then
            return 0
        fi

        echo "[WARN] Missing dependency '$pip_name'; installing in current environment..."
        set +e
        python -m pip install -U "$pip_name"
        local pip_status=$?
        set -e
        if [[ $pip_status -ne 0 ]]; then
            echo "[WARN] Failed to install '$pip_name' automatically. $hint" >&2
        fi
    }

    ensure_python_module_installed "qwen_vl_utils" "qwen-vl-utils" "Base model may fail with NameError: process_vision_info."
    ensure_python_module_installed "Levenshtein" "python-Levenshtein" "MathVista tasks may fail with ModuleNotFoundError: Levenshtein."
}

fix_diffusionvl_vision_tower_if_needed() {
    local model_dir="$1"
    local vision_tower_dir="$2"
    local cfg="$model_dir/config.json"

    if [[ ! -f "$cfg" ]]; then
        return 0
    fi
    if [[ ! -d "$vision_tower_dir" ]]; then
        echo "[WARN] Base model path is not a local directory; skip DiffusionVL vision tower fix: $vision_tower_dir"
        return 0
    fi

    set +e
    python - "$cfg" "$vision_tower_dir" <<'PY'
import json
import os
import sys

cfg_path = sys.argv[1]
vision_tower = sys.argv[2]

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cur = cfg.get("mm_vision_tower", "")
need_fix = (not cur) or str(cur).startswith("/path/to/") or (str(cur).startswith("/") and not os.path.exists(cur))

if not need_fix:
    raise SystemExit(0)

cfg["mm_vision_tower"] = vision_tower
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[INFO] Patched mm_vision_tower in {cfg_path}: {cur} -> {vision_tower}")
PY
    local status=$?
    set -e

    if [[ $status -ne 0 ]]; then
        return 0
    fi
}

mkdir -p "$OUTPUT_PATH"

DIFFUSIONVL_MODEL_PATH_RESOLVED="$(resolve_model_path "$DIFFUSIONVL_MODEL_PATH")"
BASE_MODEL_PATH_RESOLVED="$(resolve_model_path "$BASE_MODEL_PATH")"

check_and_install_runtime_deps
fix_diffusionvl_vision_tower_if_needed "$DIFFUSIONVL_MODEL_PATH_RESOLVED" "$BASE_MODEL_PATH_RESOLVED"

IFS=',' read -ra TASKS_RAW <<< "$TASK_NAMES"
TASKS=()
for task in "${TASKS_RAW[@]}"; do
    task_trimmed="$(trim "$task")"
    if [[ -n "$task_trimmed" ]]; then
        resolved_task="$(normalize_task_name "$task_trimmed")"
        if [[ -n "$resolved_task" ]]; then
            if [[ "$resolved_task" != "$task_trimmed" ]]; then
                echo "[INFO] Task '$task_trimmed' mapped to '$resolved_task'."
            fi
            TASKS+=("$resolved_task")
        else
            if [[ "$SKIP_UNKNOWN_TASKS" == "1" ]]; then
                echo "[WARN] Skip unknown task '$task_trimmed' (not found locally or upstream sync failed)."
            else
                echo "[ERROR] Unknown task '$task_trimmed' (not found locally or upstream sync failed)." >&2
                exit 1
            fi
        fi
    fi
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "[ERROR] No runnable tasks found in TASK_NAMES='$TASK_NAMES'." >&2
    exit 1
fi

TASK_QUEUE=()
for task in "${TASKS[@]}"; do
    ensure_hf_token_for_task "$task"
    ensure_dataset_for_task "$task"

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
