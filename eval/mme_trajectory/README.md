# MME Trajectory Analysis

This folder adds an MME-only, minimally invasive trajectory-analysis workflow on top of the existing DiffusionVL evaluation path.

## What it does

- Reuses the same `lmms_eval` MME evaluation route as `eval/scripts/eval_qwenvl_and_base.sh`.
- Runs only `llava_onevision_diffusionvl_qwenvl` on only `mme`.
- Monkey-patches the DiffusionVL BD3-LM generation loop at runtime instead of editing original repo files.
- Writes:
  - normal `lmms_eval` prediction/sample logs,
  - `trajectory/mme_trajectories.jsonl`,
  - `run_metadata.json`.

## Run

```bash
python eval/mme_trajectory/run_mme_trajectory.py \
  --model-path /path/to/DiffusionVL-Qwen2.5VL-7B \
  --output-dir ./eval/mme_trajectory_runs/run1 \
  --steps 8 \
  --gen-length 128 \
  --block-size 8 \
  --remasking-strategy low_confidence_static \
  --trajectory
```

## Analyze

```bash
python eval/mme_trajectory/analyze_mme_trajectory.py \
  --trajectory-jsonl ./eval/mme_trajectory_runs/run1/trajectory/mme_trajectories.jsonl
```

## Metrics

- final-pass accuracy
- ever-pass accuracy
- ever-final gap
- trajectory voting accuracy
- average flip count
- earliest correct step
- stable convergence step
- offline early-commit simulation via confidence-gap thresholds
