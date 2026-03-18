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

The analysis script can read both the trajectory JSONL and the `lmms_eval` sample log. If `--predictions-jsonl` is omitted, it will try to infer the `*_samples_mme.jsonl` file from the run directory.

```bash
python eval/mme_trajectory/analyze_mme_trajectory.py \
  --trajectory-jsonl ./eval/mme_trajectory_runs/run1/trajectory/mme_trajectories.jsonl \
  --predictions-jsonl ./eval/mme_trajectory_runs/run1/20260318_samples_mme.jsonl
```

## Metrics

- sample-level final-pass / ever-pass / vote accuracy
- pair-level final / ever / vote accuracy grouped by MME `question_id`
- official-style MME pairwise total/category scores for final, ever, and vote answers
- average flip count and flip-pattern breakdowns
- earliest correct step and stable convergence step
- final-wrong subset analysis
- offline early-commit simulation via confidence-gap thresholds
