"""
attention_allocation.py  —  Layer-wise attention-mass allocation  (Figure 5a style).

Captures, per transformer layer, the fraction of attention mass that
instruction/answer tokens assign to each of three token groups:

    visual tokens      [vis_start, vis_end)   — all video-frame patches
    instruction tokens [vis_end,   seq_len)   — question text + role markers after image
    system tokens      [0,         vis_start) — system prompt + role markers before image

The three fractions are normalised to sum=1 per layer and plotted as a
layer-indexed line chart reproducing Figure 5a.

Compatible with both LLaDA-V (video_heatmap.py) and LLaVA-Video
(video_heatmap_llava_video.py) pipelines.  Select via --model_type.

Two modes
---------
single       (default)
    One forward pass on --video_path / --question.  Saves Fig-5a-single plot.

dataset_avg
    Forward passes over all samples of --dataset (videomme | videommmu) or a
    custom --samples_json list, averages per-layer, saves Fig-5a-avg plot.
    Also saves Fig-5a-single for the last sample run (= Figure-5b case by default).

Usage examples
--------------
# LLaDA-V single case (Figure 5b sample — Biology_3 Video-MMMU):
python attention_allocation.py \\
    --video_path /data/.../dev_Biology_3.mp4 \\
    --question   "What will be the name of..." \\
    --model_type llada --model_id GSAI-ML/LLaDA-V \\
    --output_dir ./attn_alloc_out --max_frames_num 32 --gpu_id 1

# LLaDA-V dataset-average (Video-MME):
python attention_allocation.py \\
    --model_type llada --model_id GSAI-ML/LLaDA-V \\
    --mode dataset_avg --dataset videomme \\
    --video_dir /data/liuyx/.cache/huggingface/videomme/data \\
    --output_dir ./attn_alloc_out --max_frames_num 32 --gpu_id 1
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared pipeline utilities  (loaded lazily to avoid import-time side effects)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _llada_imports():
    """Import shared functions from video_heatmap.py (LLaDA-V pipeline)."""
    from video_heatmap import (
        load_model,
        load_video,
        build_prompt_input,
        compute_visual_token_layout,
    )
    return load_model, load_video, build_prompt_input, compute_visual_token_layout


def _llava_imports():
    """Import shared functions from video_heatmap_llava_video.py (LLaVA-Video pipeline)."""
    from video_heatmap_llava_video import (
        load_model,
        load_video,
        build_prompt_input,
        compute_visual_token_layout,
    )
    return load_model, load_video, build_prompt_input, compute_visual_token_layout


# ---------------------------------------------------------------------------
# 1.  Allocation hooks
#     Two variants differing only in WHERE they attach:
#       LLaDA-V     → decoder layer   (output = hidden, attn, past_kv)
#       LLaVA-Video → self_attn module (output = attn_out, attn_weights)
# ---------------------------------------------------------------------------

def _make_alloc_hook(layer_idx, vis_start, vis_end, alloc_dict):
    """
    Return a forward-hook closure that:
      • Uses instruction-token rows [vis_end, S) as the query perspective.
        (For bidirectional LLaDA these are question text + role markers after the image;
         for causal LLaVA they are the same suffix positions.)
      • Sums attention mass over three column regions:
            visual      [vis_start, vis_end)
            instruction [vis_end,   S)
            system      [0,         vis_start)
      • Normalises the three sums to sum=1 and stores in alloc_dict[layer_idx].
      • Replaces attn_weights with None in the output tuple to free GPU memory.
    """
    def _hook(module, _input, output):
        # Locate the attention-weights tensor in the output tuple.
        # Decoder layer:  (hidden, attn_weights, past_kv, ...)  — index 1
        # self_attn:      (attn_out, attn_weights)               — index 1
        if not (isinstance(output, tuple) and len(output) >= 2
                and output[1] is not None):
            return

        w = output[1]          # (1, H, S, S)
        S = w.shape[-1]

        # ── Query slice: instruction token rows [vis_end, S) ──────────────
        # These are the positions AFTER the visual region (question text,
        # role markers, and masked answer tokens for LLaDA).
        if vis_end < S:
            q = w[0, :, vis_end:S, :]   # (H, q_len, S)
        else:
            # Fallback when the full sequence is inside the visual region
            q = w[0, :, :, :]           # (H, S, S)

        # Average over heads and query positions → (S,)  attention distribution
        avg = q.float().mean(dim=(0, 1)).cpu().numpy()

        if np.isnan(avg).any() or np.isinf(avg).any():
            return  # skip numerically degenerate layers

        # ── Sum attention mass in each group ──────────────────────────────
        m_sys = float(avg[:vis_start].sum())
        m_vis = float(avg[vis_start:vis_end].sum())
        m_ins = float(avg[vis_end:S].sum())

        total = m_sys + m_vis + m_ins
        if total > 1e-10:
            # Store as [visual_frac, instruction_frac, system_frac]
            alloc_dict[layer_idx] = np.array(
                [m_vis / total, m_ins / total, m_sys / total], dtype=np.float32
            )

        # Free the full attention matrix to keep GPU memory constant per layer
        return (output[0], None) + output[2:]

    return _hook


def register_allocation_hooks_llada(model, vis_start, vis_end, alloc_dict):
    """Register hooks on each LLaDADecoderLayer (output[1] = attn_weights)."""
    handles = []
    for li, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(_make_alloc_hook(li, vis_start, vis_end, alloc_dict))
        handles.append(h)
    return handles


def register_allocation_hooks_llava(model, vis_start, vis_end, alloc_dict):
    """Register hooks on each Qwen2Attention.self_attn (output[1] = attn_weights).

    Necessary because Qwen2DecoderLayer.forward() discards attn_weights with `_ =`.
    """
    handles = []
    for li, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(
            _make_alloc_hook(li, vis_start, vis_end, alloc_dict)
        )
        handles.append(h)
    return handles


# ---------------------------------------------------------------------------
# 2.  Single forward pass  →  per-layer allocation dict
# ---------------------------------------------------------------------------

def get_allocation_llada(
    model, tokenizer, image_processor,
    frames_np, question,
    max_frames_num=32, mm_spatial_pool_stride=2, think_mode="no_think",
):
    """
    LLaDA-V forward pass.
    Returns  alloc_dict : {layer_idx: np.array([vis_frac, instr_frac, sys_frac])}.
    """
    _, _, build_prompt_input, compute_visual_token_layout = _llada_imports()
    device = "cuda:0"

    model.config.mm_spatial_pool_stride = mm_spatial_pool_stride
    model.config.mm_spatial_pool_mode   = "bilinear"

    img = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
    img = img.half().to(device)

    input_ids, IMG_IDX = build_prompt_input(
        tokenizer, question, think_mode=think_mode, device=device
    )
    pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id
    attn_mask = input_ids.ne(pad_id).to(device)

    vis_start, vis_end, _, _ = compute_visual_token_layout(
        model, input_ids, IMG_IDX, max_frames_num, stride=mm_spatial_pool_stride
    )
    print(f"  [alloc] vis_start={vis_start}  vis_end={vis_end}  "
          f"approx_seq={input_ids.shape[1] - 1 + (vis_end - vis_start)}")

    alloc_dict = {}
    handles    = register_allocation_hooks_llada(model, vis_start, vis_end, alloc_dict)
    try:
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=input_ids.clone(),   # dummy labels required by LLaDA-V
                images=[img],
                modalities=["video"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    return alloc_dict


def get_allocation_llava(
    model, tokenizer, image_processor,
    frames_np, question,
    max_frames_num=32, mm_spatial_pool_stride=2,
    conv_template="qwen_1_5",
):
    """
    LLaVA-Video forward pass.
    Returns  alloc_dict : {layer_idx: np.array([vis_frac, instr_frac, sys_frac])}.
    """
    _, _, build_prompt_input, compute_visual_token_layout = _llava_imports()
    device = "cuda:0"

    model.config.mm_spatial_pool_stride = mm_spatial_pool_stride
    model.config.mm_spatial_pool_mode   = "average"

    img = image_processor.preprocess(frames_np, return_tensors="pt")["pixel_values"]
    img = img.half().to(device)

    input_ids, IMG_IDX = build_prompt_input(
        tokenizer, question, conv_template=conv_template, device=device,
        pre_prompt="", post_prompt="",   # raw question; no VideoMME wrapper
    )
    pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id
    attn_mask = input_ids.ne(pad_id).to(device)

    vis_start, vis_end, _, _ = compute_visual_token_layout(
        model, input_ids, IMG_IDX, max_frames_num, stride=mm_spatial_pool_stride
    )
    print(f"  [alloc] vis_start={vis_start}  vis_end={vis_end}  "
          f"approx_seq={input_ids.shape[1] - 1 + (vis_end - vis_start)}")

    alloc_dict = {}
    handles    = register_allocation_hooks_llava(model, vis_start, vis_end, alloc_dict)
    try:
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[img],
                modalities=["video"],
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    return alloc_dict


# ---------------------------------------------------------------------------
# 3.  Average allocation over multiple forward passes
# ---------------------------------------------------------------------------

def average_allocations(alloc_list):
    """
    Given a list of alloc_dicts (one per sample), return a single alloc_dict
    where each layer value is the mean of all per-sample arrays at that layer.
    Layers missing in some samples are excluded from the average for that layer.
    """
    from collections import defaultdict
    accum = defaultdict(list)
    for alloc in alloc_list:
        for li, v in alloc.items():
            accum[li].append(v)
    return {li: np.stack(vals).mean(axis=0) for li, vals in accum.items()}


# ---------------------------------------------------------------------------
# 4.  Figure 5a style plot  —  stacked bar chart
# ---------------------------------------------------------------------------

def plot_allocation(alloc_dict, title, save_path, rect_top=0.20):
    """
    Stacked bar chart of layer-wise normalised attention mass — Figure 5a style.

    Bottom to top per bar:
        pink   (#F4AFAB) — Visual Tokens
        blue   (#9DC3E6) — Instruction Tokens
        orange (#F4B942) — System Tokens

    alloc_dict : {layer_idx (int, 0-based): np.array([vis_frac, instr_frac, sys_frac])}
    """

    layers = sorted(alloc_dict.keys())
    if not layers:
        print(f"[plot] nothing to plot — {save_path}")
        return

    arr  = np.stack([alloc_dict[l] for l in layers])   # (L, 3)
    vis  = arr[:, 0]
    ins  = arr[:, 1]
    sys_ = arr[:, 2]

    x = np.array(layers, dtype=float)   # 0-indexed layer numbers, matching paper

    fig, ax = plt.subplots(figsize=(9, 4))

    # Stacked bars: visual (bottom), instruction (middle), system (top)
    ax.bar(x, vis,  color="#F4AFAB", width=0.8, label="Visual Tokens")
    ax.bar(x, ins,  color="#9DC3E6", width=0.8, label="Instruction Tokens", bottom=vis)
    ax.bar(x, sys_, color="#F4B942", width=0.8, label="System Tokens",      bottom=vis + ins)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attention", fontsize=12)

    # X-ticks every 3 layers, 0-indexed — matches paper (0, 3, 6, …, 27)
    max_layer = int(x[-1])
    ax.set_xticks(np.arange(0, max_layer + 1, 3))
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, 1.0)

    # Horizontal legend above the chart, three columns — matches paper legend box
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        fontsize=10,
        frameon=True,
        handlelength=1.5,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")


# ---------------------------------------------------------------------------
# 6.  Built-in sample lists  (mirrors run_heatmaps_videomme.sh / _videommmu.sh)
# ---------------------------------------------------------------------------

def _videomme_samples(video_dir):
    """Four Video-MME samples matching run_heatmaps_videomme.sh."""
    return [
        {
            "id": "001-3",
            "video_path": f"{video_dir}/fFjv93ACGo8.mp4",
            "question": (
                "How many red socks are above the fireplace at the end of this video?\n"
                "A. 1.\nB. 4.\nC. 2.\nD. 3."
            ),
        },
        {
            "id": "008-2",
            "video_path": f"{video_dir}/_tvmjsKXTu8.mp4",
            "question": (
                "How many different guitar-shaped instruments are there in the video?\n"
                "A. 2.\nB. 7.\nC. 3.\nD. 11."
            ),
        },
        {
            "id": "602-3",
            "video_path": f"{video_dir}/w0Wmc8C0Eq0.mp4",
            "question": (
                "In line with the video evidence, which of the following statements "
                "about the world's longest railway line is not correct?\n"
                "A. It was built by Russia and China.\n"
                "B. The length of it is 9289km.\n"
                "C. It was completed in 1916.\n"
                "D. French loans helped a lot in the process of building it."
            ),
        },
        {
            "id": "603-2",
            "video_path": f"{video_dir}/7D-gxaie6UI.mp4",
            "question": (
                "What was the event that put an end to the romanticization of TB in the video?\n"
                "A. The prevalence of TB in colonial territories.\n"
                "B. TB spread to the working class.\n"
                "C. Mycobacterium tuberculosis has a thick cell wall that makes it resistant "
                "to infection-fighting cells.\n"
                "D. The course of the disease can be unpredictable, causing death within a "
                "few weeks or over many years."
            ),
        },
    ]


def _videommmu_samples(video_dir):
    """Four Video-MMMU samples matching run_heatmaps_videommmu.sh."""
    return [
        {
            "id": "Chemistry_17",
            "video_path": f"{video_dir}/validation_Chemistry_17.mp4",
            "question": (
                "You should watch and learn the video content. Then apply what you learned "
                "to answer the following multi-choice question. The image for this question "
                "is at the end of the video.\n"
                "Among the following, the Newmann projections of meso-2, 3-butanediol are : "
                "<image 1>\n"
                "A. P, Q\nB. P, R\nC. R, S\nD. Q, S\nE. P, S\nF. Q, R\n"
                "G. R, Q\nH. S, P\nI. S, Q\nJ. R, P"
            ),
        },
        {
            "id": "Biology_3",
            "video_path": f"{video_dir}/dev_Biology_3.mp4",
            "question": (
                "What will be the name of the two compounds shown in the video if they both "
                "have one more carbon atom?\n"
                "A. left: Aldotetrose, right: Ketotetrose\n"
                "B. left: Aldotetrose, right: Ketopentose\n"
                "C. left: Aldotetrose, right: Ketohexose\n"
                "D. left: Aldopentose, right: Ketotetrose\n"
                "E. left: Aldopentose, right: Ketopentose\n"
                "F. left: Aldopentose, right: Ketohexose\n"
                "G. left: Aldohexose, right: Ketotetrose\n"
                "H. left: Aldohexose, right: Ketopentose\n"
                "I. left: Aldohexose, right: Ketohexose\n"
                "J. left: Aldotriose, right: Ketotetrose\n"
                "Please ignore the Quiz question in last frame of the video."
            ),
        },
        {
            "id": "Chemistry_9",
            "video_path": f"{video_dir}/validation_Chemistry_9.mp4",
            "question": (
                "You should watch and learn the video content. Then apply what you learned "
                "to answer the following multi-choice question. The image for this question "
                "is at the end of the video.\n"
                "When an unsymmetrical alkene such as propene is treated with N-bromosuccinimide "
                "in aqueous dimethyl sulfoxide, the major product has the bromine atom bonded to "
                "the less highly substituted carbon atom. Is the following one Markovnikov or "
                "non-Markovnikov orientation? <image 1>\n"
                "A. Markovnikov orientation.\n"
                "B. Non-Markovnikov orientation.\n"
                "C. None of the above."
            ),
        },
        {
            "id": "Chemistry_19",
            "video_path": f"{video_dir}/validation_Chemistry_19.mp4",
            "question": (
                "Consider the two molecules that appeared in the video from 0:28 to 2:00, "
                "if we change the positions of the two green atoms in the second molecule, "
                "will the two isomers now become Optical Isomerism?\n"
                "A. Yes, as they cannot realign with each other by rotation\n"
                "B. Yes, as the molecular formula changes after swapping\n"
                "C. No, as the bonds will break during the swap\n"
                "D. Yes, as it creates new asymmetric centers\n"
                "E. No, as they will become two same molecules\n"
                "F. Yes, as it changes the overall symmetry of the molecule\n"
                "G. No, as the molecule will become unstable\n"
                "H. Yes, as it creates mirror videos that cannot superimpose\n"
                "I. No, as green atoms have special bonding properties\n"
                "J. Yes, as it changes the molecular weight distribution\n"
                "Please ignore the Quiz question in last frame of the video."
            ),
        },
    ]


def _load_samples_json(path):
    """Load sample list from a JSON file.  Each entry must have video_path + question."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 7.  Single-case and dataset-average runners
# ---------------------------------------------------------------------------

def run_single(args, model, tokenizer, image_processor):
    """Run one forward pass; return alloc_dict and save single-case Figure 5a."""
    load_video_fn = (_llada_imports if args.model_type == "llada" else _llava_imports)()[1]
    frames_np, _ = load_video_fn(args.video_path, args.max_frames_num)
    print(f"  [video] {frames_np.shape}  →  {args.max_frames_num} frames")

    if args.model_type == "llada":
        alloc = get_allocation_llada(
            model, tokenizer, image_processor, frames_np, args.question,
            max_frames_num=args.max_frames_num,
            mm_spatial_pool_stride=args.mm_spatial_pool_stride,
            think_mode=args.think_mode,
        )
    else:
        alloc = get_allocation_llava(
            model, tokenizer, image_processor, frames_np, args.question,
            max_frames_num=args.max_frames_num,
            mm_spatial_pool_stride=args.mm_spatial_pool_stride,
        )

    sample_id = os.path.splitext(os.path.basename(args.video_path))[0]
    title     = f"{args.model_type.upper()}  {sample_id}"
    save_path = os.path.join(args.output_dir, "fig5a_single.png")
    plot_allocation(alloc, title, save_path, rect_top=args.rect_top)

    # Save raw allocation as JSON for reproducibility
    alloc_json = {str(k): v.tolist() for k, v in alloc.items()}
    with open(os.path.join(args.output_dir, "alloc_single.json"), "w") as f:
        json.dump(alloc_json, f, indent=2)

    return alloc


def run_dataset_avg(args, model, tokenizer, image_processor):
    """Loop over all dataset samples; average per layer; save dataset-average Figure 5a."""
    # ── Collect sample list ──────────────────────────────────────────────────
    if args.samples_json:
        samples = _load_samples_json(args.samples_json)
    elif args.dataset == "videomme":
        samples = _videomme_samples(args.video_dir)
    elif args.dataset == "videommmu":
        samples = _videommmu_samples(args.video_dir)
    else:
        raise ValueError(f"Unknown --dataset '{args.dataset}'.  Use videomme / videommmu.")

    load_video_fn = (_llada_imports if args.model_type == "llada" else _llava_imports)()[1]

    alloc_list = []

    for i, s in enumerate(samples):
        vpath = s["video_path"]
        q     = s["question"]
        sid   = s.get("id", os.path.splitext(os.path.basename(vpath))[0])
        print(f"\n[{i+1}/{len(samples)}] Sample: {sid}")

        if not os.path.exists(vpath):
            print(f"  WARNING: video not found, skipping — {vpath}")
            continue

        frames_np, _ = load_video_fn(vpath, args.max_frames_num)

        if args.model_type == "llada":
            alloc = get_allocation_llada(
                model, tokenizer, image_processor, frames_np, q,
                max_frames_num=args.max_frames_num,
                mm_spatial_pool_stride=args.mm_spatial_pool_stride,
                think_mode=args.think_mode,
            )
        else:
            alloc = get_allocation_llava(
                model, tokenizer, image_processor, frames_np, q,
                max_frames_num=args.max_frames_num,
                mm_spatial_pool_stride=args.mm_spatial_pool_stride,
            )

        # ── Save per-sample Figure 5a immediately after each forward pass ──
        sample_title = f"{args.model_type.upper()}  {sid}"
        sample_save  = os.path.join(args.output_dir, f"fig5a_{sid}.png")
        plot_allocation(alloc, sample_title, sample_save, rect_top=args.rect_top)

        # Save per-sample raw allocation JSON
        alloc_json = {str(k): v.tolist() for k, v in alloc.items()}
        with open(os.path.join(args.output_dir, f"alloc_{sid}.json"), "w") as f:
            json.dump(alloc_json, f, indent=2)

        alloc_list.append(alloc)

    if not alloc_list:
        print("[dataset_avg] No valid samples — aborting.")
        return

    # ── Average and save dataset-average Figure 5a ───────────────────────────
    avg_alloc = average_allocations(alloc_list)
    n         = len(alloc_list)
    dataset_label = args.dataset if not args.samples_json else "custom"
    title_a   = f"{args.model_type.upper()}  {dataset_label}  (avg {n} samples)"
    plot_allocation(
        avg_alloc, title_a,
        os.path.join(args.output_dir, "fig5a_dataset_avg.png"),
        rect_top=args.rect_top,
    )

    # Save raw averaged allocation
    avg_json = {str(k): v.tolist() for k, v in avg_alloc.items()}
    with open(os.path.join(args.output_dir, "alloc_dataset_avg.json"), "w") as f:
        json.dump(avg_json, f, indent=2)

    return avg_alloc


# ---------------------------------------------------------------------------
# 8.  Comparison note
# ---------------------------------------------------------------------------

def print_comparison(alloc_single, alloc_avg):
    """
    Print a short comparison note: which plot looks closer to Figure 5a in the paper.
    Heuristic: Figure 5a in 'More thinking, Less Seeing' shows that visual-token
    attention DECREASES across layers while instruction/system attention INCREASES.
    We measure the slope of visual-token attention to judge alignment.
    """
    def _slope(alloc):
        layers = sorted(alloc.keys())
        if len(layers) < 2:
            return 0.0
        vis = np.array([alloc[l][0] for l in layers])
        x   = np.arange(len(vis), dtype=float)
        return float(np.polyfit(x, vis, 1)[0])

    slope_s = _slope(alloc_single)
    slope_a = _slope(alloc_avg) if alloc_avg else float("nan")

    print("\n" + "=" * 60)
    print("COMPARISON NOTE  (Figure 5a alignment)")
    print("=" * 60)
    print(f"  Single-case  visual-attn slope: {slope_s:+.5f}")
    if not np.isnan(slope_a):
        print(f"  Dataset-avg  visual-attn slope: {slope_a:+.5f}")
    print()
    print("  Paper Figure 5a shows visual-token attention steadily DECLINING")
    print("  across layers (negative slope) while instruction/system tokens")
    print("  absorb more mass in deeper layers.")
    if slope_s < -1e-4:
        closer = "single-case"
    elif not np.isnan(slope_a) and slope_a < -1e-4:
        closer = "dataset-average"
    else:
        closer = "neither clearly matches"
    print(f"  → '{closer}' shows declining visual attention, closer to the paper.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Layer-wise attention-mass allocation plot (Figure 5a)."
    )
    # Core
    p.add_argument("--mode",          type=str, default="single",
                   choices=["single", "dataset_avg"])
    p.add_argument("--model_type",    type=str, default="llada",
                   choices=["llada", "llava"],
                   help="llada = LLaDA-V (video_heatmap.py); "
                        "llava = LLaVA-Video (video_heatmap_llava_video.py)")
    p.add_argument("--model_id",      type=str, default="GSAI-ML/LLaDA-V")
    p.add_argument("--output_dir",    type=str, default="./attn_alloc_out")
    p.add_argument("--gpu_id",        type=int, default=1)
    p.add_argument("--max_frames_num",type=int, default=32)
    p.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    p.add_argument("--rect_top",      type=float, default=0.20,
                   help="Height of the red bounding-box that highlights visual tokens (default 0.20).")

    # Single mode
    p.add_argument("--video_path",    type=str, default="",
                   help="(single mode) Path to the video file.")
    p.add_argument("--question",      type=str, default="",
                   help="(single mode) Question string.")
    p.add_argument("--think_mode",    type=str, default="no_think",
                   choices=["no_think", "think", "none"],
                   help="LLaDA-V think mode suffix.")

    # Dataset-average mode
    p.add_argument("--dataset",       type=str, default="videommmu",
                   choices=["videomme", "videommmu"],
                   help="(dataset_avg) Built-in dataset to average over.")
    p.add_argument("--video_dir",     type=str,
                   default="/data/liuyx/.cache/huggingface/video_mmmu/Science",
                   help="(dataset_avg) Root directory containing the video files.")
    p.add_argument("--samples_json",  type=str, default="",
                   help="(dataset_avg) Optional JSON file with custom sample list. "
                        "Overrides --dataset.")

    args = p.parse_args()

    # Must set CUDA_VISIBLE_DEVICES BEFORE any CUDA init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"[1/3] Loading {args.model_type.upper()} model: {args.model_id}  (GPU {args.gpu_id})")
    if args.model_type == "llada":
        load_model_fn = _llada_imports()[0]
    else:
        load_model_fn = _llava_imports()[0]
    tokenizer, model, image_processor = load_model_fn(args.model_id, args.gpu_id)

    # ── Run mode ─────────────────────────────────────────────────────────────
    alloc_single = None
    alloc_avg    = None

    if args.mode == "single":
        if not args.video_path or not args.question:
            p.error("--mode single requires --video_path and --question")
        print(f"[2/3] Single-case forward pass ...")
        alloc_single = run_single(args, model, tokenizer, image_processor)

    else:  # dataset_avg
        print(f"[2/3] Dataset-average mode: {args.dataset} ...")
        alloc_avg = run_dataset_avg(args, model, tokenizer, image_processor)

        # Use the dataset average as the reference for the comparison note.
        alloc_single = alloc_avg

    # ── Comparison note ──────────────────────────────────────────────────────
    print("[3/3] Printing comparison note ...")
    print_comparison(
        alloc_single if alloc_single is not None else {},
        alloc_avg,
    )

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
