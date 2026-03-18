"""
Average layer-wise attention allocation for image VQA tasks on:
  - Qwen2.5-VL base model
  - DiffusionVL-QwenVL model

Token groups (Figure-5a style):
  - visual tokens
  - instruction tokens
  - system tokens

Supported datasets:
  - mme
  - mmvp
  - mmmu (single-image subset from validation split)
"""

import argparse
import ast
import copy
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from attention_allocation import plot_allocation

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _avg_alloc(alloc_list):
    acc = defaultdict(list)
    for alloc in alloc_list:
        for li, v in alloc.items():
            acc[li].append(v)
    return {li: np.stack(vals).mean(axis=0) for li, vals in acc.items()}


def _parse_mmvp_options(options_str):
    pattern = r"\([a-z]\)\s*([^(]+?)(?=\s*\([a-z]\)|$)"
    opts = re.findall(pattern, options_str, re.IGNORECASE)
    return [x.strip() for x in opts]


def _format_mme(doc):
    q = doc["question"].replace(" Please answer yes or no.", "").strip()
    q = f"{q}\nAnswer the question using a single word or phrase."
    return doc["image"].convert("RGB"), q, doc["question_id"]


def _format_mmvp(doc):
    opts = _parse_mmvp_options(doc["Options"])
    opts_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(opts)])
    q = f'{doc["Question"]}\n{opts_text}\nAnswer with the option\'s letter from the given choices directly.'
    sid = f'mmvp_{int(doc["Index"]):04d}'
    return doc["image"].convert("RGB"), q, sid


def _format_mmmu(doc):
    # Keep only single-image subset.
    for idx in range(2, 8):
        if doc.get(f"image_{idx}") is not None:
            return None

    image = doc["image_1"]
    if image is None:
        return None

    q = re.sub(r"<image\s*\d+>", "", doc["question"]).strip()
    opts = doc["options"]
    if isinstance(opts, str):
        try:
            opts = ast.literal_eval(opts)
        except Exception:
            opts = []
    if opts:
        q += "\n" + "\n".join([f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)])
    q += "\n\nAnswer with the option's letter from the given choices directly."
    return image.convert("RGB"), q, doc["id"]


def _load_samples(dataset_name, num_samples, seed):
    random.seed(seed)
    samples = []

    if dataset_name == "mme":
        ds = load_dataset("lmms-lab/MME", data_dir="data", split="test")
        for doc in ds:
            image, question, sid = _format_mme(doc)
            samples.append({"id": sid, "image": image, "question": question})

    elif dataset_name == "mmvp":
        ds = load_dataset("lmms-lab-eval/MMVP", split="train")
        for doc in ds:
            image, question, sid = _format_mmvp(doc)
            samples.append({"id": sid, "image": image, "question": question})

    elif dataset_name == "mmmu":
        ds = load_dataset("lmms-lab/MMMU", split="validation")
        for doc in ds:
            one = _format_mmmu(doc)
            if one is not None:
                image, question, sid = one
                samples.append({"id": sid, "image": image, "question": question})
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_samples > 0 and len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    return samples


def _compute_alloc_from_attn(attn_tensor, vis_start, vis_end):
    seq_len = attn_tensor.shape[-1]
    if vis_end >= seq_len:
        query_slice = attn_tensor[0, :, :, :]
    else:
        query_slice = attn_tensor[0, :, vis_end:seq_len, :]
    avg = query_slice.float().mean(dim=(0, 1)).cpu().numpy()
    if np.isnan(avg).any() or np.isinf(avg).any():
        return None

    m_sys = float(avg[:vis_start].sum())
    m_vis = float(avg[vis_start:vis_end].sum())
    m_ins = float(avg[vis_end:seq_len].sum())
    total = m_sys + m_vis + m_ins
    if total <= 1e-10:
        return None
    return np.array([m_vis / total, m_ins / total, m_sys / total], dtype=np.float32)


def _find_subsequence(haystack, needle, start=0, end=None):
    if not needle:
        return -1
    if end is None:
        end = len(haystack)
    n = len(needle)
    for i in range(start, max(start, end - n + 1)):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _infer_chatml_spans_from_ids(ids, image_start, image_end, im_start_id, im_end_id):
    # Find the closest <|im_end|> before image placeholder/span: end of system block.
    end_before = [i for i, t in enumerate(ids[:image_start]) if t == im_end_id]
    # Find the first <|im_end|> after image placeholder/span: end of user/question block.
    end_after = [i for i, t in enumerate(ids[image_end:]) if t == im_end_id]

    system_span = None
    instruction_span = None
    if end_before:
        system_span = (0, end_before[-1] + 1)
    if end_after:
        instruction_span = (image_end, image_end + end_after[0])
    return system_span, instruction_span


def _find_text_span(tokenizer, ids, text, search_start=0, search_end=None):
    candidates = [text, text.strip(), "\n" + text.strip(), " " + text.strip()]
    for cand in candidates:
        token_ids = tokenizer.encode(cand, add_special_tokens=False)
        pos = _find_subsequence(ids, token_ids, start=search_start, end=search_end)
        if pos >= 0:
            return pos, pos + len(token_ids)
    return None


def _compute_alloc_from_attn_with_ranges(attn_tensor, vis_start, vis_end, sys_range=None, ins_range=None):
    seq_len = attn_tensor.shape[-1]
    vis_start = max(0, min(vis_start, seq_len))
    vis_end = max(vis_start, min(vis_end, seq_len))

    if sys_range is None:
        sys_l, sys_r = 0, vis_start
    else:
        sys_l, sys_r = sys_range
    if ins_range is None:
        ins_l, ins_r = vis_end, seq_len
    else:
        ins_l, ins_r = ins_range

    sys_l, sys_r = max(0, min(sys_l, seq_len)), max(0, min(sys_r, seq_len))
    ins_l, ins_r = max(0, min(ins_l, seq_len)), max(0, min(ins_r, seq_len))

    # Use instruction tokens as query rows; if empty, fallback to suffix rows.
    q_l, q_r = ins_l, ins_r
    if q_r <= q_l:
        q_l, q_r = vis_end, seq_len
    if q_r <= q_l:
        q_l, q_r = 0, seq_len

    query_slice = attn_tensor[0, :, q_l:q_r, :]
    avg = query_slice.float().mean(dim=(0, 1)).cpu().numpy()
    if np.isnan(avg).any() or np.isinf(avg).any():
        return None

    m_sys = float(avg[sys_l:sys_r].sum()) if sys_r > sys_l else 0.0
    m_vis = float(avg[vis_start:vis_end].sum()) if vis_end > vis_start else 0.0
    m_ins = float(avg[ins_l:ins_r].sum()) if ins_r > ins_l else 0.0
    total = m_sys + m_vis + m_ins
    if total <= 1e-10:
        return None
    return np.array([m_vis / total, m_ins / total, m_sys / total], dtype=np.float32)


def _chatml_system_block(system_prompt):
    return f"<|im_start|>system\n{system_prompt}"


def _validate_spans(seq_len, sys_range, vis_range, ins_range, tag=""):
    ranges = {"system": sys_range, "visual": vis_range, "instruction": ins_range}
    for name, (l, r) in ranges.items():
        if not (0 <= l <= r <= seq_len):
            raise RuntimeError(f"[{tag}] invalid {name} range: [{l}, {r}) with seq_len={seq_len}")
    if not (sys_range[1] <= vis_range[0] and vis_range[1] <= ins_range[0]):
        raise RuntimeError(
            f"[{tag}] overlapping/out-of-order ranges: "
            f"sys={sys_range}, vis={vis_range}, ins={ins_range}, seq_len={seq_len}"
        )


def _load_diffusionvl(model_id, system_prompt):
    # Local import to avoid forcing llava dependency for qwen-base path.
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    overwrite_cfg = {
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
        "enable_bd3lm": True,
        "bd3lm_block_size": 8,
    }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_id,
        model_base=None,
        model_name="diffusionvl_qwenvl",
        device_map="cuda:0",
        torch_dtype="float16",
        attn_implementation="eager",
        overwrite_config=overwrite_cfg,
        force_model_type="diffusionvl_qwenvl",
    )
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "default_image_token": DEFAULT_IMAGE_TOKEN,
        "image_token_index": IMAGE_TOKEN_INDEX,
        "conv_templates": conv_templates,
        "tokenizer_image_token": tokenizer_image_token,
        "system_prompt": system_prompt,
    }


def _get_alloc_diffusionvl(backend, pil_image, question):
    model = backend["model"]
    tokenizer = backend["tokenizer"]
    image_processor = backend["image_processor"]
    conv_templates = backend["conv_templates"]
    tokenizer_image_token = backend["tokenizer_image_token"]
    default_image_token = backend["default_image_token"]
    image_token_index = backend["image_token_index"]

    processor_output = image_processor.preprocess(pil_image.convert("RGB"), return_tensors="pt")
    img = processor_output["pixel_values"]
    if img.dim() == 4:
        img = img[0]
    elif img.dim() == 3:
        img = img[0]
    # Expected Qwen-vision format for DiffusionVL: [num_patches, hidden_dim]
    img = img.half().to("cuda:0")

    image_grid_thw = None
    if "image_grid_thw" in processor_output:
        grid = processor_output["image_grid_thw"]
        if isinstance(grid, torch.Tensor):
            image_grid_thw = [grid[0].tolist()]
        else:
            image_grid_thw = [grid]

    question_with_token = default_image_token + "\n" + question
    conv = copy.deepcopy(conv_templates["qwen_2_5"])
    conv.system = _chatml_system_block(backend["system_prompt"])
    conv.append_message(conv.roles[0], question_with_token)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors="pt").unsqueeze(0).to("cuda:0")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attn_mask = input_ids.ne(pad_id).to("cuda:0")
    # Robustly infer visual span from attention sequence length:
    # expanded_len = text_len - num_image_placeholders + num_visual_tokens
    # => num_visual_tokens = expanded_len - (text_len - num_image_placeholders)
    vis_start = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0].item()
    text_len = int(input_ids.shape[1])
    num_image_placeholders = int((input_ids[0] == image_token_index).sum().item())
    vis_end_inferred = None
    ids_list = input_ids[0].detach().cpu().tolist()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sys_span_orig, q_span_orig = _infer_chatml_spans_from_ids(ids_list, vis_start, vis_start + 1, im_start_id, im_end_id)
    if sys_span_orig is None:
        # Fallback to text matching if ChatML tags are unavailable.
        sys_span_orig = (
            _find_text_span(
                tokenizer,
                ids_list,
                backend["system_prompt"],
                search_start=0,
                search_end=vis_start,
            )
            or (0, vis_start)
        )
    if q_span_orig is None:
        q_span_orig = _find_text_span(tokenizer, ids_list, question, search_start=vis_start + 1)

    alloc_dict = {}
    handles = []

    def _make_hook(layer_idx):
        def _hook(_module, _inp, out):
            if not (isinstance(out, tuple) and len(out) >= 2 and out[1] is not None):
                return
            nonlocal vis_end_inferred
            if vis_end_inferred is None:
                expanded_len = int(out[1].shape[-1])
                num_visual_tokens = expanded_len - (text_len - num_image_placeholders)
                if num_visual_tokens <= 0:
                    raise RuntimeError(
                        f"Failed to infer visual token span: expanded_len={expanded_len}, "
                        f"text_len={text_len}, image_placeholders={num_image_placeholders}"
                    )
                vis_end_inferred = vis_start + num_visual_tokens

            # Map original-text spans to expanded sequence positions.
            expanded_shift = (vis_end_inferred - vis_start) - 1
            sys_span_exp = sys_span_orig if sys_span_orig is not None else (0, vis_start)
            if q_span_orig is None:
                ins_span_exp = (vis_end_inferred, int(out[1].shape[-1]))
            else:
                ins_span_exp = (q_span_orig[0] + expanded_shift, q_span_orig[1] + expanded_shift)
            _validate_spans(int(out[1].shape[-1]), sys_span_exp, (vis_start, vis_end_inferred), ins_span_exp, tag="diffusionvl")

            vec = _compute_alloc_from_attn_with_ranges(
                out[1],
                vis_start,
                vis_end_inferred,
                sys_range=sys_span_exp,
                ins_range=ins_span_exp,
            )
            if vec is not None:
                alloc_dict[layer_idx] = vec
            return (out[0], None) + out[2:]

        return _hook

    for li, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(_make_hook(li)))

    try:
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                images=[img],
                image_grid_thws=image_grid_thw,
                modalities=["image"],
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()
    return alloc_dict


def _load_qwen_base(model_id, system_prompt):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    return {"processor": processor, "model": model, "system_prompt": system_prompt}


def _find_visual_bounds_qwen(input_ids, model, processor):
    ids = input_ids[0].tolist()
    cfg = model.config
    vis_start_id = getattr(cfg, "vision_start_token_id", None)
    vis_end_id = getattr(cfg, "vision_end_token_id", None)

    if vis_start_id is None:
        vis_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if vis_end_id is None:
        vis_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    if vis_start_id not in ids or vis_end_id not in ids:
        raise RuntimeError("Cannot find vision_start/vision_end token ids in Qwen input.")

    vis_start = ids.index(vis_start_id)
    vis_end = len(ids) - 1 - ids[::-1].index(vis_end_id) + 1  # end-exclusive
    if vis_end <= vis_start:
        raise RuntimeError(f"Invalid visual range in Qwen input: [{vis_start}, {vis_end})")
    return vis_start, vis_end


def _get_alloc_qwen_base(backend, pil_image, question):
    model = backend["model"]
    processor = backend["processor"]

    messages = [
        {"role": "system", "content": backend["system_prompt"]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt")
    inputs = {k: v.to("cuda:0") for k, v in inputs.items() if hasattr(v, "to")}

    vis_start, vis_end = _find_visual_bounds_qwen(inputs["input_ids"], model, processor)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)

    ids_list = inputs["input_ids"][0].detach().cpu().tolist()
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    sys_span, ins_span = _infer_chatml_spans_from_ids(ids_list, vis_start, vis_end, im_start_id, im_end_id)
    if sys_span is None:
        sys_span = (
            _find_text_span(processor.tokenizer, ids_list, backend["system_prompt"], search_start=0, search_end=vis_start)
            or (0, vis_start)
        )
    if ins_span is None:
        ins_span = _find_text_span(processor.tokenizer, ids_list, question, search_start=vis_end) or (vis_end, len(ids_list))
    _validate_spans(len(ids_list), sys_span, (vis_start, vis_end), ins_span, tag="qwen_base")

    alloc_dict = {}
    for li, attn in enumerate(outputs.attentions):
        vec = _compute_alloc_from_attn_with_ranges(
            attn,
            vis_start,
            vis_end,
            sys_range=sys_span,
            ins_range=ins_span,
        )
        if vec is not None:
            alloc_dict[li] = vec
    return alloc_dict


def main():
    p = argparse.ArgumentParser("Average attention allocation for image VQA tasks on Qwen base / DiffusionVL.")
    p.add_argument("--dataset", type=str, required=True, choices=["mme", "mmvp", "mmmu"])
    p.add_argument("--model_type", type=str, required=True, choices=["diffusionvl_qwenvl", "qwen2_5_vl_base"])
    p.add_argument("--model_id", type=str, default="")
    p.add_argument("--num_samples", type=int, default=128, help="0 means all samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_each", action="store_true")
    p.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_type == "diffusionvl_qwenvl":
        model_id = args.model_id or "hustvl/DiffusionVL-Qwen2.5VL-7B"
        print(f"[1/3] load DiffusionVL model: {model_id}")
        backend = _load_diffusionvl(model_id, args.system_prompt)
        run_one = _get_alloc_diffusionvl
    else:
        model_id = args.model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"[1/3] load Qwen base model: {model_id}")
        backend = _load_qwen_base(model_id, args.system_prompt)
        run_one = _get_alloc_qwen_base

    print(f"[2/3] load dataset: {args.dataset}")
    samples = _load_samples(args.dataset, args.num_samples, args.seed)
    print(f"  using {len(samples)} samples")
    if not samples:
        raise RuntimeError("No usable samples found.")

    allocs = []
    for i, s in enumerate(samples, start=1):
        sid = s["id"]
        print(f"[{i}/{len(samples)}] {sid}")
        try:
            alloc = run_one(backend, s["image"], s["question"])
            if alloc:
                allocs.append(alloc)
                if args.save_each:
                    plot_allocation(
                        alloc,
                        title=f"{args.model_type} {args.dataset} {sid}",
                        save_path=os.path.join(args.output_dir, f"fig5a_{sid}.png"),
                        rect_top=0.20,
                    )
                    with open(os.path.join(args.output_dir, f"alloc_{sid}.json"), "w") as f:
                        json.dump({str(k): v.tolist() for k, v in alloc.items()}, f, indent=2)
        except Exception as e:
            print(f"  skip {sid}: {e}")

    if not allocs:
        raise RuntimeError("All samples failed; no allocation computed.")

    print("[3/3] aggregate and save")
    avg_alloc = _avg_alloc(allocs)
    plot_allocation(
        avg_alloc,
        title=f"{args.model_type} {args.dataset} avg ({len(allocs)} samples)",
        save_path=os.path.join(args.output_dir, "fig5a_dataset_avg.png"),
        rect_top=0.20,
    )
    with open(os.path.join(args.output_dir, "alloc_dataset_avg.json"), "w") as f:
        json.dump({str(k): v.tolist() for k, v in avg_alloc.items()}, f, indent=2)
    print(f"done: {args.output_dir}")


if __name__ == "__main__":
    main()

