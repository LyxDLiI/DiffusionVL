from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))
from typing import Dict, List, Optional

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils_answers import normalize_answer_mme

_MASK_ID = 151671
_EOS_ID = 151645


class TrajectoryRecorder:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.output_dir / "mme_trajectories.jsonl"
        self.context: Optional[Dict] = None

    def set_context(self, context: Dict) -> None:
        self.context = context

    def clear_context(self) -> None:
        self.context = None

    def write(self, record: Dict) -> None:
        with self.trajectory_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


_RECORDER: Optional[TrajectoryRecorder] = None
_INSTALLED = False


def _clean_decoded_text(tokenizer, token_ids: torch.Tensor, stop_terms: List[str]) -> str:
    keep = [int(tok) for tok in token_ids.tolist() if int(tok) not in {_MASK_ID}]
    if not keep:
        return ""
    text = tokenizer.decode(keep, skip_special_tokens=True).strip()
    for term in stop_terms:
        if term and term in text:
            text = text.split(term)[0]
    return text.strip().rstrip(".")


def _build_step_record(tokenizer, generated_ids: torch.Tensor, prompt_len: int, gen_length: int, step_index: int, step_probs: torch.Tensor, is_mask: torch.Tensor, transfer_mask: torch.Tensor) -> Dict:
    gen_ids = generated_ids[0, prompt_len:prompt_len + gen_length]
    decoded_answer = _clean_decoded_text(tokenizer, gen_ids, ["\n", tokenizer.eos_token or ""])

    masked_probs = step_probs[is_mask]
    if masked_probs.numel() == 0:
        top1 = top2 = gap = entropy = None
    else:
        top2_vals = torch.topk(masked_probs, k=2, dim=-1).values
        top1 = float(top2_vals[:, 0].mean().item())
        top2 = float(top2_vals[:, 1].mean().item())
        gap = float((top2_vals[:, 0] - top2_vals[:, 1]).mean().item())
        entropy_vals = -(masked_probs * torch.log(masked_probs.clamp_min(1e-12))).sum(dim=-1)
        entropy = float(entropy_vals.mean().item())

    remaining_mask = (gen_ids == _MASK_ID).float().mean().item()
    transferred = int(transfer_mask[:, :].sum().item())

    return {
        "t": step_index,
        "decoded_answer": decoded_answer,
        "normalized_answer": normalize_answer_mme(decoded_answer),
        "top1": top1,
        "top2": top2,
        "gap": gap,
        "entropy": entropy,
        "mask_ratio": float(remaining_mask),
        "transferred_tokens": transferred,
    }


def install_patches(output_dir: str) -> TrajectoryRecorder:
    global _RECORDER, _INSTALLED
    if _INSTALLED and _RECORDER is not None:
        return _RECORDER

    _RECORDER = TrajectoryRecorder(output_dir)

    from lmms_eval.models.llava_onevision_diffusionvl_qwenvl import Llava_OneVision_DiffusionVL_QwenVL
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from lmms_eval import utils

    def patched_generate_with_bd3lm(self, inputs_embeds: torch.FloatTensor, steps: int = 4, gen_length: int = 128, temperature: float = 0.0, **kwargs):
        from transformers.cache_utils import DynamicCache

        tokenizer = kwargs.get("tokenizer")
        if tokenizer is None:
            return original_generate_with_bd3lm(self, inputs_embeds=inputs_embeds, steps=steps, gen_length=gen_length, temperature=temperature, **kwargs)

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        prompt_len = inputs_embeds.shape[1]
        block_size = self.model.bd3lm_block_size
        mask_id = _MASK_ID

        is_full_diffusion_ablation = block_size >= (prompt_len + gen_length)
        if is_full_diffusion_ablation:
            total_length = prompt_len + gen_length
            num_blocks = 1
        else:
            num_blocks = (prompt_len + gen_length + block_size - 1) // block_size
            total_length = num_blocks * block_size

        x_ids = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
        mask_embed = self.get_input_embeddings()(torch.tensor([mask_id], device=device))
        x_embeds = mask_embed.repeat(batch_size, total_length, 1)
        x_embeds[:, :prompt_len] = inputs_embeds.clone()

        prompt_logits = self.lm_head(inputs_embeds)
        prompt_ids_reconstructed = torch.argmax(prompt_logits, dim=-1)
        x_ids[:, :prompt_len] = prompt_ids_reconstructed

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device)).to(inputs_embeds.dtype)
        block_diffusion_mask_bool = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1).unsqueeze(0)
        block_diffusion_mask = block_diffusion_mask_bool.unsqueeze(1)
        block_diffusion_mask = torch.where(block_diffusion_mask == 0.0, torch.full_like(block_diffusion_mask, float("-inf")), 0.0)
        if is_full_diffusion_ablation:
            block_diffusion_mask = block_diffusion_mask[:, :, :total_length, :total_length]

        position_ids = torch.arange(total_length, device=device).unsqueeze(0).expand(batch_size, -1)
        prefill_blocks = prompt_len // block_size
        prefill_length = prefill_blocks * block_size

        past_key_values = DynamicCache()
        if prefill_length > 0:
            prefill_embeds = x_embeds[:, :prefill_length]
            prefill_mask = block_diffusion_mask[:, :, :prefill_length, :prefill_length]
            prefill_pos_ids = position_ids[:, :prefill_length]
            model_mask = {"full_attention": prefill_mask, "sliding_attention": prefill_mask}
            prefill_outputs = self.model(inputs_embeds=prefill_embeds, attention_mask=model_mask, position_ids=prefill_pos_ids, past_key_values=past_key_values, use_cache=True, store_kv=True)
            past_key_values = prefill_outputs.past_key_values

        num_transfer_tokens = self.get_bd3lm_num_transfer_tokens(block_size, steps)
        step_records: List[Dict] = []
        logical_step = 0

        for block_idx in range(prefill_blocks, num_blocks):
            block_start = block_idx * block_size
            block_end = block_start + block_size
            cur_block_embeds = x_embeds[:, block_start:block_end].clone()
            cur_block_ids = x_ids[:, block_start:block_end]
            cur_mask = block_diffusion_mask[:, :, block_start:block_end, :block_end]
            cur_pos_ids = position_ids[:, block_start:block_end]
            model_mask = {"full_attention": cur_mask, "sliding_attention": cur_mask}

            for step in range(steps + 1):
                is_mask = torch.all(torch.abs(cur_block_embeds - mask_embed) < 1e-5, dim=-1)
                if not is_mask.any():
                    _ = self.model(inputs_embeds=cur_block_embeds, attention_mask=model_mask, position_ids=cur_pos_ids, past_key_values=past_key_values, use_cache=True, store_kv=True)
                    break

                outputs = self.model(inputs_embeds=cur_block_embeds, attention_mask=model_mask, position_ids=cur_pos_ids, past_key_values=past_key_values, use_cache=True, store_kv=False)
                logits = self.lm_head(outputs[0]).float()
                probs = F.softmax(logits, dim=-1)

                top_k = kwargs.get("top_k", 0)
                top_p = kwargs.get("top_p", 1.0)
                x0, x0_p = self._sample_with_temperature_topk_topp(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                remasking_strategy = kwargs.get("remasking_strategy", "low_confidence_static")
                num_to_transfer = num_transfer_tokens[step].item()

                transfer_mask = torch.zeros_like(x0, dtype=torch.bool, device=device)
                if remasking_strategy == "low_confidence_static":
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        num_masks = is_mask[j].sum().item()
                        k = min(num_to_transfer, num_masks)
                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_mask[j, idx] = True
                elif remasking_strategy == "low_confidence_dynamic":
                    confidence_threshold = kwargs.get("confidence_threshold", 0.85)
                    confidence = torch.where(is_mask, x0_p, -torch.inf)
                    for j in range(confidence.shape[0]):
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = high_conf_mask.sum().item()
                        if num_high_confidence >= num_to_transfer:
                            transfer_mask[j] = high_conf_mask
                        else:
                            num_masks = is_mask[j].sum().item()
                            k = min(num_to_transfer, num_masks)
                            if k > 0:
                                _, idx = torch.topk(confidence[j], k)
                                transfer_mask[j, idx] = True
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

                cur_block_ids = torch.where(transfer_mask, x0, cur_block_ids)
                x0_embeds = self.get_input_embeddings()(x0)
                cur_block_embeds = torch.where(transfer_mask.unsqueeze(-1), x0_embeds, cur_block_embeds)
                x_ids[:, block_start:block_end] = cur_block_ids

                logical_step += 1
                step_records.append(
                    _build_step_record(
                        tokenizer=tokenizer,
                        generated_ids=x_ids,
                        prompt_len=prompt_len,
                        gen_length=gen_length,
                        step_index=logical_step,
                        step_probs=probs[0],
                        is_mask=is_mask[0],
                        transfer_mask=transfer_mask,
                    )
                )

            x_embeds[:, block_start:block_end] = cur_block_embeds
            x_ids[:, block_start:block_end] = cur_block_ids
            if block_end > prompt_len:
                gen_start_in_block = max(prompt_len, block_start)
                gen_ids_check = x_ids[:, gen_start_in_block:block_end]
                if _EOS_ID in gen_ids_check:
                    break

        result = x_ids[:, prompt_len:prompt_len + gen_length]
        if _RECORDER and _RECORDER.context is not None:
            stop_terms = kwargs.get("stopping_criteria") or []
            if isinstance(stop_terms, str):
                stop_terms = [stop_terms]
            stop_terms = list(stop_terms) + ["\n", getattr(tokenizer, "eos_token", "")]
            final_output = _clean_decoded_text(tokenizer, result[0], stop_terms)
            context = dict(_RECORDER.context)
            context.update(
                {
                    "steps": step_records,
                    "final_output": final_output,
                    "meta": {
                        "gen_length": gen_length,
                        "steps": steps,
                        "remasking_strategy": kwargs.get("remasking_strategy", "low_confidence_static"),
                        "block_size": block_size,
                        "model_name": "diffusionvl_qwenvl",
                        "target": context.get("target"),
                        "category": context.get("category"),
                    },
                }
            )
            _RECORDER.write(context)
        return result

    def patched_generate_until(self, requests):
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="MME Trajectory")
        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            if task != "mme":
                raise ValueError(f"MME trajectory wrapper only supports task='mme', got {task!r}")

            doc = self.task_dict[task][split][doc_id]
            visual = doc_to_visual(doc)
            processor_output = self._image_processor(images=visual, return_tensors="pt")
            image_tensor = processor_output["pixel_values"]
            image_grid_thws = processor_output.get("image_grid_thw")
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                question = f"{DEFAULT_IMAGE_TOKEN}\n{context}"
            else:
                question = context

            conv = copy.deepcopy(conv_templates[self.conv_template]) if ("llama_3" in self.conv_template or "llava_llada" in self.conv_template) else conv_templates[self.conv_template].copy()
            if utils.is_json(question):
                question_json = json.loads(question)
                for idx, item in enumerate(question_json):
                    role = conv.roles[idx % 2]
                    conv.append_message(role, item["value"])
                conv.append_message(conv.roles[1], None)
            else:
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            active_gen_kwargs = dict(gen_kwargs)
            active_gen_kwargs.setdefault("temperature", 0)
            active_gen_kwargs.setdefault("cfg", 0.0)
            active_gen_kwargs.setdefault("gen_length", 128)
            active_gen_kwargs.setdefault("steps", 8)
            active_gen_kwargs.setdefault("remasking_strategy", "low_confidence_static")
            active_gen_kwargs["tokenizer"] = self.tokenizer
            active_gen_kwargs["image_sizes"] = [img.size for img in visual]
            if image_grid_thws is not None:
                active_gen_kwargs["image_grid_thws"] = image_grid_thws
            stop_terms = active_gen_kwargs.get("stopping_criteria", [])
            if isinstance(stop_terms, str):
                stop_terms = [stop_terms]
            stop_terms = list(stop_terms)
            for term in [conv.sep, "\n"]:
                if term and term not in stop_terms:
                    stop_terms.append(term)
            active_gen_kwargs["stopping_criteria"] = stop_terms

            if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                self._config.image_aspect_ratio = origin_image_aspect_ratio
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_id).to(self.device)

            if _RECORDER:
                _RECORDER.set_context(
                    {
                        "sample_id": str(doc.get("question_id", doc_id)),
                        "task": "mme",
                        "prompt": context,
                        "doc_id": int(doc_id),
                        "target": doc.get("answer"),
                        "category": doc.get("category"),
                        "question": doc.get("question"),
                    }
                )

            with torch.inference_mode():
                generated_ids = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_id, images=image_tensor, use_cache=False, **active_gen_kwargs)

            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            for term in stop_terms:
                if term and term in decoded:
                    decoded = decoded.split(term)[0]
            decoded = decoded.strip().rstrip(".")
            res.append(decoded)
            if _RECORDER:
                _RECORDER.clear_context()
            pbar.update(1)

        pbar.close()
        return res

    original_generate_with_bd3lm = getattr(__import__("llava.model.language_model.llava_diffusionvl_qwenvl", fromlist=["DiffusionVLQwenVLForCausalLM"]), "DiffusionVLQwenVLForCausalLM").generate_with_bd3lm
    diffusion_cls = getattr(__import__("llava.model.language_model.llava_diffusionvl_qwenvl", fromlist=["DiffusionVLQwenVLForCausalLM"]), "DiffusionVLQwenVLForCausalLM")
    diffusion_cls.generate_with_bd3lm = patched_generate_with_bd3lm
    Llava_OneVision_DiffusionVL_QwenVL.generate_until = patched_generate_until

    _INSTALLED = True
    return _RECORDER
