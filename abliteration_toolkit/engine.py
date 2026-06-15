"""Core abliteration engine for removing refusal directions from language models.

Implements Arditi et al. 2024 refusal-direction orthogonalization with support
for weight surgery, runtime hooks, and multi-direction re-abliteration.
"""

import csv
import gc
import json
import logging
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .prompts import HARMFUL, HARMLESS, is_refusal

log = logging.getLogger(__name__)


class SurgeryMode(str, Enum):
    WEIGHT_SURGERY = "weight_surgery"
    RUNTIME_HOOKS = "runtime_hooks"
    REABLITERATE = "reabliterate"


def _render_prompt(tokenizer, content: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            "<|im_start|>user\n"
            + content
            + "\n<|im_end|>\n<|im_start|>assistant\n"
        )


def _first_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _layer_name_patterns() -> List[str]:
    return ["layers", "decoder", "block"]


_VISION_KEYWORDS = ("vision", "visual", "patch_embed", "image", "vit", "rotary_emb")


class Abliterator:
    def __init__(
        self,
        model_id: str,
        output_dir: str,
        mode: str = "auto",
        n_harmful: int = 32,
        n_harmless: int = 32,
        n_eval: int = 16,
        n_val: int = 8,
        alpha: float = 1.0,
        hf_token: Optional[str] = None,
        quant_bits: Optional[int] = None,
        device_map: str = "auto",
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = True,
        no_split_modules: Optional[list] = None,
        custom_transformers_url: Optional[str] = None,
        pre_patch_fn: Optional[Callable] = None,
        skip_vision_layers: bool = True,
        push_to_hf: bool = False,
        adapter_repo: Optional[str] = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = self._resolve_mode(mode)
        self.n_harmful = n_harmful
        self.n_harmless = n_harmless
        self.n_eval = n_eval
        self.n_val = n_val
        self.alpha = alpha
        self.hf_token = hf_token
        self.quant_bits = quant_bits
        self.device_map = device_map
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code
        self.no_split_modules = no_split_modules or []
        self.custom_transformers_url = custom_transformers_url
        self.pre_patch_fn = pre_patch_fn
        self.skip_vision_layers = skip_vision_layers
        self.push_to_hf = push_to_hf
        self.adapter_repo = adapter_repo

        self.model = None
        self.tokenizer = None
        self.layers: List[Tuple[int, nn.Module, str]] = []
        self.directions: Dict[int, torch.Tensor] = {}
        self.best_layer: Optional[int] = None
        self._hooks: List = []

    def _resolve_mode(self, mode: str) -> SurgeryMode:
        if mode == "auto":
            if self.quant_bits is not None:
                return SurgeryMode.RUNTIME_HOOKS
            return SurgeryMode.WEIGHT_SURGERY
        return SurgeryMode(mode)

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.custom_transformers_url:
            log.info("Installing custom transformers from %s", self.custom_transformers_url)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", self.custom_transformers_url]
            )

        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )
        if self.max_memory is not None:
            load_kwargs["max_memory"] = self.max_memory
        if self.no_split_modules:
            load_kwargs["no_split_module"] = self.no_split_modules

        if self.quant_bits in (4, 8):
            from transformers import BitsAndBytesConfig

            if self.quant_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = bnb_config
        else:
            load_kwargs["torch_dtype"] = torch.float16

        log.info("Loading model %s (quant=%s)", self.model_id, self.quant_bits)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.config.use_cache = True
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                token=self.hf_token,
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                token=self.hf_token,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.pre_patch_fn is not None:
            log.info("Applying pre-patch function: %s", self.pre_patch_fn)
            if callable(self.pre_patch_fn):
                self.pre_patch_fn(self.model)
            else:
                name = str(self.pre_patch_fn)
                try:
                    mod_path, _, fn_name = name.rpartition(".")
                    mod = importlib.import_module(mod_path) if mod_path else None
                    fn = getattr(mod, fn_name) if mod else globals().get(fn_name)
                    if fn is None:
                        fn = getattr(self, fn_name, None)
                    if fn is not None:
                        result = fn(self.model)
                        if result is not None:
                            self.model = result
                    else:
                        log.warning("Pre-patch function %s not found", name)
                except Exception as exc:
                    log.warning("Pre-patch failed: %s", exc)

        return self.model, self.tokenizer

    def detect_layers(self, model) -> List[Tuple[int, nn.Module, str]]:
        best = None
        for name, module in model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            low = name.lower()
            if self.skip_vision_layers and any(k in low for k in _VISION_KEYWORDS):
                continue
            if not any(name.endswith(p) for p in _layer_name_patterns()):
                continue
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            if not hasattr(parent, "embed_tokens"):
                continue
            if best is None or len(module) > len(best[0]):
                best = (list(module), parent, ".".join(parts[:-1]), name)

        if best is None:
            try:
                layers_module = model.model.layers
                self.layers = [
                    (i, layer, "model.layers")
                    for i, layer in enumerate(layers_module)
                ]
                return self.layers
            except AttributeError:
                log.error("Could not locate decoder layers")
                self.layers = []
                return self.layers

        layers_module, parent, parent_path, full_path = best
        self.layers = [
            (i, layer, full_path) for i, layer in enumerate(layers_module)
        ]
        return self.layers

    def collect_residuals(
        self, model, tokenizer, prompts: List[str], layers: List[Tuple[int, nn.Module, str]]
    ) -> Dict[int, torch.Tensor]:
        device = _first_device(model)
        acts: Dict[int, List[torch.Tensor]] = {i: [] for i, _, _ in layers}
        handles = []

        def make_hook(idx):
            def hook(_m, _inp, out):
                y = out[0] if isinstance(out, tuple) else out
                if hasattr(y, "detach") and y.ndim >= 3:
                    acts[idx].append(y[:, -1, :].detach().float().cpu())
            return hook

        for i, layer, _ in layers:
            handles.append(layer.register_forward_hook(make_hook(i)))

        try:
            model.eval()
            for p in prompts:
                enc = tokenizer(
                    _render_prompt(tokenizer, p),
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(device)
                with torch.inference_mode():
                    model(**enc)
        finally:
            for h in handles:
                h.remove()

        out = {}
        for idx in acts:
            if acts[idx]:
                out[idx] = torch.cat(acts[idx], dim=0)
        return out

    def compute_directions(
        self,
        h_acts: Dict[int, torch.Tensor],
        n_acts: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        directions = {}
        for i in h_acts:
            if i in n_acts and i > 0:
                v = h_acts[i].mean(0) - n_acts[i].mean(0)
                n = v.norm().item()
                if n > 1e-6:
                    directions[i] = v / n
        self.directions = directions
        return directions

    def find_best_layer(
        self,
        model,
        tokenizer,
        directions: Dict[int, torch.Tensor],
        val_prompts: List[str],
        layers: List[Tuple[int, nn.Module, str]],
    ) -> int:
        best_layer = None
        best_score = len(val_prompts) + 1

        for li, vec in directions.items():
            handles = self.attach_runtime_hooks(model, layers, vec, strength=1.0)
            try:
                refusals = 0
                for p in val_prompts:
                    text = self.generate(model, tokenizer, p, max_new_tokens=80)
                    refusals += int(is_refusal(text))
                log.info("Layer %d: %d/%d refusals", li, refusals, len(val_prompts))
                if refusals < best_score:
                    best_score = refusals
                    best_layer = li
            finally:
                for h in handles:
                    h.remove()

        self.best_layer = best_layer
        return best_layer

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 160,
    ) -> str:
        device = _first_device(model)
        enc = tokenizer(
            _render_prompt(tokenizer, prompt),
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def attach_runtime_hooks(
        self,
        model,
        layers: List[Tuple[int, nn.Module, str]],
        direction: torch.Tensor,
        strength: float = 1.0,
    ) -> List:
        handles = []
        v = direction

        def make_hook():
            def hook(_m, _inp, out):
                y, *rest = (out if isinstance(out, tuple) else (out,))
                yf = y.float()
                vv = v.to(yf.device, dtype=torch.float32)
                proj = (yf @ vv).unsqueeze(-1) * vv
                y2 = (yf - proj * strength).to(y.dtype)
                if rest:
                    return (y2,) + tuple(rest)
                return y2
            return hook

        for _i, layer, _ in layers:
            handles.append(layer.register_forward_hook(make_hook()))
        self._hooks.extend(handles)
        return handles

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def orthogonalize_weight(W: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        orig_dtype = W.dtype
        W32 = W.to(torch.float32)
        r32 = direction.to(torch.float32).to(W.device)
        r32 = r32 / (r32.norm() + 1e-12)
        d = r32.numel()
        if W32.shape[1] == d:
            new = W32 - (W32 @ r32).unsqueeze(-1) * r32.unsqueeze(0)
        elif W32.shape[0] == d:
            new = W32 - r32.unsqueeze(1) @ (r32.unsqueeze(0) @ W32)
        else:
            return W
        return new.to(orig_dtype)

    def apply_weight_surgery(
        self,
        model,
        layers: List[Tuple[int, nn.Module, str]],
        direction: torch.Tensor,
    ):
        parent = model
        for name in layers[0][2].split(".")[:-1] if layers else []:
            parent = getattr(parent, name, None)

        if parent is not None and hasattr(parent, "embed_tokens"):
            W = parent.embed_tokens.weight.data
            new = self.orthogonalize_weight(W, direction.to(W.dtype))
            if new.shape == W.shape:
                parent.embed_tokens.weight.data.copy_(new)

        for _i, layer, _path in layers:
            for sub_path in ["self_attn.o_proj", "mlp.down_proj"]:
                obj = layer
                ok = True
                for part in sub_path.split("."):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        ok = False
                        break
                if ok and hasattr(obj, "weight"):
                    W = obj.weight.data
                    new = self.orthogonalize_weight(W, direction.to(W.dtype))
                    if new.shape == W.shape:
                        obj.weight.data.copy_(new)

    def apply_reabliterate(
        self,
        model,
        layers: List[Tuple[int, nn.Module, str]],
        directions: Dict[int, torch.Tensor],
        alpha: float = 3.0,
    ):
        parent = model
        for name in layers[0][2].split(".")[:-1] if layers else []:
            parent = getattr(parent, name, None)

        if parent is not None and hasattr(parent, "embed_tokens"):
            W = parent.embed_tokens.weight.data
            for _li, d in directions.items():
                d32 = d.to(torch.float32).to(W.device)
                d32 = d32 / (d32.norm() + 1e-12)
                d_scaled = d32 * alpha
                W = self.orthogonalize_weight(W, d_scaled.to(W.dtype))
            if W.shape == parent.embed_tokens.weight.data.shape:
                parent.embed_tokens.weight.data.copy_(W)

        for i, layer, _path in layers:
            d = directions.get(i)
            if d is None:
                continue
            for sub_path in [
                "self_attn.o_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "mlp.down_proj",
                "mlp.up_proj",
                "mlp.gate_proj",
            ]:
                obj = layer
                ok = True
                for part in sub_path.split("."):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        ok = False
                        break
                if ok and hasattr(obj, "weight"):
                    W = obj.weight.data
                    d32 = d.to(torch.float32).to(W.device)
                    d32 = d32 / (d32.norm() + 1e-12)
                    d_scaled = d32 * alpha
                    new = self.orthogonalize_weight(W, d_scaled.to(W.dtype))
                    if new.shape == W.shape:
                        obj.weight.data.copy_(new)

    def evaluate(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: Optional[List[Tuple[int, nn.Module, str]]] = None,
        direction: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        use_hooks = direction is not None and layers is not None
        handles = []
        if use_hooks:
            handles = self.attach_runtime_hooks(model, layers, direction, strength=1.0)

        rows = []
        try:
            for i, p in enumerate(prompts):
                text = self.generate(model, tokenizer, p)
                refused = is_refusal(text)
                rows.append(
                    {
                        "idx": i,
                        "prompt": p[:200],
                        "refused": refused,
                        "output": text[:1500],
                    }
                )
                tag = "REFUSED" if refused else "OK"
                log.info("[%s] %d %s %s", tag, i, p[:60], text[:80].replace("\n", " "))
        finally:
            for h in handles:
                h.remove()

        refusal_count = sum(1 for r in rows if r["refused"])
        return {
            "rows": rows,
            "refusal_count": refusal_count,
            "total": len(rows),
        }

    def _save_artifacts(
        self,
        directions: Dict[int, torch.Tensor],
        best_layer: Optional[int],
        eval_before: Dict[str, Any],
        eval_after: Dict[str, Any],
    ):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.best_layer is not None and self.best_layer in directions:
            torch.save(
                {
                    "direction": directions[self.best_layer],
                    "best_layer": self.best_layer,
                    "all_directions": {k: v for k, v in directions.items()},
                },
                self.output_dir / "refusal_direction.pt",
            )

        rows = []
        for r in eval_before.get("rows", []):
            rows.append({"phase": "before", **r})
        for r in eval_after.get("rows", []):
            rows.append({"phase": "after", **r})

        if rows:
            with open(self.output_dir / "abliteration_eval.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        surgery_ok = (
            eval_after["refusal_count"] * 2 < eval_before["refusal_count"]
        ) or (
            eval_before["refusal_count"] >= 8
            and eval_after["refusal_count"] <= max(2, eval_before["refusal_count"] // 4)
        )

        summary = {
            "model_id": self.model_id,
            "mode": self.mode.value if isinstance(self.mode, SurgeryMode) else self.mode,
            "best_layer": best_layer,
            "candidate_layers": sorted(directions.keys()),
            "before_refusals": eval_before["refusal_count"],
            "before_total": eval_before["total"],
            "after_refusals": eval_after["refusal_count"],
            "after_total": eval_after["total"],
            "refusal_drop": eval_before["refusal_count"] - eval_after["refusal_count"],
            "surgery_ok": surgery_ok,
            "alpha": self.alpha,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        apply_script = self._generate_apply_script(directions, best_layer)
        (self.output_dir / "apply_runtime_ablation.py").write_text(
            apply_script, encoding="utf-8"
        )

        return summary

    def _generate_apply_script(
        self,
        directions: Dict[int, torch.Tensor],
        best_layer: Optional[int],
    ) -> str:
        direction = directions.get(best_layer) if best_layer is not None else None
        if direction is None:
            return "# No direction found; cannot generate apply script."

        return (
            '''#!/usr/bin/env python3
"""Apply runtime refusal-direction ablation to a loaded model."""
import torch

def apply(model, strength=1.0):
    direction = torch.load(
        "refusal_direction.pt", map_location="cpu"
    )["direction"]
    handles = []

    def make_hook(d, s):
        def hook(_m, _inp, out):
            y, *rest = (out if isinstance(out, tuple) else (out,))
            yf = y.float()
            vv = d.to(yf.device, dtype=torch.float32)
            proj = (yf @ vv).unsqueeze(-1) * vv
            y2 = (yf - proj * s).to(y.dtype)
            if rest:
                return (y2,) + tuple(rest)
            return y2
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            handles.append(
                module.register_forward_hook(make_hook(direction, strength))
            )

    return handles
'''
        )

    def push_to_hub(self, repo: str, token: Optional[str] = None):
        hf_token = token or self.hf_token
        if self.model is None:
            raise RuntimeError("No model loaded")
        log.info("Pushing to HuggingFace: %s", repo)
        self.model.push_to_hub(
            repo, token=hf_token, private=False, safe_serialization=True
        )
        self.tokenizer.push_to_hub(repo, token=hf_token, private=False)

    def run(self):
        try:
            log.info("=== Abliteration pipeline start ===")

            log.info("Stage: load_model")
            self._load_model()

            log.info("Stage: detect_layers")
            self.layers = self.detect_layers(self.model)
            if not self.layers:
                raise RuntimeError("Could not locate decoder layers")
            log.info("Found %d layers at %s", len(self.layers), self.layers[0][2])

            log.info("Stage: collect_residuals")
            harmful = HARMFUL[: self.n_harmful]
            harmless = HARMLESS[: self.n_harmless]
            h_acts = self.collect_residuals(self.model, self.tokenizer, harmful, self.layers)
            n_acts = self.collect_residuals(self.model, self.tokenizer, harmless, self.layers)

            log.info("Stage: compute_directions")
            directions = self.compute_directions(h_acts, n_acts)
            if not directions:
                raise RuntimeError("No usable refusal directions extracted")
            log.info("Candidate layers: %s", sorted(directions.keys()))

            log.info("Stage: find_best_layer")
            val_start = self.n_eval
            val_end = min(self.n_eval + self.n_val, len(HARMFUL))
            val_prompts = HARMFUL[val_start:val_end] if val_end > val_start else HARMFUL[: self.n_val]
            best_layer = self.find_best_layer(
                self.model, self.tokenizer, directions, val_prompts, self.layers
            )
            log.info("Best layer: %d", best_layer)

            log.info("Stage: evaluate_before")
            eval_prompts = HARMFUL[: self.n_eval]
            eval_before = self.evaluate(self.model, self.tokenizer, eval_prompts)
            log.info("Before: %d/%d refusals", eval_before["refusal_count"], eval_before["total"])

            log.info("Stage: apply_ablation")
            if self.mode == SurgeryMode.REABLITERATE:
                self.apply_reabliterate(self.model, self.layers, directions, alpha=self.alpha)
            elif self.mode == SurgeryMode.WEIGHT_SURGERY:
                self.apply_weight_surgery(self.model, self.layers, directions[best_layer])
            else:
                log.info("Runtime hooks mode; model not modified in-place")

            log.info("Stage: evaluate_after")
            if self.mode == SurgeryMode.RUNTIME_HOOKS:
                eval_after = self.evaluate(
                    self.model,
                    self.tokenizer,
                    eval_prompts,
                    layers=self.layers,
                    direction=directions[best_layer],
                )
            else:
                eval_after = self.evaluate(self.model, self.tokenizer, eval_prompts)
            log.info("After: %d/%d refusals", eval_after["refusal_count"], eval_after["total"])

            log.info("Stage: save_artifacts")
            summary = self._save_artifacts(directions, best_layer, eval_before, eval_after)

            if self.mode == SurgeryMode.WEIGHT_SURGERY or self.mode == SurgeryMode.REABLITERATE:
                log.info("Stage: save_model")
                model_dir = self.output_dir / "model"
                model_dir.mkdir(exist_ok=True)
                self.model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="4GB")
                self.tokenizer.save_pretrained(model_dir)

            if self.push_to_hf and self.adapter_repo and summary.get("surgery_ok"):
                log.info("Stage: push_to_hf")
                self.push_to_hub(self.adapter_repo)

            log.info("=== Pipeline complete ===")
            return summary

        except Exception as exc:
            log.error("Pipeline failed: %s", exc, exc_info=True)
            raise

        finally:
            self.remove_hooks()
            _gc_cuda()
