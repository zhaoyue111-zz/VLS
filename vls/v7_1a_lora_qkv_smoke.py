from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths
from vls.v2_experiment import prepare_functional_seg_head, resolve_device
from vls.v6_0_imagined_world_reliability import SOURCE_PROMPT, load_world_model
from vls.voxtell_states import VoxTellStateInterface
from vls.v7_0d_protocol_sanity import (
    ORDERS,
    VARIANTS,
    binary_metrics,
    build_evaluation_cache,
    build_train_cache,
    evaluate_network as evaluate_base_network,
    iter_cases,
    iter_image_cases,
    pooled_rows as pooled_rows_v7d,
    set_seed,
    trainable_parameters,
)


class PackedQKVLoRA(nn.Module):
    """LoRA on a packed MultiheadAttention in_proj_weight.

    The wrapped base attention is unchanged. The three deltas are concatenated
    in PyTorch's native Q/K/V row order, so this supports the actual packed
    implementation rather than pretending q_proj/k_proj/v_proj exist.
    """

    def __init__(self, base: nn.MultiheadAttention, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not base._qkv_same_embed_dim or base.in_proj_weight is None:
            raise ValueError("V7.1a only supports packed same-dimension MultiheadAttention")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        dim = base.embed_dim
        self.lora_A_q = nn.Parameter(torch.empty(rank, dim))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, rank))
        self.lora_A_k = nn.Parameter(torch.empty(rank, dim))
        self.lora_B_k = nn.Parameter(torch.zeros(dim, rank))
        self.lora_A_v = nn.Parameter(torch.empty(rank, dim))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, rank))
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for parameter in (self.lora_A_q, self.lora_A_k, self.lora_A_v):
            nn.init.kaiming_uniform_(parameter, a=np.sqrt(5))
        for parameter in (self.lora_B_q, self.lora_B_k, self.lora_B_v):
            nn.init.zeros_(parameter)

    def delta_weight(self) -> torch.Tensor:
        delta_q = self.lora_B_q @ self.lora_A_q
        delta_k = self.lora_B_k @ self.lora_A_k
        delta_v = self.lora_B_v @ self.lora_A_v
        return torch.cat((delta_q, delta_k, delta_v), dim=0) * self.scaling

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                need_weights: bool = True, attn_mask: torch.Tensor | None = None,
                average_attn_weights: bool = True, is_causal: bool = False):
        base = self.base
        delta = self.delta_weight().to(dtype=base.in_proj_weight.dtype)
        return F.multi_head_attention_forward(
            query, key, value, base.embed_dim, base.num_heads,
            base.in_proj_weight + delta, base.in_proj_bias, base.bias_k, base.bias_v,
            base.add_zero_attn, base.dropout, base.out_proj.weight, base.out_proj.bias,
            training=base.training, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
            v_proj_weight=None, static_k=None, static_v=None,
            average_attn_weights=average_attn_weights, is_causal=is_causal,
        )


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = qualified_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, replacement)


def inject_lora_qkv(network: nn.Module, rank: int, alpha: float, dropout: float) -> list[str]:
    for parameter in network.parameters():
        parameter.requires_grad = False
    candidates = [
        (name, module) for name, module in network.named_modules()
        if name.startswith("transformer_decoder") and isinstance(module, nn.MultiheadAttention)
    ]
    if not candidates:
        raise RuntimeError("No packed MultiheadAttention found under transformer_decoder")
    targets = []
    for name, module in candidates:
        _replace_module(network, name, PackedQKVLoRA(module, rank, alpha, dropout))
        targets.append(name)
    return targets


def lora_parameters(network: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for name, parameter in network.named_parameters() if name.startswith("transformer_decoder") and "lora_" in name]
    if not parameters:
        raise RuntimeError("No LoRA parameters were injected")
    return parameters


def metric_pool(rows: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        case = "__global__" if level == "global" else row["case"]
        key = (row["variant"], row["order"], case, int(row["step"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (variant, order, case, step), group in sorted(grouped.items()):
        counts = {key: sum(int(row[key]) for row in group) for key in ("tp", "fp", "tn", "fn")}
        tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
        output.append({
            "variant": variant, "order": order, "step": step, "case": case,
            "case_count": 4 if case == "__global__" else 1,
            "patch_count": len(group), **counts,
            "dice": 1.0 if tp + fp + fn == 0 else 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        })
    return output


def train_one_order(
    variant: str, source: str, order: str, base_network: nn.Module,
    train_cache: list[dict[str, Any]], evaluation_cache: list[dict[str, Any]],
    interface: VoxTellStateInterface, args: argparse.Namespace, device: torch.device,
    target_names: list[str] | None, base_total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    student = copy.deepcopy(base_network)
    targets = inject_lora_qkv(student, args.lora_rank, args.lora_alpha, args.lora_dropout)
    student = student.to(device)
    trainable = lora_parameters(student)
    base_trainable = sum(parameter.numel() for name, parameter in student.named_parameters() if parameter.requires_grad and "lora_" not in name)
    if base_trainable != 0:
        raise AssertionError(f"base_trainable_parameters={base_trainable}, expected 0")
    if target_names is not None and targets != target_names:
        raise AssertionError("LoRA target modules differ across reinitializations")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    ordered = train_cache if order == "forward" else list(reversed(train_cache))
    losses, pseudo_rows, eval_rows = [], [], []
    for step, sample in enumerate(ordered, start=1):
        image = sample["image"].to(device).clone()
        embedding = sample["embedding"].to(device).clone().float()
        pseudo = sample["pseudo"].to(device).clone()
        weight = torch.from_numpy(sample["weights"][source]).to(device=device, dtype=torch.float32).view_as(pseudo)
        optimizer.zero_grad(set_to_none=True)
        result = interface._network_forward_with_states(student, image, embedding)
        logits = result["final_prediction"][:, 0:1].float()
        loss = (F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none") * weight).sum() / weight.sum().clamp_min(1e-6)
        loss.backward()
        grad_norm = float(torch.sqrt(sum(parameter.grad.float().pow(2).sum() for parameter in trainable if parameter.grad is not None)).detach().cpu())
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer.step()
        update_delta = float(torch.sqrt(sum((parameter.detach() - old).float().pow(2).sum() for parameter, old in zip(trainable, before, strict=True))).detach().cpu())
        losses.append({
            "variant": variant, "order": order, "step": step, "case": sample["case"],
            "augmentation": sample["augmentation"], "loss": float(loss.detach().cpu()),
            "gradient_norm": grad_norm, "update_delta_norm": update_delta,
            "learning_rate": args.learning_rate, "weight_decay": 0.0,
            "base_trainable_parameters": base_trainable,
            "lora_trainable_parameter_tensors": len(trainable),
            "lora_trainable_parameters": sum(p.numel() for p in trainable),
        })
        with torch.inference_mode():
            student.eval()
            for sample_eval in evaluation_cache:
                eval_image = sample_eval["image"].to(device)
                eval_embedding = sample_eval["embedding"].to(device).float()
                prediction = (torch.sigmoid(interface._network_forward_with_states(student, eval_image, eval_embedding)["final_prediction"][:, 0:1]) > args.prediction_threshold).flatten().cpu().numpy()
                eval_rows.append({"variant": variant, "order": order, "step": step, "case": sample_eval["case"], "patch_index": sample_eval["patch_index"], **binary_metrics(prediction, sample_eval["gt_np"])})
            student.train()
        pseudo_rows.append({
            "variant": variant, "order": order, "step": step, "case": sample["case"],
            "augmentation": sample["augmentation"], "reliability_source": source,
            "pseudo_positive_voxels": int(np.count_nonzero(sample["pseudo"].numpy())),
            "weight_sum": float(sample["weights"][source].sum()), "weight_mean": float(sample["weights"][source].mean()),
        })
        del result, logits, loss, image, embedding, pseudo, weight, before
    stats = {"target_modules": targets, "base_trainable_parameters": base_trainable, "lora_parameter_count": sum(p.numel() for p in trainable), "lora_ratio_of_base_model": sum(p.numel() for p in trainable) / max(base_total, 1)}
    del student, optimizer, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return losses, pseudo_rows, eval_rows, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    paths = ProjectPaths(); parser = argparse.ArgumentParser(description="V7.1a LoRA packed-QKV stability smoke")
    parser.add_argument("--model-dir", default=str(paths.voxtell_model_dir)); parser.add_argument("--voxtell-root", default=str(paths.voxtell_root)); parser.add_argument("--data-root", default=str(paths.data_root)); parser.add_argument("--split-json", default=str(paths.split_json))
    parser.add_argument("--world-checkpoint", default="outputs/v3_2e_frozen_input_projection/unified_world_predictor_step200.pt"); parser.add_argument("--output-dir", default="outputs/v7_1a_lora_qkv_smoke"); parser.add_argument("--selected-stage", default="decoder_stage_1_low_to_high")
    parser.add_argument("--train-cases", type=int, default=4); parser.add_argument("--evaluation-cases", type=int, default=4); parser.add_argument("--patches-per-case", type=int, default=2); parser.add_argument("--foreground-patches-per-case", type=int, default=1); parser.add_argument("--foreground-candidate-patches", type=int, default=16); parser.add_argument("--foreground-threshold", type=float, default=0.5); parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--hidden-channels", type=int, default=16); parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--lora-rank", type=int, default=4); parser.add_argument("--lora-alpha", type=float, default=8.0); parser.add_argument("--lora-dropout", type=float, default=0.0); parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--device", default="cuda"); parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed); device = resolve_device(args)
    if args.device == "cuda" and device.type != "cuda": raise RuntimeError(f"V7.1a requires CUDA, resolved {device}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths(voxtell_root=Path(args.voxtell_root), voxtell_model_dir=Path(args.model_dir), data_root=Path(args.data_root), split_json=Path(args.split_json))
    train_cases = iter_image_cases(paths, "train", args.train_cases); eval_cases = iter_cases(paths, "test", args.evaluation_cases)
    train_names = [c.case for c in train_cases]; eval_names = [c.case for c in eval_cases]; overlap = sorted(set(train_names) & set(eval_names)); assert not overlap
    teacher = VoxTellStateInterface.from_model_dir(paths.voxtell_model_dir, device=device, voxtell_root=paths.voxtell_root); prepare_functional_seg_head(teacher, args.selected_stage); prompt = teacher.embed_text_prompts([SOURCE_PROMPT]).detach().cpu()
    checkpoint_path = Path(args.world_checkpoint); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); world = load_world_model(checkpoint_path, int(checkpoint["state_dict"]["output_projection.bias"].shape[0]), device, args.hidden_channels)
    train_cache = build_train_cache(teacher, world, train_cases, prompt, args, device); eval_cache = build_evaluation_cache(teacher, world, eval_cases, prompt, args, device); world.to("cpu"); teacher.network.to("cpu"); teacher.functional_seg_head.to("cpu"); torch.cuda.empty_cache()
    train_samples = []
    for case in train_cases:
        candidates = [s for s in train_cache if s["case"] == case.case and s["patch_kind"] == "foreground" and s["has_foreground"]]
        if not candidates: raise RuntimeError(f"No nonempty foreground patch for {case.case}")
        train_samples.append(sorted(candidates, key=lambda s: s["patch_index"])[0])
    base = copy.deepcopy(teacher.network).cpu().eval(); base_total = sum(p.numel() for p in base.parameters())
    for p in base.parameters(): p.requires_grad = False
    base.to(device); init = evaluate_base_network("A_init_no_adaptation", "forward", 0, base, eval_cache, teacher, device, args.prediction_threshold); base.cpu(); torch.cuda.empty_cache()
    losses=[]; pseudo=[]; step_rows=list(init); final_rows=list(init); target_names=None; stats=[]
    for variant, source in VARIANTS.items():
        for order in ORDERS:
            l,p,e,s=train_one_order(variant,source,order,base,train_samples,eval_cache,teacher,args,device,target_names,base_total)
            target_names=s["target_modules"] if target_names is None else target_names; losses.extend(l); pseudo.extend(p); step_rows.extend(e); final_rows.extend([r for r in e if r["step"]==len(train_samples)]); stats.append({"variant":variant,"order":order,**s})
    # Recompute step-0 baseline for each order/variant via the shared identity row.
    global_rows=metric_pool(step_rows,"global"); final_global=metric_pool(final_rows,"global"); final_case=metric_pool(final_rows,"case")
    init_global=next(r for r in final_global if r["variant"]=="A_init_no_adaptation")
    init_by_case={r["case"]: r for r in final_case if r["variant"]=="A_init_no_adaptation"}
    for r in final_global+final_case:
        if r["variant"]!="A_init_no_adaptation":
            baseline = init_global if r["case"] == "__global__" else init_by_case[r["case"]]
            for m in ("dice","precision","recall"): r[f"delta_{m}_vs_A_init"]=r[m]-baseline[m]
    lookup={(r["variant"],r["order"]):r for r in final_global}; sensitivity={}
    for v in VARIANTS:
        f=lookup[(v,"forward")]; rev=lookup[(v,"reverse")]; sensitivity[v]={f"forward_minus_reverse_{m}":f[m]-rev[m] for m in ("dice","precision","recall")}
    gpu_name=torch.cuda.get_device_name(device); summary={"stage":"V7.1a LoRA-QKV stability smoke","source_checkpoint":str(checkpoint_path),"resolved_device":str(device),"gpu_name":gpu_name,"peak_cuda_allocated_mb":torch.cuda.max_memory_allocated(device)/1024**2,"peak_cuda_reserved_mb":torch.cuda.max_memory_reserved(device)/1024**2,"seed":args.seed,"adaptation_cases":train_names,"evaluation_cases":eval_names,"case_overlap":overlap,"case_overlap_count":len(overlap),"effective_updates":len(train_samples),"variants":VARIANTS,"orders":list(ORDERS),"lora":{"rank":args.lora_rank,"alpha":args.lora_alpha,"dropout":args.lora_dropout,"target_modules":target_names,"base_trainable_parameters":0,"stats":stats},"training":{"loss":"V7.0d weighted pseudo-label BCE","learning_rate":args.learning_rate,"weight_decay":0.0,"student_trainable_scope":"LoRA parameters only; all VoxTell base parameters frozen","world_predictor_updated":False,"same_cache_and_strong_view":True},"final_global_metrics":final_global,"order_sensitivity":sensitivity,"step1_and_step4":{"step1":[r for r in global_rows if r["step"]==1],"step4":[r for r in global_rows if r["step"]==4]},"outputs":{k:str(out/f) for k,f in {"per_step_eval":"per_step_eval.csv","training_loss":"training_loss.csv","pooled_by_case":"pooled_by_case.csv","pooled_global":"pooled_global.csv","lora_targets":"lora_targets.json","parameter_stats":"parameter_stats.json","summary":"summary.json"}.items()},"status":"smoke_complete; no 10-50 step training"}
    write_csv(out/"training_loss.csv",losses); write_csv(out/"per_step_eval.csv",global_rows); write_csv(out/"pooled_global.csv",final_global); write_csv(out/"pooled_by_case.csv",final_case); write_csv(out/"pseudo_label_stats.csv",pseudo)
    (out/"lora_targets.json").write_text(json.dumps(target_names,indent=2)); (out/"parameter_stats.json").write_text(json.dumps(stats,indent=2)); (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__ == "__main__": run(parse_args())
