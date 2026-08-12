from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from vls.augmentations import gamma_augment
from vls.config import DEFAULT_LABEL_VALUE, DEFAULT_PROMPTS, ProjectPaths
from vls.data import binary_gt_from_label, iter_cases, read_image_and_label
from vls.metrics import binary_metrics
from vls.voxtell_states import ensure_voxtell_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0 VoxTell baselines and state-interface smoke tests.")
    parser.add_argument("--voxtell-root", default=str(ProjectPaths().voxtell_root))
    parser.add_argument("--model-dir", default=str(ProjectPaths().voxtell_model_dir))
    parser.add_argument("--data-root", default=str(ProjectPaths().data_root))
    parser.add_argument("--split-json", default=str(ProjectPaths().split_json))
    parser.add_argument("--split", default="test", choices=["train", "test", "train_cases", "test_cases"])
    parser.add_argument("--output-dir", default="outputs/v0")
    parser.add_argument("--baseline", default="B0", choices=["B0", "B1", "B2"])
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--label-value", type=int, default=DEFAULT_LABEL_VALUE)
    parser.add_argument("--limit-cases", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--state-smoke", action="store_true")
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def predict_baseline(predictor, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.baseline == "B0":
        return predictor.predict_single_image(image, args.prompts, output_type="binary")

    if args.baseline == "B1":
        original = predictor.predict_single_image(image, args.prompts, output_type="probabilities")
        augmented = predictor.predict_single_image(gamma_augment(image, args.gamma), args.prompts, output_type="probabilities")
        return ((original + augmented) * 0.5 > 0.5).astype(np.uint8)

    if args.baseline == "B2":
        probabilities = [
            predictor.predict_single_image(image, [prompt], output_type="probabilities")[0]
            for prompt in args.prompts
        ]
        mean_probability = np.mean(np.stack(probabilities, axis=0), axis=0, keepdims=True)
        return (mean_probability > 0.5).astype(np.uint8)

    raise ValueError(f"Unsupported baseline: {args.baseline}")


def run_state_smoke(predictor, image: np.ndarray, prompts: list[str], device: torch.device) -> dict[str, object]:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    from vls.voxtell_states import VoxTellStateInterface

    interface = VoxTellStateInterface(predictor)
    preprocessed, _, _ = predictor.preprocess(image)
    padded, _ = pad_nd_image(preprocessed, predictor.patch_size, "constant", {"value": 0}, True, None)
    slicer = predictor._internal_get_sliding_window_slicers(padded.shape[1:])[0]
    patch = torch.clone(padded[slicer][None], memory_format=torch.contiguous_format)
    result = interface.forward_with_states(patch, prompts)
    return {
        "final_prediction": list(result["final_prediction"].shape),
        "decoder_states": {key: list(value.shape) for key, value in result["decoder_states"].items()},
        "intermediate_predictions": {
            key: list(value.shape) for key, value in result["intermediate_predictions"].items()
        },
        "device": str(device),
    }


def run(args: argparse.Namespace) -> None:
    paths = ProjectPaths(
        voxtell_root=Path(args.voxtell_root),
        voxtell_model_dir=Path(args.model_dir),
        data_root=Path(args.data_root),
        split_json=Path(args.split_json),
    )
    ensure_voxtell_on_path(paths.voxtell_root)
    from voxtell.inference.predictor_multiclass import VoxTellPredictor

    device = resolve_device(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = iter_cases(paths, split=args.split, limit=args.limit_cases)
    predictor = VoxTellPredictor(model_dir=str(paths.voxtell_model_dir), device=device)

    rows = []
    state_summary = None
    for case in cases:
        image, label_map, _ = read_image_and_label(case)
        prediction = predict_baseline(predictor, image, args)
        gt = binary_gt_from_label(label_map, args.label_value)
        for prompt_index, prompt in enumerate(args.prompts[: prediction.shape[0]]):
            metrics = binary_metrics(prediction[prompt_index], gt)
            rows.append({
                "case": case.case,
                "baseline": args.baseline,
                "prompt": prompt,
                **metrics,
            })

        if args.state_smoke and state_summary is None:
            state_summary = run_state_smoke(predictor, image, args.prompts, device)

    metrics_path = output_dir / f"{args.baseline.lower()}_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "baseline", "prompt", "dice", "iou", "pred_voxels", "gt_voxels", "intersection"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "args": vars(args),
        "num_cases": len(cases),
        "mean_dice": float(np.mean([row["dice"] for row in rows])) if rows else 0.0,
        "mean_iou": float(np.mean([row["iou"] for row in rows])) if rows else 0.0,
        "metrics_csv": str(metrics_path),
        "state_smoke": state_summary,
    }
    summary_path = output_dir / f"{args.baseline.lower()}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
