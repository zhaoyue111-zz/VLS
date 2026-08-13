from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from vls.config import DEFAULT_LABEL_VALUE, ProjectPaths


@dataclass(frozen=True)
class CaseRecord:
    case: str
    image_path: Path
    label_path: Path


def load_split(split_json: Path) -> dict:
    with split_json.open() as f:
        return json.load(f)


def iter_cases(paths: ProjectPaths, split: str = "test", limit: int = 0) -> list[CaseRecord]:
    split_data = load_split(paths.split_json)
    split_key = {"train": "train_cases", "test": "test_cases"}.get(split, split)
    case_names = split_data[split_key]
    if limit:
        case_names = case_names[:limit]

    cases: list[CaseRecord] = []
    for case_name in case_names:
        image_path = paths.image_dir / case_name
        label_path = paths.label_dir / case_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")
        cases.append(CaseRecord(case_name, image_path, label_path))
    return cases


def iter_image_cases(paths: ProjectPaths, split: str = "train", limit: int = 0) -> list[CaseRecord]:
    """List cases for image-only adaptation without touching target labels."""
    split_data = load_split(paths.split_json)
    split_key = {"train": "train_cases", "test": "test_cases"}.get(split, split)
    case_names = split_data[split_key]
    if limit:
        case_names = case_names[:limit]
    cases: list[CaseRecord] = []
    for case_name in case_names:
        image_path = paths.image_dir / case_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        cases.append(CaseRecord(case_name, image_path, paths.label_dir / case_name))
    return cases


def read_image_and_label(case: CaseRecord) -> tuple[np.ndarray, np.ndarray, dict]:
    reader = NibabelIOWithReorient()
    image, props = reader.read_images([str(case.image_path)])
    label, _ = reader.read_images([str(case.label_path)])
    return image, label[0], props


def read_image(case: CaseRecord) -> tuple[np.ndarray, dict]:
    """Read only the image, keeping adaptation code independent of target GT."""
    reader = NibabelIOWithReorient()
    image, props = reader.read_images([str(case.image_path)])
    return image, props


def binary_gt_from_label(label_map: np.ndarray, label_value: int = DEFAULT_LABEL_VALUE) -> np.ndarray:
    return (label_map == label_value).astype(np.uint8)
