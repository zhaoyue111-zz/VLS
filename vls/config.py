from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    voxtell_root: Path = Path("/data/zy/VoxTell_from_disk")
    voxtell_model_dir: Path = Path("/data/zy/VoxTell_from_disk/model")
    data_root: Path = Path("/data/zy/CT_MRI_DATA_3D")
    split_json: Path = Path("worst_zeroshot_split_p0/worst_zeroshot_split.json")

    @property
    def image_dir(self) -> Path:
        return self.data_root / "images" / "P0"

    @property
    def label_dir(self) -> Path:
        return self.data_root / "labels" / "P0"


DEFAULT_PROMPTS = ["liver"]
DEFAULT_LABEL_VALUE = 5
