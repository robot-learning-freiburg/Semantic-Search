# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from nicr_scene_analysis_datasets.dataset_base import (
    DatasetConfig,
    OrientationDict,
    RGBDDataset,
    SampleIdentifier,
    build_dataset_config,
)

from sem_objnav.segment.hm3d.hm3d import Hm3dMeta


class Hm3d(Hm3dMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = "train",
        sample_keys: Tuple[str] = ("rgb", "depth", "semantic"),
        use_cache: bool = False,
        depth_mode: str = "raw",
        semantic_n_classes: int = 81,
        cameras: Optional[Tuple[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs,
        )
        print(split)
        assert split in self.SPLITS
        assert depth_mode in self.DEPTH_MODES
        assert all(sk in self.get_available_sample_keys(split) for sk in sample_keys)
        self._semantic_n_classes = semantic_n_classes
        self._split = split
        self._depth_mode = depth_mode
        # cameras
        if cameras is None:
            # use all available cameras (=default dummy camera)
            self._cameras = self.CAMERAS
        else:
            # use subset of cameras (does not really apply to this dataset)
            assert all(c in self.CAMERAS for c in cameras)
            self._cameras = cameras

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load filenames
            fp = Path(self._dataset_path) / self._split / "rgb"
            self._filenames = sorted([p.name for p in fp.glob("*.png")])
        elif not self._disable_prints:
            print(f"Loaded HM3D dataset without files")

        # build config object
        semantic_label_list = getattr(
            self, f"SEMANTIC_LABEL_LIST_{self._semantic_n_classes}"
        )
        self._config = build_dataset_config(
            semantic_label_list=semantic_label_list,
            scene_label_list=[],
            depth_stats=self.TRAIN_SPLIT_DEPTH_STATS,
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    @property
    def cameras(self) -> Tuple[str]:
        return self._cameras

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def split(self) -> str:
        return self._split

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    def __len__(self) -> int:
        return len(self._filenames)

    def _load(self, directory: str, filename: str) -> np.ndarray:
        fp = os.path.join(self._dataset_path, self.split, directory, filename)

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.ndarray:
        return self._load(self.RGB_DIR, self._filenames[idx])

    def _load_depth(self, idx) -> np.ndarray:
        return self._load(self.DEPTH_DIR, self._filenames[idx])

    def _load_identifier(self, idx: int) -> Tuple[str]:
        return SampleIdentifier((self._filenames[idx],))

    def _load_semantic(self, idx: int) -> np.ndarray:
        return self._load(
            self.SEMANTIC_DIR_FMT.format(self._semantic_n_classes), self._filenames[idx]
        )

    def _load_instance(self, idx: int) -> np.array:
        instance = self._load(self.INSTANCES_DIR, self._filenames[idx])
        return instance.astype("int32")

    def _load_orientations(self, idx: int) -> Dict[int, float]:
        fp = os.path.join(
            self._dataset_path,
            self.split,
            self.ORIENTATIONS_DIR,
            f"{self._filenames[idx]}.json",
        )
        with open(fp, "r") as f:
            orientations = json.load(f)

        orientations = {int(k): v for k, v in orientations.items()}
        return OrientationDict(orientations)

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return Hm3dMeta.SPLIT_SAMPLE_KEYS[split]
