# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Some parts of this code are based on:
    https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
"""
from typing import Optional, Sequence, Tuple

from collections import Counter

import cv2
import numpy as np
from PIL import Image


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


class PanopticColorGenerator:
    def __init__(
        self,
        classes_colors: Sequence[Tuple[int, int, int]],
        classes_is_thing: Tuple[bool],
        max_instances: int = 100,
        void_label: int = 0
    ) -> None:
        assert len(classes_colors) == len(classes_is_thing)
        self._classes_colors = classes_colors
        self._classes_is_thing = classes_is_thing
        self._max_instances = max_instances
        self._max_classes = len(self._classes_colors)
        self._void_label = void_label

        self._taken_colors = set((0, 0, 0))    # exclude black as it is void
        self._color_counter = Counter()

        # color cache as cmap
        self._cmap = np.zeros(
            (self._max_instances*self._max_classes, 3),
            dtype='uint8'
        )
        self._known_panoptic_ids = np.zeros(
            (max_instances*self._max_classes),
            dtype='bool'
        )

        for i in range(len(self._classes_is_thing)):
            if not self._classes_is_thing[i]:
                # stuff class
                self._taken_colors.add(self._classes_colors[i])

    def get_colored_image(self, panoptic_img: np.ndarray):
        # ensure that all colors are present in internal color map
        for pan_id in np.unique(panoptic_img):
            sem = pan_id // self._max_instances

            if sem == self._void_label:
                # skip void -> keep black (see initialization)
                continue
            if not self._known_panoptic_ids[pan_id]:    # pan_id is unknown
                # add new color to cmap
                self._known_panoptic_ids[pan_id] = True
                self._cmap[pan_id] = self._get_new_color(sem)

        # note that this is faster than working masks in a loop
        return self._cmap[panoptic_img]

    def _get_next_hsv_color(
        self,
        base: Tuple[int, int, int],
        s_step: int = round(255*0.05),
        v_step: int = round(255*0.01)
    ) -> Tuple[int, int, int]:
        base = np.array(base, dtype=np.uint8)
        base_hsv = cv2.cvtColor(base.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0][0]

        current_ctr = self._color_counter[str(base_hsv)]
        self._color_counter[str(base_hsv)] += 1
        base_hsv[1] = (base_hsv[1]-s_step*current_ctr) % 255
        base_hsv[2] = (base_hsv[2]-v_step*current_ctr) % 255
        base_rgb = cv2.cvtColor(base_hsv.reshape(1, 1, 3),
                                cv2.COLOR_HSV2RGB)[0][0]
        return tuple(base_rgb)

    def _get_new_color(self, semantic_class_idx: int):
        if not self._classes_is_thing[semantic_class_idx]:
            # stuff class
            return self._classes_colors[semantic_class_idx]

        # thing class
        def random_color(base, max_dist=45):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        base_color = self._classes_colors[semantic_class_idx]
        if base_color not in self._taken_colors:
            # first instance of thing class gets the color of the thing class
            self._taken_colors.add(base_color)
            return base_color
        else:
            # otherwise determine a new color
            while True:
                color = random_color(np.array(base_color))
                # color = self._get_next_hsv_color(base_color)
                if color not in self._taken_colors:
                    self._taken_colors.add(color)
                    return color


def visualize_panoptic(
    panoptic_img: np.ndarray,
    semantic_classes_colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    semantic_classes_is_thing: Optional[Tuple[bool]] = None,
    max_instances: Optional[int] = None,
    void_label: Optional[int] = None,
    shared_color_generator: Optional[PanopticColorGenerator] = None
) -> np.ndarray:
    if shared_color_generator is None:
        assert semantic_classes_colors is not None
        assert semantic_classes_is_thing is not None
        assert max_instances is not None
        assert void_label is not None
        color_generator = PanopticColorGenerator(
            classes_colors=semantic_classes_colors,
            classes_is_thing=semantic_classes_is_thing,
            max_instances=max_instances,
            void_label=void_label
        )
    else:
        color_generator = shared_color_generator

    return color_generator.get_colored_image(panoptic_img)


def visualize_panoptic_pil(
    panoptic_img: np.ndarray,
    semantic_classes_colors: Optional[Sequence[Tuple[int, int, int]]] = None,
    semantic_classes_is_thing: Optional[Tuple[bool]] = None,
    max_instances: Optional[int] = None,
    void_label: Optional[int] = None,
    shared_color_generator: Optional[PanopticColorGenerator] = None
) -> Image.Image:
    panoptic_img = visualize_panoptic(
        panoptic_img,
        semantic_classes_colors=semantic_classes_colors,
        semantic_classes_is_thing=semantic_classes_is_thing,
        max_instances=max_instances,
        void_label=void_label,
        shared_color_generator=shared_color_generator
    )
    return Image.fromarray(panoptic_img)
