# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
from typing import Callable, Union, List, Tuple
import warnings
import torch
from torchvision.transforms import FiveCrop, TenCrop
from torchvision.transforms import Resize, RandomResizedCrop
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode

from ...types import BatchType
from .resize import FULLRES_SUFFIX
from .utils import _get_relevant_spatial_keys


class TorchTransformWrapper:
    def __init__(
        self,
        transform_obj: Callable[[torch.Tensor], torch.Tensor],
        keys: Union[str, List[str], Tuple[str]] = None
    ) -> None:
        """
        This class enables using torchvision transforms when you have a
        multimodal input. For example, if both RGB and depth image should be
        randomly cropped with the same cropping parameters. This class first
        concatenates all  relevant spatial inputs, puts the result through the
        provided torchvision transform and splits the final result again along
        the channel axis. Note that this only works if all inputs are torch
        tensors.
        """
        super().__init__()
        self.raise_error_if_resize(transform_obj)
        self._transform_obj = transform_obj
        if isinstance(keys, str):
            keys = [keys]
        self._keys = keys

    def __call__(self, sample: BatchType) -> BatchType:
        # automatically select keys from the sample if no keys were specified
        if self._keys is None:
            keys = _get_relevant_spatial_keys(sample)
            keys = [key for key in keys if FULLRES_SUFFIX not in key]
        else:
            keys = self._keys

        # collect all tensors in a list and save their shape
        shapes = {}
        stacked_tensor = []
        for key in keys:
            assert key in sample
            # only torch tensors are supported. Please use ToTorchTensor()
            # before this TorchTransformWrapper.
            if not torch.is_tensor(sample[key]):
                warnings.warn(f"{key} is not a torch tensor! Skipping...")
                continue
            value = sample[key].clone()
            shapes[key] = value.shape
            # masks typically only have 2 dimensions. We add one dimension for
            # concatenating, which will be removed afterwards again
            if 2 == value.ndim:
                value = value.unsqueeze(dim=0)
            if 3 == value.ndim:
                stacked_tensor.append(value)

        # check if all tensors have the same data type
        dtypes = [x.dtype for x in stacked_tensor]
        if len(set(dtypes)) > 1:
            key_dtype = {key: dtype for key, dtype in zip(keys, dtypes)}
            warnings.warn(
                f"Tensors have different datatypes! {key_dtype}"
            )

        # concatenate all tensors and apply transform
        stacked_tensor = torch.concat(stacked_tensor)
        out = self._transform_obj(stacked_tensor)

        n_channels = [v[0] if 3 == len(v) else 1 for v in shapes.values()]

        # multi crop stuff:
        if isinstance(self._transform_obj, (FiveCrop, TenCrop)):
            out_dict = {key: [] for key in shapes.keys()}
            # first split each crop
            for o in out:
                splitted_tensors = torch.split(o, n_channels)
                # accumulate the crops for each key
                for i, key in enumerate(shapes.keys()):
                    out_dict[key].append(splitted_tensors[i])
            for key, value in out_dict.items():
                # stack the crops
                sample[key] = torch.stack(value).type(sample[key].dtype)

        else:
            # split tensor again and assign to sample with original data type
            splitted_tensors = torch.split(out, n_channels)
            for i, key in enumerate(shapes.keys()):
                sample[key] = splitted_tensors[i].type(sample[key].dtype)

        # remove additional dimension again
        for key, shape in shapes.items():
            if sample[key].ndim != len(shape):
                sample[key] = sample[key].squeeze(dim=0)

        return sample

    def raise_error_if_resize(
        self,
        transform_obj: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        if isinstance(transform_obj, Compose):
            for trans_obj in transform_obj.transforms:
                self.raise_error_if_resize(trans_obj)
        if isinstance(transform_obj, (Resize, RandomResizedCrop)):
            if transform_obj.interpolation != InterpolationMode.NEAREST:
                raise ValueError(
                    "Resize operations are only supported with nearest "
                    f"interpolation! Got {transform_obj.interpolation}. "
                    "Other interpolation methods will lead to unexpected "
                    "results for e.g. depth, segmentation, instance, ..."
                )
