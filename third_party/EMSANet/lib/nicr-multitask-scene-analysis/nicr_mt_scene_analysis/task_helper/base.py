# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import abc
from functools import wraps
from typing import Any, Dict, List, Optional, Sequence, Tuple
from time import perf_counter
import warnings

import torch

from ..data.preprocessing.multiscale_supervision import get_downscale
from ..types import BatchType


TOTAL_LOSS_SUFFIX = '_total_loss'


def get_total_loss_key(key):
    return f'{key}{TOTAL_LOSS_SUFFIX}'


def append_detached_losses_to_logs(disabled=False):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if disabled:
                return f(*args, **kwargs)

            # apply train / validation step function
            losses, logs = f(*args, **kwargs)

            # append detached losses to logs
            logs.update({
                k: v.detach().clone()
                for k, v in losses.items()
            })

            return losses, logs

        return wrapper

    return decorator
    pass


def append_profile_to_logs(key, disabled=False):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if disabled:
                return f(*args, **kwargs)

            start = perf_counter()
            results = f(*args, **kwargs)
            end = perf_counter()
            assert isinstance(results[-1], dict)  # last element is dict of logs
            results[-1][key] = end - start

            return results

        return wrapper

    return decorator


class TaskHelperBase(abc.ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, device: torch.device):
        pass

    def collect_predictions_and_targets_for_loss(
        self,
        batch: BatchType,
        batch_key: str,
        predictions_post: BatchType,
        predictions_post_key: str,
        side_outputs_key: Optional[str] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        # collect inputs
        input_tensors, keys, downscales = self.collect_predictions_for_loss(
            predictions_post=predictions_post,
            predictions_post_key=predictions_post_key,
            side_outputs_key=side_outputs_key
        )
        # collect targets
        target_tensors = self.collect_targets_for_loss(
            batch=batch,
            batch_key=batch_key,
            downscales=downscales
        )
        return input_tensors, target_tensors, keys

    def collect_predictions_for_loss(
        self,
        predictions_post: BatchType,
        predictions_post_key: str,
        side_outputs_key: Optional[str] = None
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # main output
        prediction_tensors = [predictions_post[predictions_post_key]]
        keys = ['main']
        downscales = []

        # side outputs
        # to get the matching targets form the given batch in
        # 'collect_targets_for_loss' later, we have to determine the downscales
        # of the optional side outputs by comparing the spatial resolution
        # (both width and height behave the same way)
        def _get_width(output):
            if isinstance(output, torch.Tensor):
                # side output is a simple tensor
                return output.shape[-1]
            if isinstance(output, tuple):
                # side output is a tuple of tensors (instance task)
                return output[0].shape[-1]
            raise Exception("Error while determining downscale")

        # parse side outputs
        if side_outputs_key is not None:
            # part1: determine downscales (scales) in a quite ugly way
            width_main = _get_width(predictions_post[predictions_post_key])

            # append tensors
            for side_output in predictions_post[side_outputs_key]:
                if side_output is None:
                    # there are no side outpus or we are in eval mode
                    continue

                prediction_tensors.append(side_output)

                # part2: determine downscales (scales) in a quite ugly way
                width_side = _get_width(side_output)
                downscales.append(width_main // width_side)
                keys.append(f'down_{downscales[-1]}')

        return prediction_tensors, keys, downscales

    def collect_targets_for_loss(
        self,
        batch: BatchType,
        batch_key: str,
        downscales: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], List[str]]:
        # main target
        target_tensors = [batch[batch_key]]

        # side outputs
        if downscales is not None:
            for downscale in downscales:
                batch_downscale = get_downscale(batch, downscale)
                if batch_downscale is None:
                    # mutliscale is disabled
                    continue
                target_tensors.append(batch_downscale[batch_key])
        return target_tensors

    def accumulate_losses(
        self,
        losses: Sequence[torch.Tensor],
        n_elements: Sequence[int]
    ) -> torch.Tensor:
        # compute total loss over main output and opt. side outputs

        # our default reduction is to divide the accumulated losses by the
        # total number of elements that contributed to this total loss

        # note, due to the smaller resolution of the side outputs their weight
        # gets smaller as the resolution decreases

        total_loss = torch.sum(torch.stack(losses))
        total_n_elements = sum(n_elements)

        if 0 == total_n_elements:
            warnings.warn("Total number of loss elements is 0. Returning 0 as  "
                          "loss to avoid division by zero.")
            return total_loss     # sum of [] is 0

        return total_loss/total_n_elements

    def mark_as_total(self, key: str) -> str:
        return get_total_loss_key(key)

    @abc.abstractmethod
    def training_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute (loss_dict, log_dict)
        pass

    @abc.abstractmethod
    def validation_step(
        self,
        batch: BatchType,
        batch_idx: int,
        predictions_post: BatchType
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # compute (loss_dict, log_dict)
        pass

    @abc.abstractmethod
    def validation_epoch_end(self):
        # compute (artifact_dict, example_dict, log_dict)
        pass
