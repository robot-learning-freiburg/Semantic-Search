# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Optional

import torch
from torchmetrics import Metric


class RootMeanSquaredError(Metric):
    full_state_update = False

    """
    Computes Root Mean Squared Error (RMSE) (averaged over pixels).
    """
    def __init__(self) -> None:
        super().__init__()

        # note that states are automatically reset to their defaults when
        # calling metric.reset()
        self.add_state('sum_root_mean_squared_error',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_observations',
                       default=torch.tensor(0, dtype=torch.int64),
                       dist_reduce_fx='sum')

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """Update state with predictions and targets by only considering valid
        pixels when `mask` is given.

        Args:
            preds: Predictions from model. Shape: (BCHW)
            target: Ground truth values. Shape: (BCHW)
            mask: Boolean mask for valid pixels to consider. Shape: (BHW)
        """
        squared_error = (preds - target) ** 2
        # average only along channel axis to get rmse per pixel
        mean_squared_error = torch.mean(squared_error, dim=1)
        rmse_per_pixel = torch.sqrt(mean_squared_error)

        if mask is not None:
            # accumulate valid rmse values
            # as the number of valid pixels varies from batch to batch, we can
            # not compute average right away here
            self.sum_root_mean_squared_error += torch.sum(rmse_per_pixel[mask])
            self.n_observations += mask.sum()
        else:
            self.sum_root_mean_squared_error += torch.sum(rmse_per_pixel)
            self.n_observations += rmse_per_pixel.numel()

    def compute(self) -> torch.Tensor:
        """Computes root mean squared error over state."""
        rmse = self.sum_root_mean_squared_error / self.n_observations
        return rmse.to(torch.float32)
