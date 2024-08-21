# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Sequence, Tuple

from copy import deepcopy


class CheckpointHelper:
    def __init__(
        self,
        metric_names: Optional[Sequence[str]] = None,
        debug: bool = True
    ) -> None:
        # metric_names can contain a single metric or a combination of metrics,
        # e.g., ('miou', 'bacc', 'miou+bacc') leads to checkpointing when
        # either miou, bacc, or the sum of miou and bacc reaches its highest
        # value
        # note that current implemention only supports combining metrics
        # using "+". None disables checkpointing

        # initialize variables for mapping
        if metric_names is None:
            self._metric_mapping = None
        else:
            self._metric_mapping = {name: [] for name in metric_names}
        self._metrics_determined = False

        self._debug = debug
        self._cache_bests = {}

    @property
    def metric_mapping(self) -> Dict[str, Tuple[str]]:
        return self._metric_mapping

    @property
    def metric_mapping_joined(self) -> Dict[str, str]:
        if self._metric_mapping is None:
            return {}

        return {n: '+'.join(ms) for n, ms in self._metric_mapping.items()}

    @staticmethod
    def _determine_checkpoint_metrics(
        to_search_for: str,
        logs: Dict[str, Any]
    ) -> Tuple[str]:
        matched_metrics = []

        for m in to_search_for.split('+'):    # split combined metrics
            # determine suitable metric from logs
            candidates = []
            for log_metric in logs:
                if 'best' in log_metric or 'valid' not in log_metric:
                    continue

                if m in log_metric:
                    candidates.append(log_metric)

            if len(candidates) == 0:
                raise ValueError(
                    f"No suitable metric found for '{m}'. "
                    f"Available keys for matching: {list(logs.keys())}"
                )
            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple suitable metrics: '{candidates}' for "
                    f"'{m}' found."
                )

            matched_metrics.append(candidates[0])

        return tuple(matched_metrics)

    @staticmethod
    def _is_new_better(metric, new_value, old_value) -> bool:
        """Determines whether an incoming value is better than the previous"""
        larger = any(to_find in metric
                     for to_find in ('miou', 'acc', 'rq', 'sq', 'pq'))
        smaller = any(to_find in metric
                      for to_find in ('mae', 'rmse'))

        if not (larger ^ smaller):     # xor, at least one but only one
            raise ValueError(f"Cannot determine better value for '{metric}' "
                             f"(new: '{new_value}' '>' vs. '<' old: "
                             f"'{old_value}').")

        if old_value is None:
            return True

        if larger:
            return new_value > old_value

        if smaller:
            return new_value < old_value

    def check_for_checkpoint(
        self,
        logs: Dict[str, Any],
        add_checkpoint_metrics_to_logs: bool = True
    ) -> Dict[str, bool]:
        """Determines for all metric names whether a new checkpoint should be
        created"""
        if self._metric_mapping is None:
            # checkpointing is disabled
            return {}

        # lazily try to get suitable metrics from logs
        if not self._metrics_determined:
            for name in self._metric_mapping:
                self._metric_mapping[name] = self._determine_checkpoint_metrics(
                    to_search_for=name, logs=logs
                )
            if self._debug:
                print(f"Using '{self._metric_mapping}' for checkpointing.")

            self._metrics_determined = True

        # model should be checkpointed if a new best value is reached
        do_create_checkpoint = {}
        for name, metrics in self._metric_mapping.items():
            # old value
            old_value = self._cache_bests.get(name, None)    # first call: None

            # new value
            new_value = logs[metrics[0]]
            # handle combined metrics (simply sum values)
            assert len(metrics) == 1 or (len(metrics) > 1 and '+' in name)
            for m in metrics[1:]:
                new_value = new_value + logs[m]

            # check for checkpointing
            if self._is_new_better(metric=name,
                                   new_value=new_value, old_value=old_value):
                if self._debug:
                    print(f"Checkpoint metric '{name}: {metrics}' reached new "
                          f"best value! (new: '{new_value}', old: "
                          f"'{old_value}')")

                self._cache_bests[name] = new_value
                do_create_checkpoint[name] = True

            # add (combined) metric to logs
            if add_checkpoint_metrics_to_logs:
                full_mapped_name = self.metric_mapping_joined[name]
                logs[f'ckpt_{full_mapped_name}'] = deepcopy(new_value)

        return do_create_checkpoint
