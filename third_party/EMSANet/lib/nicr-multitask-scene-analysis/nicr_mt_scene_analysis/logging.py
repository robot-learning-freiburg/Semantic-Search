# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict

import csv
import os

import torch


class CSVLogger:
    def __init__(self, filepath: str, write_interval: int = 1) -> None:
        self._filepath = filepath
        self._write_interval = write_interval

        if os.path.isfile(filepath):
            # read existing file (resumed training)
            with open(filepath, 'r') as f:
                csv_reader = csv.DictReader(f)
                self._rows = list(csv_reader)
        else:
            self._rows = []

    def write(self) -> None:
        # determine keys
        unique_keys = set()
        for log in self._rows:
            unique_keys.update(list(log.keys()))

        # write logs to file
        with open(self._filepath, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=sorted(unique_keys))
            csv_writer.writeheader()
            csv_writer.writerows(self._rows)

    def log(self, logs: Dict[str, Any]) -> None:
        # convert to simple types
        row = {}
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                row[key] = value.cpu().item()
            elif isinstance(value, (float, int, str)):
                row[key] = value
            else:
                raise NotImplementedError(
                    f"CSV logging for type: '{type(value)}' is not yet "
                    f"implemented."
                )
        # add row
        self._rows.append(row)

        # write csv file
        if 0 == (len(self._rows)-1) % self._write_interval:
            self.write()

    def __del__(self):
        # avoid data loss
        self.write()
