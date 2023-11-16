"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Web Traffic dataset class.
"""
import json
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from overrides import override

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.data.web_traffic_generation import _create_week_with_seasonality


class WebTrafficDataset(MILTSCDataset):
    """MIL TSC Dataset implementation for the synthetic WebTraffic dataset."""

    def __init__(self, split: str, name: str = "WebTraffic", apply_transform: bool = True):
        super().__init__(name, split, apply_transform=apply_transform)
        # Load dataset metadata
        metadata_path = "data/WebTraffic/{:s}_{:s}_metadata.json".format(self.dataset_name, split.upper())
        with open(metadata_path, "r") as f:
            self._metadata: List[Dict[str, Any]] = json.load(f)

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load list of time series from csv and their targets (classes).

        :param split: Dataset split to load, e.g. train or test.
        :return: List of time series tensors, tensor of classes
        """
        df = pd.read_csv("data/WebTraffic/{:s}_{:s}.csv".format(self.dataset_name, split.upper()))
        ts_collection = []
        targets = []
        for row_idx in range(len(df)):
            ts_pd = df.iloc[row_idx, 1:]
            target = df.iloc[row_idx, 0]
            ts_tensor = torch.as_tensor(ts_pd.to_numpy(), dtype=torch.float)
            ts_tensor = ts_tensor.unsqueeze(1)
            ts_collection.append(ts_tensor)
            targets.append(target)
        targets_tensor = torch.as_tensor(targets, dtype=torch.int)
        return ts_collection, targets_tensor

    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata associated with this dataset.

        :return: A list the size of the dataset, and provides a dictionary for each timestep.
        """
        return self._metadata

    def get_signature_locations(self, idx: int) -> List:
        """
        Get location of discriminatory region for time series.

        :param idx: Dataset index.
        :return: List of signature locations.
        """
        return self._metadata[idx]["signature_locations"]

    def get_original_time_series(self, idx: int) -> torch.Tensor:
        """
        Get the original time series before signature injection.
        Note the random noise (sample) will be different, but the rate will be the same.

        :param idx: Dataset idx.
        :return: Time series as a tensor.
        """
        ts, _ = _create_week_with_seasonality(**self._metadata[idx]["time_series_params"])
        return torch.as_tensor(ts)

    @override
    def __getitem__(self, idx: int) -> Dict:
        """
        Override get dataset item as we as need to return instance targets (discriminatory regions).

        :param idx: Dataset index.
        :return: Dictionary containing bag, target, and instance targets.
        """
        # Get bag
        bag = self.get_bag(idx)
        if self.apply_transform:
            bag = self.apply_bag_transform(bag)
        # Get target
        target = self.targets[idx]
        # Convert signature locations to instance targets
        if target == 0:
            instance_targets = torch.ones(len(bag))
        else:
            signature_locs = self.get_signature_locations(idx)
            instance_targets = torch.zeros(len(bag))
            for signature_loc in signature_locs:
                sig_start, sig_end = signature_loc
                if sig_start == sig_end:
                    instance_targets[sig_start] = 1
                else:
                    instance_targets[sig_start:sig_end] = 1
        return {
            "bag": bag,
            "target": target,
            "instance_targets": instance_targets,
        }
