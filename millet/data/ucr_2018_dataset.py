"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

MIL TSC dataset implementation for UCR Archive 2018.
UCR Time Series Classification datasets provided by https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
"""
from typing import List, Tuple

import pandas as pd
import torch

from millet.data.mil_tsc_dataset import MILTSCDataset


class UCRDataset(MILTSCDataset):
    """MIL TSC Dataset implementation for UCR TSC datasets."""

    def __init__(self, dataset_name: str, split: str):
        super().__init__(dataset_name, split)

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load list of time series from csv and their targets (classes).
        Will also adjust class labels if required to meet our standard assumption that they are  0,...,c.

        :param split: Dataset split to load, e.g. train or test.
        :return: List of time series tensors, tensor of classes
        """
        # Load dataframe
        csv_path = "data/UCR/{:s}/{:s}_{:s}.tsv".format(self.dataset_name, self.dataset_name, split.upper())
        df = pd.read_csv(csv_path, sep="\t", header=None)
        # Parse dataframe
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
        # Adjust targets so zero indexed
        unique_targets = sorted(torch.unique(targets_tensor))
        # -1, 1 case
        if -1 in unique_targets:
            if 0 in unique_targets:
                raise ValueError
            targets_tensor[targets_tensor == -1] = 0
        # Not zero indexed
        else:
            targets_tensor -= torch.min(targets_tensor)
        return ts_collection, targets_tensor
