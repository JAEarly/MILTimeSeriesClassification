"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Base dataset class for MIL TSC.
"""

from abc import abstractmethod, ABC
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MILTSCDataset(Dataset, ABC):
    """Abstract class implementation for MIL TSC."""

    def __init__(self, dataset_name: str, split: str, apply_transform: bool = True):
        self.dataset_name = dataset_name
        self.split = split
        self.apply_transform = apply_transform
        data = self.get_time_series_collection_and_targets(self.split)
        self.ts_collection, self.targets = data
        self.n_clz = self._get_n_clz()

    @abstractmethod
    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load list of time series and their targets (classes).

        :param split: Dataset split, e.g. train or test
        :return: List of time series tensors, tensor of classes
        """
        pass

    def get_bags(self) -> List[torch.Tensor]:
        """
        Get a list of all (un-normalised) time series in this dataset.

        :return: List of all time series bags.
        """
        return [self.get_bag(idx) for idx in range(len(self))]

    def get_bag(self, idx: int) -> torch.Tensor:
        """
        Get a single (un-normalised) time series in this dataset.

        :param idx: Dataset idx
        :return: Time series bag as a tensor.
        """
        return self.ts_collection[idx]

    @staticmethod
    def apply_bag_transform(bag: torch.Tensor) -> torch.Tensor:
        """
        Apply z-normalisation.

        :param bag: Time series bag tensor to be transformed.
        :return: Transformed tensor.
        """
        std = torch.std(bag).item()
        # Account for time series that have the exact same value for very time step.
        if std == 0:
            std = 1
        norm_bag = (bag - torch.mean(bag)) / std
        return norm_bag

    def get_target(self, idx: int) -> int:
        """
        Get the target for a particular time series (by dataset index).

        :param idx: Dataset index.
        :return: Integer class label.
        """
        return int(self.targets[idx])

    def create_dataloader(self, shuffle: bool = False, batch_size: int = 16, num_workers: int = 0) -> DataLoader:
        """
        Create a batch dataloader.

        :param shuffle: Whether the dataloader should randomise the dataset order or not.
        :param batch_size: Size of each batch returned by the dataloader.
        :param num_workers: Number of works for parallel data loading.
        :return: PyTorch dataloader with custom collate function for MIL bags.
        """
        torch_dataloader = DataLoader(
            self,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=mil_collate_fn,
        )
        return torch_dataloader

    def get_n_idxs(self, n: int, clz: Optional[int] = None, shuffle: bool = False) -> torch.Tensor:
        """
        Get n indices from this dataset, either at random or matching a particular class.

        :param n: Number of indices to return.
        :param clz: Only get indices for a particular class.
        :param shuffle: If True, get random indices (matching clz if given). If False, get the first n (matching clz).
        :return: Tensor of dataset indices matching criteria.
        """
        if clz is not None:
            candidate_idxs = self.get_clz_idxs(clz)
        else:
            candidate_idxs = torch.arange(len(self))
        if shuffle:
            perm = torch.randperm(len(candidate_idxs))
            candidate_idxs = candidate_idxs[perm]
        selected_idxs = candidate_idxs[:n]
        return selected_idxs

    def get_clz_idxs(self, clz: int) -> torch.Tensor:
        """
        Get all indices of this dataset that match a particular class.

        :param clz: Class target
        :return: Tensor of dataset indices that belong to the particular class.
        """
        all_targets = torch.as_tensor(self.targets)
        return (all_targets == clz).nonzero(as_tuple=True)[0]

    def _get_n_clz(self) -> int:
        """
        Get the number of classes in this dataset.
        Also verifies the classes meet the labelling assumption.

        :return: Number of classes.
        """
        # Assumes classes are numbered 0,...,c
        n_clz = int(max(self.targets) + 1)
        # Double-check the classes in the dataset match this assumption
        unique_clzs = sorted(torch.unique(self.targets))
        assert unique_clzs == list(range(n_clz))
        return n_clz

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a dataset item. Applies time series normalisation (if self.apply_transform = True).

        :param idx: Dataset index.
        :return: Dictionary containing the bag and its target.
        """
        bag = self.get_bag(idx)
        if self.apply_transform:
            bag = self.apply_bag_transform(bag)
        return {
            "bag": bag,
            "target": self.targets[idx],
        }

    def __len__(self) -> int:
        return len(self.targets)


def mil_collate_fn(orig_batch: List[Dict]) -> Dict:
    """
    Custom batch collation function for MIL settings.

    :param orig_batch: List of dictionaries (one for each item in the batch).
    :return: Dictionary containing lists of bags, targets, and instance targets (if they exist).
    """
    bags = []
    targets = []
    instance_targets: Optional[List[torch.Tensor]] = [] if "instance_targets" in orig_batch[0] else None
    for batch_item in orig_batch:
        bags.append(batch_item["bag"])
        targets.append(batch_item["target"])
        if instance_targets is not None:
            instance_targets.append(batch_item["instance_targets"])
    new_batch = {
        "bags": bags,
        "targets": torch.as_tensor(targets),
        "instance_targets": instance_targets,
    }
    return new_batch
