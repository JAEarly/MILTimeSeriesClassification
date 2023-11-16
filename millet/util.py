"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import sys
from functools import partial

import torch
from torch import nn
from tqdm import tqdm

# tqdm wrapper to write to stdout rather than stderr
custom_tqdm = partial(tqdm, file=sys.stdout)


def get_gpu_device_for_os() -> torch.device:
    """
    Get GPU device for different operating systems. Currently setup for MacOS and Linux.

    :return: Torch GPU device.
    """
    if sys.platform == "darwin":
        return torch.device("mps")
    elif sys.platform == "linux":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Cuda GPU device not found on Linux.")
    raise NotImplementedError("GPU support not configured for platform {:s}".format(sys.platform))


def cross_entropy_criterion(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Cross entropy criterion wrapper that ensures targets are integer class indices.

    :param predictions: Tensor of logits.
    :param targets: Tensor of targets.
    :return:
    """
    loss: float = nn.CrossEntropyLoss()(predictions, targets.long())
    return loss
