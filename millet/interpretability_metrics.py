"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from millet.model.millet_model import MILLETModel
from millet.util import custom_tqdm


def calculate_aopcr(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    verbose: bool = True,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    seed: int = 72,
    batch_interpretations: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the Area Over the Perturbation Curve to Random (AOPCR) for a batch of bags.
    Evaluation of MIL interpretability that does not require instance labels.

    :param model: MILLET model.
    :param bags: List of bags (each bag is a 2D tensor).
    :param verbose: Provide logging (progress bar).
    :param stop: Perturbation limit as proportion of bag size, e.g. 0.5 means perturb up to half the bag
    :param step: Proportion of bag to group together into single perturbation, e.g. 0.05 means 5% chunks.
    :param n_random: Number of random orderings to compare to.
    :param seed: Fixed seed for generating random orderings (to ensure consistent evaluation).
    :param batch_interpretations: Optional pre-computed interpretations. If not provided, model.interpret will be used.
    :return: AOPCR score, the perturbation curve, (average) random perturbation curve.
    """
    # Get model output for bags
    batch_model_out = model(bags)
    # Get bag logits for predicted classes, these are the "original" logits before perturbation
    batch_bag_logits = batch_model_out["bag_logits"].cpu()
    batch_pred_clzs = torch.argmax(batch_bag_logits, dim=1).tolist()
    batch_orig_logits = torch.zeros(len(bags))
    for i in range(len(bags)):
        clz = batch_pred_clzs[i]
        batch_orig_logits[i] = batch_bag_logits[i, clz]
    # Get interpretations for the predicted class for each bag (if not already provided)
    if batch_interpretations is None:
        all_batch_interpretations = model.interpret(batch_model_out)
        # Filter by predicted class
        batch_interpretations_list = []
        for batch_idx, clz in enumerate(batch_pred_clzs):
            batch_interpretations_list.append(all_batch_interpretations[batch_idx, clz, :])
        batch_interpretations = torch.stack(batch_interpretations_list)
    # Calculate AOPC for given interpretations
    aopc, pc = _calculate_aopc(
        model,
        bags,
        batch_orig_logits,
        batch_pred_clzs,
        batch_interpretations,
        stop,
        step,
        verbose,
    )
    # Calculate AOPC for random orderings
    r_aopc, r_pc = _calculate_random_aopc(
        model,
        bags,
        batch_orig_logits,
        batch_pred_clzs,
        stop,
        step,
        verbose,
        n_random,
        seed,
    )
    # Compute AOPCR and return
    aopcr = aopc - r_aopc
    return aopcr, pc, r_pc


def _calculate_aopc(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    batch_interpretation_scores: torch.Tensor,
    stop: float,
    step: float,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Area Over the Perturbation Curve

    :param model: MILLET model.
    :param bags: List of bags (each bag is a 2D tensor).
    :param batch_orig_logits: Original model prediction (logit) for the predicted class for each bag.
    :param clzs: Predicted class for each bag.
    :param batch_interpretation_scores: Model interpretation scores for each bag for the predicted classes.
    :param stop: Perturbation limit as proportion of bag size, e.g. 0.5 means perturb up to half the bag
    :param step: Proportion of bag to group together into single perturbation, e.g. 0.05 means 5% chunks.
    :param verbose: Provide logging (progress bar).
    :return: AOPC score for each bag in batch, Perturbation curve for each bag in batch.
    """
    # Compute perturbation steps
    n_bags = len(bags)
    n_instances = len(bags[0])
    steps = np.linspace(1, stop, num=math.ceil((1 - stop) / step + 1))
    n_steps = len(steps)
    # Set up perturbation curve and fill first row with original logits
    batch_pc = torch.zeros((n_bags, n_steps))
    batch_pc[:, 0] = batch_orig_logits
    # Get the instance orderings by most relevant (greatest score) first.
    batch_morf = torch.argsort(batch_interpretation_scores.cpu(), descending=True, stable=True)
    # Actually compute the perturbation curve
    for step_idx, step in custom_tqdm(
        enumerate(steps[1:]),
        total=n_steps - 1,
        desc="Computing perturbation curve",
        leave=False,
        disable=not verbose,
    ):
        # Work out how many instances to remove
        n_to_remove = int((1 - step) * n_instances)
        # Create perturbed bags and their respective positions
        perturbed_bags = []
        perturbed_bags_pos = []
        for i in range(n_bags):
            b, p = _create_perturbed_bag(bags[i], batch_morf[i], n_to_remove)
            perturbed_bags.append(b)
            perturbed_bags_pos.append(p)
        # Pass perturbed bags through the model to get the new logits
        with torch.no_grad():
            new_logits = model(perturbed_bags, torch.stack(perturbed_bags_pos))["bag_logits"]
        # Update output
        for batch_idx, clz in enumerate(clzs):
            batch_pc[batch_idx, step_idx + 1] = new_logits[batch_idx, clz].item()
    # Compute the AOPC for each bag in the batch
    batch_aopc = torch.zeros(n_bags)
    for k in range(1, n_steps):
        batch_aopc += batch_pc[:, 0] - batch_pc[:, k]
    batch_aopc /= n_steps
    # Adjust the perturbation curves to start at 0
    batch_pc -= batch_orig_logits.unsqueeze(1)
    return batch_aopc, batch_pc


def _calculate_random_aopc(
    model: "MILLETModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    stop: float,
    step: float,
    verbose: bool,
    n_repeats: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Area of the Perturbation Curve for random orderings.

    :param model: MILLET model.
    :param bags: List of bags (each bag is a 2D tensor).
    :param batch_orig_logits: Original model prediction (logit) for the predicted class for each bag.
    :param clzs: Predicted class for each bag.
    :param stop: Perturbation limit as proportion of bag size, e.g. 0.5 means perturb up to half the bag
    :param step: Proportion of bag to group together into single perturbation, e.g. 0.05 means 5% chunks.
    :param verbose: Provide logging (progress bar).
    :param n_repeats: Number of random orderings to compare to.
    :param seed: Fixed seed for generating random orderings (to ensure consistent evaluation).
    :return: (Average) Random AOPC score for each bag in batch, Random perturbation curve for each bag in batch.
    """
    n_bags = len(bags)
    n_instances = len(bags[0])
    torch.random.manual_seed(seed)
    random_aopcs = []
    random_pcs = []
    for r in range(n_repeats):
        # Create random interpretation scores (random ordering)
        random_interpretation_scores = torch.rand((n_bags, n_instances))
        # Compute AOPC and PC for this random ordering
        r_aopc, r_pc = _calculate_aopc(
            model,
            bags,
            batch_orig_logits,
            clzs,
            random_interpretation_scores,
            stop,
            step,
            verbose,
        )
        random_aopcs.append(r_aopc)
        random_pcs.append(r_pc)
    # Compute average over repeats
    aopc = torch.stack(random_aopcs).mean(dim=0)
    pc = torch.stack(random_pcs).mean(dim=0)
    return aopc, pc


def _create_perturbed_bag(
    bag: torch.Tensor, bag_morf: torch.Tensor, n_to_remove: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perturb a bag by removing the n most important instances.

    :param bag: Original bag to be perturbed.
    :param bag_morf: Ordering of bag instances by importance.
    :param n_to_remove: Number of instances to remove.
    :return: Perturbed bag, Original positions of each instance (i.e. position in the time series)
    """
    # Create mask of indices to remove
    mask = torch.ones(len(bag), dtype=torch.int)
    idxs_to_remove = bag_morf[:n_to_remove]
    mask[idxs_to_remove] = 0
    # Remove based on mask and create positions
    perturbed_bag = bag[mask == 1]
    perturbed_bag_pos = torch.arange(len(bag))[mask == 1]
    return perturbed_bag, perturbed_bag_pos


def calculate_ndcg_at_n(instance_importance_scores: torch.Tensor, instance_labels: torch.Tensor) -> float:
    """
    Calculate Normalised Discounted Cumulative Gain @ n (NDCG@n).
    Evaluation of MIL interpretability that requires instance labels.

    :param instance_importance_scores: Importance scores for each instance.
    :param instance_labels: Binary class labels for instances (1 = important, 0 = not important)
    :return: NDCG@n score
    """
    # Identify number of discriminatory instances
    n = int((instance_labels == 1).sum().item())
    # No targets so return nan
    if n == 0:
        print("Shouldn't happen!")
        raise ValueError("Trying to assess interpretability with no discriminatory instances")
    # Find idxs of the n largest interpretation scores
    top_n = torch.topk(instance_importance_scores.to("cpu"), n)[1]
    # Compute normalised discounted cumulative gain
    dcg = 0.0
    norm = 0.0
    for i, order_idx in enumerate(top_n):
        rel = instance_labels[order_idx].item()
        dcg += rel / math.log2(i + 2)
        norm += 1.0 / math.log2(i + 2)
    ndcg = dcg / norm
    return ndcg
