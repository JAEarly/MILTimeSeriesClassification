"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import copy
from typing import Callable, Dict, Tuple, Optional, Union, List, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import nn

from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.interpretability_metrics import calculate_aopcr, calculate_ndcg_at_n
from millet.util import custom_tqdm, cross_entropy_criterion


class MILLETModel:
    """Wrapper for models in the MILLET framework."""

    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module):
        super().__init__()
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.net = net.to(self.device)

    def fit(
        self,
        train_dataset: MILTSCDataset,
        n_epochs: int = 1500,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        criterion: Callable = cross_entropy_criterion,
    ) -> None:
        """
        Fit the MILLET model.

        :param train_dataset: MIL TSC dataset to fit to.
        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Learning rate of optimizer.
        :param weight_decay: Weight decay of optimizer.
        :param criterion: Loss function to train to minimise.
        :return:
        """
        # Setup
        batch_size = min(len(train_dataset) // 10, 16)
        batch_size = max(batch_size, 2)
        torch_train_dataloader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Early stopping setup
        best_net = None
        best_loss = np.Inf
        # Train over multiple epochs
        for _ in custom_tqdm(range(n_epochs), desc="Training model"):
            self.net.train()
            # Train model for an epoch
            for batch in torch_train_dataloader:
                bags = batch["bags"]
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad()
                model_out = self(bags)
                loss = criterion(model_out["bag_logits"], targets)
                loss.backward()
                optimizer.step()
            # Evaluate
            self.net.eval()
            epoch_train_results = self.evaluate(train_dataset, criterion)
            # Handle early stopping
            epoch_loss = epoch_train_results["loss"]
            if epoch_loss < best_loss:
                best_net = copy.deepcopy(self.net)
                best_loss = epoch_loss
                if epoch_loss == 0:
                    print("Training finished - early stopping (zero loss)")
                    break
        # Set net to best net found during training
        if best_net is not None:
            self.net = best_net
        else:
            raise ValueError("Best net not set during training - shouldn't be here so something has gone wrong!")

    def evaluate(
        self,
        dataset: MILTSCDataset,
        criterion: Callable = cross_entropy_criterion,
    ) -> Dict:
        # Iterate through data loader and gather preds and targets
        all_bag_logits_list = []
        all_targets_list = []
        # Don't need to worry about batch size being too big during evaluation (only training)
        dataloader = dataset.create_dataloader(batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                bags = batch["bags"]
                targets = batch["targets"]
                model_out = self(bags)
                bag_logits = model_out["bag_logits"]
                all_bag_logits_list.append(bag_logits.cpu())
                all_targets_list.append(targets)
        # Gather bag logits and targets into tensors
        all_bag_logits = torch.cat(all_bag_logits_list)
        all_targets = torch.cat(all_targets_list)
        # Get probas from logits
        all_pred_probas = torch.softmax(all_bag_logits, dim=1)
        # If in binary case, reduce probas to single prediction (not doing so breaks some of the evaluation metrics)
        if all_pred_probas.shape[1] == 2:
            all_pred_probas = all_pred_probas[:, 1]
        # Get the actual predicted classes
        _, all_pred_clzs = torch.max(all_bag_logits, dim=1)
        # Compute metrics
        loss = criterion(all_bag_logits, all_targets).item()
        acc = accuracy_score(all_targets.long(), all_pred_clzs)
        bal_acc = balanced_accuracy_score(all_targets.long(), all_pred_clzs)
        auroc = roc_auc_score(all_targets, all_pred_probas, multi_class="ovo", average="weighted")
        conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_clzs), dtype=torch.float)
        # Return results in dict
        all_results = {
            "loss": loss,
            "acc": acc,
            "bal_acc": bal_acc,
            "auroc": auroc,
            "conf_mat": conf_mat,
        }
        return all_results

    def evaluate_interpretability(
        self,
        dataset: MILTSCDataset,
    ) -> Tuple[float, Optional[float]]:
        all_aopcrs = []
        all_ndcgs = []
        # Don't need to worry about batch size being too big during evaluation (only training)
        dataloader = dataset.create_dataloader(batch_size=16)
        with torch.no_grad():
            for batch in custom_tqdm(dataloader, leave=False):
                bags = batch["bags"]
                batch_targets = batch["targets"]
                # Calculate AOPCR for batch
                batch_aopcr, _, _ = calculate_aopcr(self, bags, verbose=False)
                all_aopcrs.extend(batch_aopcr.tolist())
                # Calculate NDCG@n for batch if instance targets are present
                if "instance_targets" in batch:
                    batch_instance_targets = batch["instance_targets"]
                    all_instance_importance_scores = self.interpret(self(bags))
                    for bag_idx, bag in enumerate(bags):
                        target = batch_targets[bag_idx]
                        instance_targets = batch_instance_targets[bag_idx]
                        ndcg = calculate_ndcg_at_n(
                            all_instance_importance_scores[bag_idx, target],
                            instance_targets,
                        )
                        all_ndcgs.append(ndcg)
        avg_aopcr = np.mean(all_aopcrs)
        avg_ndcg = float(np.mean(all_ndcgs)) if len(all_ndcgs) > 0 else None
        return float(avg_aopcr), avg_ndcg

    def interpret(self, model_out: Dict) -> torch.Tensor:
        return model_out["interpretation"]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save_weights(self, path: str) -> None:
        # Save net from CPU
        #  Fixes issues with saving and loading from different devices
        print("Saving model to {:s}".format(path))
        torch.save(self.net.to("cpu").state_dict(), path)
        # Ensure net is back on original device
        self.net.to(self.device)

    def load_weights(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def forward(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Get the model output for a bag input.

        :param bag_input: Either a single bag or a batch of bags.
        :param bag_instance_positions: Single or batch of instance positions for each bag.
        :return: The model output.
        """
        # Reshape input depending on whether we have a single bag or a batch
        bags, is_unbatched_bag = self._reshape_bag_input(bag_input)
        # Actually pass the input through the model
        model_output = self._internal_forward(bags, bag_instance_positions)
        # If given input was not batched, un-batch the output
        if is_unbatched_bag:
            unbatched_model_output = {}
            for key, value in model_output.items():
                unbatched_model_output[key] = value[0]
            return unbatched_model_output
        return model_output

    def _reshape_bag_input(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], bool]:
        """
        Converts bag inputs to a consistent type.

        Input options:
        * 2D tensor (one unbatched bag - dims: n_instance x d_instance)
        * 3D tensor (batched collection of bags - dims: n_batch x n_instance x d_instance)
        * List of 2D tensors (batched collection of bags, but batch dim is the list not a tensor dim).

        The output is a batched collection on bags. For each input:
        * 2D tensor -> [2D Tensor]
        * 3D tensor -> 3D tensor
        * List [2D tensor] -> List [2D tensor]

        Note only the 2D tensor needs to be reshaped as it is unbatched.

        :param bag_input: See input options.
        :return: See outputs for each input above.
        """
        reshaped_input: Union[torch.Tensor, List[torch.Tensor]]
        # Bag input is a tensor
        if torch.is_tensor(bag_input):
            # Enforce that we're now using bag_input as a tensor
            bag_input = cast(torch.Tensor, bag_input)
            input_shape = bag_input.shape
            # In unbatched bag, expected two dims (n_instance, d_instance)
            if len(input_shape) == 2:
                # Just a single bag on its own, not in a batch, therefore place in a list
                reshaped_input = [bag_input]
                is_unbatched = True
            elif len(input_shape) == 3:
                # Already batched with three dims (n_batch, n_instance, d_instance)
                reshaped_input = bag_input
                is_unbatched = False
            else:
                raise NotImplementedError("Cannot process MIL model input with shape {:}".format(input_shape))
        # Model input is list
        elif isinstance(bag_input, list):
            # Assume input is a list of 2d tensors
            reshaped_input = bag_input
            is_unbatched = False
        # Invalid input type
        else:
            raise ValueError("Invalid model input type {:}".format(type(bag_input)))
        return reshaped_input, is_unbatched

    def _internal_forward(
        self, bags: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Actually call the network.

        :param bags: Batch of bags.
        :param bag_instance_positions: Batch of instance positions for each bag.
        :return: Dictionary of the network outputs.
        """
        # Stack list of bags to a single tensor
        #  Assumes all bags are the same size
        # Otherwise the input is assumed to be a 3D tensor (already stacked).
        if isinstance(bags, list):
            bags = torch.stack(bags)
        bags = bags.to(self.device)
        # Reshape to match (n_batch, d_instance, n_instance)
        #  d_instance is number of channels
        bags = bags.transpose(1, 2)
        # Pass through network
        return self.net(bags, bag_instance_positions)

    def __call__(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        return self.forward(bag_input, bag_instance_positions=bag_instance_positions)
