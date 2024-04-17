"""
eval_pop_cnn_filters.py

Description: Code to test the popular CNN filters.
"""
from typing import Any, Dict, Tuple, Iterator

# Standard libraries
import json
import os
from dataclasses import dataclass
from datetime import datetime

# Non-standard libraries
import click
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import transformers
import random

# Custom libraries
from models.backbone import LeNet
from .train_cdc import HParams, dataloader_to_sampler, compute_accuracy, DEVICE
from utils import data

from itertools import permutations


def create_filters(sampled_filters: np.ndarray) -> Iterator[Tuple[Tuple[torch.nn.Conv2d, torch.nn.Conv2d], str]]:
    num_filters1 = 3 * 6  # Filters needed for conv layer 1
    num_filters2 = 6 * 16 # Filters needed for conv layer 2

    for perm_ind, filters_perm_ind in enumerate(permutations(np.arange(len(sampled_filters)), num_filters1 + num_filters2)):
        weights1_ind = filters_perm_ind[:num_filters1]
        weights2_ind = filters_perm_ind[num_filters1:num_filters1+num_filters2]

        weights1 = np.take(sampled_filters, weights1_ind, axis=0)
        weights1 = weights1.reshape((3, 6, 3, 3)) # reshape to match LeNet
        weights2 = np.take(sampled_filters, weights2_ind, axis=0)
        weights2 = weights2.reshape((6, 16, 3, 3)) # reshape to match LeNet
        del weights1_ind
        del weights2_ind

        model_name = f"sampled_cnn_{perm_ind}"

        yield sample_filter(weights1, weights2, model_name)
    
    return None # No more filters to sample
    
def sample_filter(weights1: np.ndarray, weights2: np.ndarray, cnn_model_name: str) -> Tuple[Tuple[torch.nn.Conv2d, torch.nn.Conv2d], str]:
    # Assumes a kernel_size of 3x3
    if weights1.shape[-2:] != (3, 3):
        raise ValueError(f"Kernel size must be 3x3, not {weights1.shape[-2:]}")

    if weights2.shape[-2:] != (3, 3):
        raise ValueError(f"Kernel size must be 3x3, not {weights2.shape[-2:]}")

    # Sample channels need 3, 6 and 6, 16
    ## Sample first conv layer (3, 6)
    out_channels = torch.tensor(random.sample(range(weights1.shape[0]), 6))
    in_channels = torch.tensor(random.sample(range(weights1.shape[1]), 3))

    sampled_weight = weights1[out_channels][in_channels]
    filter1 = torch.nn.Conv2d(3, 6, kernel_size=3, padding=2) # match LeNet, but with 3x3
    filter1.weight = sampled_weight

    ## Sample second conv layer (6, 16)
    out_channels = torch.tensor(random.sample(range(weights2.shape[0]), 16))
    in_channels = torch.tensor(random.sample(range(weights2.shape[1]), 6))

    sampled_weight = weights2[out_channels][in_channels]
    filter2 = torch.nn.Conv2d(6, 16, kernel_size=3) # match LeNet but with 3x3
    filter2.weight = sampled_weight
    

    return filter1, filter2, cnn_model_name


class CNNPopFilter(torch.nn.Module):
    """
    A LeNet Model using Popular CNN Filters
    """

    def __init__(self, batch_size: int, num_classes: int, filter_name: str, filters: Tuple[torch.nn.Conv2d, torch.nn.Conv2d]):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.filter = filter_name
        filter1, filter2 = filters

        self.cnn = LeNet(num_classes) 

        # Validate Filters (Should match LeNet)
        if filter1.out_channels != self.cnn.conv1.out_channels \
            or filter1.in_channels != self.cnn.conv1.in_channels \
            or filter1.kernel_size != (3, 3): # differs from LeNet in kernel size only
            
            raise ValueError("Filter 1 needs to match the LeNet configuration.")
        
        if filter2.out_channels != self.cnn.conv2.out_channels \
            or filter2.in_channels != self.cnn.conv2.in_channels \
            or filter2.kernel_size != (3, 3): # differs from LeNet in kernel size only
            
            raise ValueError("Filter 1 needs to match the LeNet configuration.")
        
        # Set Filters
        self.cnn.conv1 = filter1

        self.cnn.conv2 = filter2

        # Modify fc1
        self.cnn.fc1 = torch.nn.Linear(
            filter2.out_channels * 3 * 3, # Uses 3x3 Kernel size
            self.cnn.fc1.out_features 
        )

    def forward(self, x):
        return self.cnn(x)

    def eval_filter_pair(self, id_val_dl, ood_val_dl, ood_test_dl, ood_test_un_dl):
        """
        Train first (vanilla) model.

        Parameters
        ----------
        id_val_data : torch.utils.data.DataLoader
            In-distribution validation set
        ood_val_data : torch.utils.data.DataLoader
            Out-of-distribution validation set
        ood_test_data : torch.utils.data.DataLoader
            Out-of-distribution test set
        ood_test_un_data : torch.utils.data.DataLoader
            Out-of-distribution Unseen test set
        
        Returns
        -------
        dict
            Contains final accuracies on in-distribution set, 
            and out-of-distribution validation and test sets.
        """
        print("""
################################################################################
#                                CNN Filter                                    #
################################################################################""")
        # Compute accuracy
        id_val_acc = compute_accuracy(self, id_val_dl)
        print(f"ID Val Acc (Seen): {id_val_acc:.2f}")

        ood_val_acc = compute_accuracy(self, ood_val_dl)
        print(f"OOD Val Acc (Seen): {ood_val_acc:.2f}")

        ood_test_acc = compute_accuracy(self, ood_test_dl)
        print(f"OOD Acc (Seen): {ood_test_acc:.2f}")

        ood_test_un_acc = compute_accuracy(self, ood_test_un_dl)
        print(f"OOD Acc (Unseen): {ood_test_un_acc:.2f}")


        # Log metrics
        wandb.log({
            "model": f"Pop CNN Filter ({self.filter})",
            "id_val_acc": id_val_acc,
            "ood_val_acc": ood_val_acc,
            "ood_test_acc": ood_test_acc,
            "ood_test_un_acc": ood_test_un_acc
        })


    def eval(self, id_val_data, ood_val_data, ood_test_data, ood_test_un_data):
        """
        Train first and second model.

        ----------
        id_val_data : torch.utils.data.TensorDataset
            In-distribution validation set
        ood_val_data : torch.utils.data.TensorDataset
            Out-of-distribution validation set
        ood_test_data : torch.utils.data.TensorDataset
            Out-of-distribution test set
        ood_test_un_data : torch.utils.data.TensorDataset
            Out-of-distribution unseen test set
        """
        # Create data loaders
        id_val_dl = DataLoader(id_val_data, batch_size=self.batch_size)
        ood_val_dl = DataLoader(ood_val_data, batch_size=self.batch_size)
        ood_test_dl = DataLoader(ood_test_data, batch_size=self.batch_size)
        ood_test_un_dl = DataLoader(ood_test_un_data, batch_size=self.batch_size)

        dataloaders = (id_val_dl, ood_val_dl, ood_test_dl, ood_test_un_dl)

        self.eval_filter_pair(*dataloaders)

@click.group()
def cli():
    pass


@cli.command()
@click.option("--filters-path", type=str, help="Path to sampled filters")
def train(filters_path: str):
    """
    Train disagreement model.
    """
    # Use default hyperparameters
    hparams = HParams()
    wandb.init(project="csc413", config=vars(hparams)) # just for digits

    # Load data
    dset_dicts = data.load_data(wandb.config.get("seen_digits", hparams.seen_digits))
    sampled_filters = np.load(filters_path) # Load sampled CNN filters
    
    try:
        # Evaluate each filter combination
        for filter_name, filters in create_filters(sampled_filters):
            model = CNNPopFilter(hparams.batch_size, hparams.num_classes, filter_name, filters)
            model = model.to(DEVICE)
            model.eval(
                id_train_data=dset_dicts["id_train_seen"],
                id_val_data=dset_dicts["id_val_seen"],
                id_test_data=dset_dicts["id_val_seen"],
                ood_train_data=dset_dicts["ood_train_seen"],
            )

    except Exception as error_msg:
        raise error_msg


if __name__ == "__main__":
    cli()
