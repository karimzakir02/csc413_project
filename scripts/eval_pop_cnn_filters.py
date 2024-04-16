"""
eval_pop_cnn_filters.py

Description: Code to test the popular CNN filters.
"""
from typing import Any, Dict, Tuple

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



def create_filters() -> Dict[str, Tuple[torch.nn.Conv2d, torch.nn.Conv2d]]:
    filters = {}

    #timm_resnet50 = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True)
    #sample_filter(timm_resnet50.conv1, "timm_resnet50")

    #google_vit_base = transformers.AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    #sample_filter(google_vit_base.vit.embeddings.patch_embeddings.projection, "google_vit_base")

    return filters
    
def sample_filter(conv_layer: torch.nn.Conv2d, cnn_model_name: str, filters) -> None:
    # Assumes a kernel_size of 5
    if conv_layer.kernel_size != (5, 5):
        raise ValueError("Kernel size must be 5x5")

    weights = conv_layer.weight
    bias = conv_layer.bias
    # Sample channels need 3, 6 and 6, 16
    ## Sample first conv layer (3, 6)
    out_channels = torch.tensor(random.sample(range(weights.size[0]), 6))
    in_channels = torch.tensor(random.sample(range(weights.size[1]), 3))

    sampled_weight = weights[out_channels][in_channels]
    sampled_bias = bias[out_channels]
    new_filter = torch.nn.Conv2d(3, 6, kernel_size=5, padding=2)
    new_filter.weight = sampled_weight
    new_filter.bias = sampled_bias
    
    filters[cnn_model_name][0] = new_filter

    ## Sample second conv layer (6, 16)
    out_channels = torch.tensor(random.sample(range(weights.size[0]), 16))
    in_channels = torch.tensor(random.sample(range(weights.size[1]), 6))

    sampled_weight = weights[out_channels][in_channels]
    sampled_bias = bias[out_channels]
    new_filter = torch.nn.Conv2d(6, 16, kernel_size=5)
    new_filter.weight = sampled_weight
    new_filter.bias = sampled_bias
    
    filters[cnn_model_name][1] = new_filter



class CNNPopFilter(torch.nn.Module):
    """
    A LeNet Model using Popular CNN Filters
    """

    def __init__(self, num_classes: int, filter_names: Tuple[str], filters: Dict[str, Tuple[torch.nn.Conv2d, torch.nn.Conv2d]]):
        super().__init__()
        self.filter1, self.filter2 = filter_names
        filter1, filter2 = filters[self.filter1][0], filters[self.filter2][1]

        self.cnn = LeNet(num_classes) 

        # Validate Filters (Should match LeNet)
        if filter1.out_channels != self.cnn.conv1.out_channels \
            or filter1.in_channels != self.cnn.conv1.in_channels \
            or filter1.kernel_size != self.cnn.conv1.kernel_size:
            
            raise ValueError("Filter 1 needs to match the LeNet configuration.")
        
        if filter2.out_channels != self.cnn.conv2.out_channels \
            or filter2.in_channels != self.cnn.conv2.in_channels \
            or filter2.kernel_size != self.cnn.conv2.kernel_size:
            
            raise ValueError("Filter 1 needs to match the LeNet configuration.")
        
        # Set Filters
        self.cnn.conv1 = filter1

        self.cnn.conv2 = filter2

    def forward(self, x):
        return self.cnn(x)

    def eval_filter_pair(self, id_val_dl, id_test_dl,
                          ood_train_dl):
        """
        Train first (vanilla) model.

        Parameters
        ----------
        id_train_dl : torch.utils.data.DataLoader
            In-distribution training set (with seen classes)
        id_val_dl : torch.utils.data.DataLoader
            In-distribution validation set (with seen classes)
        id_test_dl : torch.utils.data.DataLoader
            In-distribution test set (with seen classes)
        ood_train_dl : torch.utils.data.DataLoader
            Out-of-distribution training set

        Returns
        -------
        dict
            Contains final accuracies on in-distribution training, validation
            and test set, and out-of-distribution training set.
        """
        print("""
################################################################################
#                                CNN Filter                                    #
################################################################################""")
        # Compute val accuracy
        id_val_acc = compute_accuracy(self, id_val_dl)
        print(f"ID Val Acc: {id_val_acc:.2f}")

        # Compute test accuracy
        id_test_acc = compute_accuracy(self, id_test_dl)
        print(f"ID Test Acc: {id_test_acc:.2f}")

        # Compute ood accuracy
        ood_train_acc = compute_accuracy(self, ood_train_dl)
        print(f"OOD Acc: {id_test_acc:.2f}")

        # Log metrics
        wandb.log({
            "model": f"Pop CNN Filter ({self.filters[0]},{self.filters[1]})",
            "id_test_acc": id_test_acc,
            "id_val_acc": id_val_acc,
            "ood_train_acc": ood_train_acc,
        })


    def eval(self, id_val_data, id_test_data, ood_train_data):
        """
        Train first and second model.

        ----------
        id_train_data : torch.utils.data.DataLoader
            In-distribution training set
        id_val_data : torch.utils.data.DataLoader
            In-distribution validation set
        id_test_data : torch.utils.data.DataLoader
            In-distribution test set
        ood_train_data : torch.utils.data.DataLoader
            Out-of-distribution training set
        """
        # Create data loaders
        id_val_dl = DataLoader(id_val_data, batch_size=self.hparams.batch_size)
        id_test_dl = DataLoader(id_test_data, batch_size=self.hparams.batch_size)
        ood_train_dl = DataLoader(ood_train_data, batch_size=self.hparams.batch_size, shuffle=True)
        dataloaders = (id_val_dl, id_test_dl, ood_train_dl)

        self.eval_filter_pair(*dataloaders)



    @torch.no_grad()
    def extract_cnn_filter_features(self, dataset):
        """
        Extract features using second (disagreement) model.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset

        Returns
        -------
        torch.Tensor
            Extracted features for each sample in the dataset
        """
        self.second_model.eval()

        # Preare dataloader
        dataloader = DataLoader(dataset, self.hparams.batch_size, shuffle=False)

        # Extract features for all data in the adtaset
        accum_feats = []
        for X, _ in dataloader:
            accum_feats.append(self.second_model.extract_features(X))
        accum_feats = torch.cat(accum_feats).numpy()

        self.second_model.train()
        return accum_feats


@click.group()
def cli():
    pass


@cli.command()
def train():
    """
    Train disagreement model.
    """
    # Use default hyperparameters
    hparams = HParams()
    wandb.init(project="csc413", config=vars(hparams)) # just for digits

    # Load data
    dset_dicts = data.load_data(wandb.config.get("seen_digits", hparams.seen_digits))
    del hparams

    # Create directory for current run
    run_dir = os.path.join("checkpoints", "cdc", wandb.run.id)
    os.makedirs(run_dir)

    try:
        # Train disagreement-based classifier
        model = CNNPopFilters(hparams)
        model = model.to(DEVICE)
        model.fit(
            id_train_data=dset_dicts["id_train_seen"],
            id_val_data=dset_dicts["id_val_seen"],
            id_test_data=dset_dicts["id_val_seen"],
            ood_train_data=dset_dicts["ood_train_seen"],
            save_dir=run_dir,
        )

    except Exception as error_msg:
        # Remove directory
        os.rmdir(run_dir)

        raise error_msg



@cli.command()
@click.option("--run_dir", type=str, help="Name of CDC run sub-directory")
def extract(run_dir):
    """
    Extract features from OOD test set.

    Parameters
    ----------
    run_dir : str
        Name of run directory
    """
    print(f"Extracting features at run directory `{run_dir}`...")

    # Prepend checkpoint directory
    run_dir = os.path.join("checkpoints", "cdc", run_dir)

    # Load model
    model = load_cdc_model(run_dir)

    # Load hyperparameters
    hparams = load_hparams(run_dir)
    # Load datasets
    dset_dicts = data.load_data(hparams.seen_digits)

    # Extract features on OOD data
    ood_test_unseen_feats = model.extract_disagreement_features(dset_dicts["ood_test_unseen"])

    # Store features
    np.savez(os.path.join(run_dir, "ood_test_unseen_feats.npz"), embeds=ood_test_unseen_feats)


if __name__ == "__main__":
    cli()
