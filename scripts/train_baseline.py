"""
train_baseline.py

Description: Code to train baseline classifier on C-MNIST
"""

# Standard libraries
import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

# Non-standard libraries
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from zero_init import ZerO_Init

# Custom libraries
from models.backbone import LeNet
from utils import data


################################################################################
#                                  Constants                                   #
################################################################################
# Format the current date and time as a string
CURR_DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Mapping string to optimizer classes
OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}

# Get device
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

################################################################################
#                               Helper Functions                               #
################################################################################
@torch.no_grad()
def compute_accuracy(model, dataloader):
    """
    Evaluate model with data from dataloader. Compute the accuracy

    Parameters
    ----------
    model : torch.nn.Module
        Model
    dataloader : torch.utils.data.DataLoader
        Data loader

    Returns
    -------
    float
        Accuracy
    """
    # Set model to evaluation
    model.eval()

    is_correct = []
    for X, y in dataloader:
        is_correct.append(torch.argmax(model(X), dim=1) == y)
    is_correct = torch.cat(is_correct)
    accuracy = torch.sum(is_correct)/len(is_correct)

    model.train()
    return accuracy.item()


def load_hparams(run_dir):
    """
    Load hyperparameters from the run directory

    Parameters
    ----------
    run_dir : str
        Path to the run directory

    Returns
    -------
    HParams
        Contains hyperparameters
    """
    # Check if JSON file exists
    hparams_path = os.path.join(run_dir, "hparams.json")
    if not os.path.exists(hparams_path):
        raise RuntimeError(f"Hyperparameters file doesn't exist! at `{hparams_path}`")

    with open(hparams_path, "r") as f:
        hparams_dict = json.load(f)
    hparams = HParams(**hparams_dict)

    return hparams


def load_baseline_model(run_dir):
    """
    Load baseline model from its run directory

    Parameters
    ----------
    run_dir : str
        Name of run directory

    Returns
    -------
    BaselineClassifier
        Trained model
    """
    # Check that paths exist
    hparams_path = os.path.join(run_dir, "hparams.json")
    if not os.path.exists(run_dir):
        raise RuntimeError(f"Run directory doesn't exist! at `{run_dir}`")

    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams_dict = json.load(f)
    hparams = HParams(**hparams_dict)

    # Load weights
    weights_path = os.path.join(run_dir, f"{hparams.train_data}_baseline_weights.pth")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Weights file doesn't exist! at `{weights_path}`")

    # Instantiate model
    model = BaselineClassifier(hparams)

    # Load model weights
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # Send to device
    model.to(DEVICE)

    # Set to eval state
    model.eval()

    return model


################################################################################
#                                   Classes                                    #
################################################################################
@dataclass
class HParams:
    seen_digits: tuple = (0, 3, 5, 6, 8, 9) # NOTE: Numbers with curves
    num_classes: int = 6
    num_epochs: int = 2
    lr: float = 0.005
    momentum: float = 0.9       # NOTE: Only applies if SGD is used
    batch_size: int = 64
    optimizer: str = "adamw"    # one of (sgd, adam, adamw)

    init: str = None            # "ZerO" or PyTorch default
    train_data: str = "ood"      # one of ("id" or "ood")


class BaselineClassifier(torch.nn.Module):
    """
    Baseline Classifier
    """

    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        print(self.hparams)

        # First model (vanilla model)
        self.model = LeNet(hparams.num_classes)

        # Set up optimizers
        opt_class = OPTIMIZERS[hparams.optimizer]
        opt_hparams = {"lr": hparams.lr}
        if hparams.optimizer == "sgd":
            opt_hparams["momentum"] = hparams.momentum
        self.first_opt = opt_class(self.model.parameters(), **opt_hparams)

        # Set up cross-entropy loss
        self.erm_loss = F.cross_entropy


    def train_model(self, train_dl, val_dl=None):
        """
        Train first model on in-distribution or out-of-distribution data.

        Parameters
        ----------
        train_dl : torch.utils.data.DataLoader
            ID or OOD training set (with seen classes)
        val_dl : torch.utils.data.DataLoader, optional
            ID or OOD validation set (with seen classes)

        Returns
        -------
        dict
            Contains final accuracies on in-distribution training, validation
            and test set, and out-of-distribution training set.
        """
        print("""
################################################################################
#                              (Vanilla) Model                                 #
################################################################################""")
        self.model.train()

        # Initialize weights with zero initialization
        if self.hparams.init == "ZerO":
            print("Initializing model with ZerO initialization!")
            self.model.apply(ZerO_Init)

        # Train model
        for epoch in range(1, self.hparams.num_epochs+1):
            for iter_idx, (x, y) in enumerate(train_dl):
                loss = self.erm_loss(self.model(x), y)

                # Perform backprop
                self.first_opt.zero_grad()
                loss.backward()
                self.first_opt.step()

            train_acc = compute_accuracy(self.model, train_dl)
            val_acc = 0.
            if val_dl is not None:
                val_acc = compute_accuracy(self.model, val_dl)
            print(f"Epoch {epoch} |\tTrain Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")

            # Log epoch metrics
            wandb.log({
                "model": "first",
                "epoch": epoch,
                "id_train_acc": train_acc,
                "id_val_acc": val_acc,
            })

        self.model.eval()


    def fit(self, id_train_data, id_val_data, ood_train_data, save_dir=None):
        """
        Train first model.

        ----------
        id_train_data : torch.utils.data.DataLoader
            In-distribution training set
        id_val_data : torch.utils.data.DataLoader
            In-distribution validation set
        ood_train_data : torch.utils.data.DataLoader
            Out-of-distribution training set
        save_dir : str, optional
            If provided, save hyperparameters and model weights to the directory
        """
        # Create data loaders
        assert self.hparams.train_data in ("id", "ood")
        dataloaders = []
        if self.hparams.train_data == "id":
            id_train_dl = DataLoader(id_train_data, batch_size=self.hparams.batch_size, shuffle=True)
            id_val_dl = DataLoader(id_val_data, batch_size=self.hparams.batch_size)
            dataloaders.extend([id_train_dl, id_val_dl])
        else:
            ood_train_dl = DataLoader(ood_train_data, batch_size=self.hparams.batch_size, shuffle=True)
            dataloaders.append(ood_train_dl)

        # Train first model
        self.train_model(*dataloaders)

        # Early exit, if no save directory provided
        if not save_dir:
            return

        # Save model
        weights_path = os.path.join(save_dir, f"{self.hparams.train_data}_baseline_weights.pth")
        torch.save(self.state_dict(), weights_path)


    @torch.no_grad()
    def extract_features(self, dataset):
        """
        Extract features from model.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset

        Returns
        -------
        torch.Tensor
            Extracted features for each sample in the dataset
        """
        self.model.eval()

        # Preare dataloader
        dataloader = DataLoader(dataset, self.hparams.batch_size, shuffle=False)

        # Extract features for all data in the adtaset
        accum_feats = []
        for X, _ in dataloader:
            accum_feats.append(self.model.extract_features(X).cpu())
        accum_feats = torch.cat(accum_feats).numpy()

        self.model.train()
        return accum_feats


################################################################################
#                                Main Functions                                #
################################################################################
def train():
    """
    Train vanilla model.
    """
    # Use default hyperparameters
    hparams = HParams()
    hparams_dict = vars(hparams)
    wandb.init(project="csc413", config=vars(hparams))

    # Load data
    dset_dicts = data.load_data(wandb.config.get("seen_digits", hparams.seen_digits))

    # Create directory for current run
    run_dir = os.path.join("checkpoints", "baseline", wandb.run.id)
    os.makedirs(run_dir)

    try:
        # Train vanilla classifier
        model = BaselineClassifier(hparams)
        model = model.to(DEVICE)
        model.fit(
            id_train_data=dset_dicts["id_train_seen"],
            id_val_data=dset_dicts["id_val_seen"],
            ood_train_data=dset_dicts["ood_train_seen"],
            save_dir=run_dir,
        )

        # Save hyperparameters in the run directory
        with open(os.path.join(run_dir, "hparams.json"), "w") as f:
            json.dump(hparams_dict, f, indent=4)

        return model

    except Exception as error_msg:
        # Remove directory
        shutil.rmtree(run_dir)

        raise error_msg


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
    run_dir = os.path.join("checkpoints", "baseline", run_dir)

    # Load model, if not provided
    model = load_baseline_model(run_dir)

    # Load hyperparameters
    hparams = load_hparams(run_dir)
    # Load datasets
    dset_dicts = data.load_data(hparams.seen_digits)

    # Extract features on OOD data
    ood_test_unseen_feats = model.extract_features(dset_dicts["ood_test_unseen"])

    # Store features
    np.savez(os.path.join(run_dir, "ood_test_unseen_feats.npz"), embeds=ood_test_unseen_feats)


if __name__ == "__main__":
    # 1. Set up argument parser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--action",
        choices=["train", "extract"],
        required=True,
    )
    PARSER.add_argument(
        "--run_dir",
        default=None,
        type=str,
    )

    # 2. Parse arguments
    ARGS = PARSER.parse_args()

    # 3. Call function
    if ARGS.action == "train":
        train()
    elif ARGS.action == "extract":
        assert ARGS.run_dir, "If extracting features, please provide `run_dir`!"
        extract(ARGS.run_dir)
