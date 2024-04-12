"""
train_cdc.py

Description: Code to train disagreement-based classifier on MNIST
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
def dataloader_to_sampler(dataloader):
    """
    Convert DataLoader to a sampler

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader

    Returns
    -------
    function
        Can be used to sample from the DataLoader
    """
    dl_iter = iter(dataloader)
    def sample():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            return next(dl_iter)
    return sample


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
    model.eval()

    is_correct = []
    for X, y in dataloader:
        is_correct.append(torch.argmax(model(X), dim=1) == y)
    is_correct = torch.cat(is_correct)
    accuracy = torch.sum(is_correct)/len(is_correct)

    model.train()
    return accuracy.item()

@torch.no_grad()
def compute_accuracy_ensemble(models, dataloader):
    """
    Evaluate ensemble of models with data from dataloader. Compute the accuracy
    based on an average prediction across models.

    Parameters
    ----------
    models : list of torch.nn.Module
        List of models
    dataloader : torch.utils.data.DataLoader
        Data loader

    Returns
    -------
    float
        Accuracy
    """
    # Set to evaluation
    for model in models:
        model.eval()

    is_correct = []
    for X, y in dataloader:
        # Average softmax probabilites for each model
        avg_probs = torch.stack([torch.softmax(model(X), dim=1) for model in models], dim=0).mean(dim=0)
        is_correct.append(torch.argmax(avg_probs, dim=1) == y)
    is_correct = torch.cat(is_correct)
    accuracy = torch.sum(is_correct)/len(is_correct)

    # Set back to training
    for model in models:
        model.train()
    return accuracy.item()


# NOTE: More numerically stable re-implementation of the disagreement loss
def disagreement_loss(first_model_logits, second_model_logits, epsilon=1e-6):
    """
    Compute the disagreement loss between the predictions of the first and
    second model.

    Parameters
    ----------
    first_model_logits : torch.Tensor
        Logits output by first model
    second_model_logits : torch.Tensor
        Logits output by second model (to disagree with on first)
    epsilon : float, optional
        Small numerical constant to prevent log(0)

    Returns
    -------
    torch.Tensor
        Disagreement loss, averaged across samples
    """
    # Compute class probabilities
    with torch.no_grad():
        first_log_probs = torch.log_softmax(first_model_logits, dim=1)

        # Get the predicted class's probability
        first_pos_log_prob, first_pos_idx = first_log_probs.max(dim=1)

        # Treat all other classes as the negative class
        first_neg_log_prob = torch.log1p(-torch.exp(first_pos_log_prob))

    # Compute class probabilities
    second_log_probs = torch.log_softmax(second_model_logits, dim=1)

    # Get probability of "positive" class (from first model's prediction)
    second_pos_log_prob = second_log_probs[:, first_pos_idx]

    # Treat all other classes as the negative class
    second_neg_log_prob = torch.log1p(-torch.exp(second_pos_log_prob) + epsilon)

    # Compute disagreement loss
    # NOTE: Disagree on the positive class
    disagreement_loss = -torch.stack([
        first_pos_log_prob + second_neg_log_prob,
        first_neg_log_prob + second_pos_log_prob,
    ]).logsumexp(dim=0)

    # Average across all samples
    disagreement_loss = disagreement_loss.mean()

    return disagreement_loss


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


def load_cdc_model(run_dir):
    """
    Load CDC model from its run directory

    Parameters
    ----------
    run_dir : str
        Name of run directory

    Returns
    -------
    DisagreementClassifier
        Trained model
    """
    # Check that paths exist
    weights_path = os.path.join(run_dir, "cdc_weights.pth")
    hparams_path = os.path.join(run_dir, "hparams.json")
    if not os.path.exists(run_dir):
        raise RuntimeError(f"Run directory doesn't exist! at `{run_dir}`")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Weights file doesn't exist! at `{weights_path}`")

    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams_dict = json.load(f)
    hparams = HParams(**hparams_dict)

    # Instantiate model
    model = DisagreementClassifier(hparams)

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

    disagreement_alpha: float = 0.5
    # Used to filter OOD data based on entropy in first model predictions
    entropy_q: float = 1.0    # 0.25 = bottom 25 entropy
    init: str = "ZerO"        # "ZerO" or PyTorch default


class DisagreementClassifier(torch.nn.Module):
    """
    Disagreement-Based Classifier
    """

    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams = hparams
        print(self.hparams)

        # First model (vanilla model)
        self.first_model = LeNet(hparams.num_classes)
        # Second model (disagrees with first model)
        self.second_model = LeNet(hparams.num_classes)

        # Set up optimizers
        opt_class = OPTIMIZERS[hparams.optimizer]
        opt_hparams = {"lr": hparams.lr}
        if hparams.optimizer == "sgd":
            opt_hparams["momentum"] = hparams.momentum
        self.first_opt = opt_class(self.first_model.parameters(), **opt_hparams)
        self.second_opt = opt_class(self.second_model.parameters(), **opt_hparams)

        # Set up cross-entropy loss
        self.erm_loss = F.cross_entropy
        self.disagreement_loss = disagreement_loss


    def train_first_model(self, id_train_dl, id_val_dl, id_test_dl,
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
#                              (Vanilla) Model 1                               #
################################################################################""")
        self.first_model.train()

        # Initialize weights with zero initialization
        if self.hparams.init == "ZerO":
            print("Initializing (first) CDC model with ZerO initialization!")
            self.first_model.apply(ZerO_Init)

        # Train model
        for epoch in range(1, self.hparams.num_epochs+1):
            for iter_idx, (x, y) in enumerate(id_train_dl):
                loss = self.erm_loss(self.first_model(x), y)

                # Perform backprop
                self.first_opt.zero_grad()
                loss.backward()
                self.first_opt.step()

                # TODO: Remove
                ood_train_acc = compute_accuracy(self.first_model, ood_train_dl)

                print(f"Epoch {epoch} Iter {iter_idx} | \tTrain Loss: {loss.item():.4f}, OOD Train Acc: {ood_train_acc:.2f}")

            id_train_acc = compute_accuracy(self.first_model, id_train_dl)
            id_val_acc = compute_accuracy(self.first_model, id_val_dl)
            ood_train_acc = compute_accuracy(self.first_model, ood_train_dl)
            print(f"Epoch {epoch} |\tID Train Acc: {id_train_acc:.2f}, OOD Train Acc: {ood_train_acc:.2f}, ID Val Acc: {id_val_acc:.2f}")

            # Log epoch metrics
            wandb.log({
                "model": "first",
                "epoch": epoch,
                "id_train_acc": id_train_acc,
                "id_val_acc": id_val_acc,
                "ood_train_acc": ood_train_acc,
            })

        # Compute test accuracy
        id_test_acc = compute_accuracy(self.first_model, id_test_dl)
        print(f"Epoch {epoch} |\tID Test Acc: {id_test_acc:.2f}")

        # Log metrics
        wandb.log({
            "model": "first",
            "epoch": epoch,
            "id_train_acc": id_train_acc,
            "id_val_acc": id_val_acc,
            "id_test_acc": id_test_acc,
            "ood_train_acc": ood_train_acc,
        })

        self.first_model.eval()


    def train_second_model(self, id_train_dl, id_val_dl, id_test_dl,
                           ood_train_dl):
        """
        Train second (disagreement) model.

        Parameters
        ----------
        id_train_dl : torch.utils.data.DataLoader
            In-distribution training set
        id_val_dl : torch.utils.data.DataLoader
            In-distribution validation set
        id_test_dl : torch.utils.data.DataLoader
            In-distribution test set
        ood_train_dl : torch.utils.data.DataLoader
            Out-of-distribution training set

        Returns
        -------
        dict
            Contains final accuracies on in-distribution training, validation
            and test set, and out-of-distribution training set.
        """
        models = [self.first_model, self.second_model]

        # Ensure first model isn't being trained
        self.first_model.eval()
        self.second_model.train()

        # Create sampler from OOD data
        ood_sampler = dataloader_to_sampler(ood_train_dl)

        print("""
################################################################################
#                            (Disagreement) Model 2                            #
################################################################################""")

        # Initialize weights with zero initialization
        if self.hparams.init == "ZerO":
            print("Initializing (second) CDC model with ZerO initialization!")
            self.second_model.apply(ZerO_Init)

        # Train model
        for epoch in range(1, self.hparams.num_epochs+1):
            for iter_idx, (x, y) in enumerate(id_train_dl):
                # Sample OOD data
                ood_x, _ = ood_sampler()

                # 1. Compute standard ERM loss
                erm_loss = self.erm_loss(self.second_model(x), y)

                # 2. Pass OOD data into first model and train second model to disagree
                with torch.no_grad():
                    first_model_logits = self.first_model(ood_x)
                    first_probs = torch.softmax(first_model_logits, dim=1).cpu()
                    entropy = -(first_probs * torch.log(first_probs + 1e-7)).sum(dim=1)
                    target_entropy = np.quantile(entropy, q=self.hparams.entropy_q)

                    # Filter OOD data for above average entropy
                    ood_x = ood_x[entropy < target_entropy]

                second_model_logits = self.second_model(ood_x)
                disagreement_loss = self.disagreement_loss(first_model_logits, second_model_logits)

                # 3. Combine losses
                loss = erm_loss + (self.hparams.disagreement_alpha * disagreement_loss)

                # print(f"Epoch {epoch} Iter {iter_idx} | \tTrain Loss: {loss.item():.4f}, ERM Loss: {erm_loss.item():.4f}, Disagreement Loss: {disagreement_loss.item():.4f}")
                # print(f"\t\tMin Entropy: {min_entropy}, Avg Entropy: {avg_entropy}, Max Entropy: {max_entropy}")

                # Perform backprop
                self.second_opt.zero_grad()
                loss.backward()
                self.second_opt.step()

            # id_train_acc = compute_accuracy_ensemble(models, id_train_dl)
            # id_val_acc = compute_accuracy_ensemble(models, id_val_dl)
            # ood_train_acc = compute_accuracy_ensemble(models, ood_train_dl)
            id_train_acc = compute_accuracy(self.second_model, id_train_dl)
            id_val_acc = compute_accuracy(self.second_model, id_val_dl)
            ood_train_acc = compute_accuracy(self.second_model, ood_train_dl) 
            print(f"Epoch {epoch} | \t Train Loss: {loss.item():.2f}, ID Train Acc: {id_train_acc:.2f}, OOD Train Acc: {ood_train_acc:.2f}, ID Val Acc: {id_val_acc:.2f}")

            # Log epoch metrics
            wandb.log({
                "model": "second",
                "epoch": epoch,
                "id_train_acc": id_train_acc,
                "id_val_acc": id_val_acc,
                "ood_train_acc": ood_train_acc,
            })

        # Compute test accuracy
        # id_test_acc = compute_accuracy_ensemble(models, id_test_dl)
        id_test_acc = compute_accuracy(self.second_model, id_test_dl)
        print(f"Epoch {epoch} | ID Test Acc: {id_test_acc:.2f}")

        # Revert first model to trainable
        self.first_model.train()
        self.second_model.eval()

        # Log metrics
        wandb.log({
            "model": "second",
            "epoch": epoch,
            "id_train_acc": id_train_acc,
            "id_val_acc": id_val_acc,
            "id_test_acc": id_test_acc,
            "ood_train_acc": ood_train_acc,
        })


    def fit(self, id_train_data, id_val_data, id_test_data, ood_train_data, save_dir=None):
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
        save_dir : str, optional
            If provided, save hyperparameters and model weights to the directory
        """
        # Create data loaders
        id_train_dl = DataLoader(id_train_data, batch_size=self.hparams.batch_size, shuffle=True)
        id_val_dl = DataLoader(id_val_data, batch_size=self.hparams.batch_size)
        id_test_dl = DataLoader(id_test_data, batch_size=self.hparams.batch_size)
        ood_train_dl = DataLoader(ood_train_data, batch_size=self.hparams.batch_size, shuffle=True)
        dataloaders = (id_train_dl, id_val_dl, id_test_dl, ood_train_dl)

        # Train first model
        self.train_first_model(*dataloaders)

        # Train second model
        self.train_second_model(*dataloaders)

        # Early exit, if no save directory provided
        if not save_dir:
            return

        # Save model
        weights_path = os.path.join(save_dir, "cdc_weights.pth")
        torch.save(self.state_dict(), weights_path)


    @torch.no_grad()
    def extract_disagreement_features(self, dataset):
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


################################################################################
#                                Main Functions                                #
################################################################################
def train():
    """
    Train disagreement model.
    """
    # CASE 1: Part of a parameter sweep
    if "WANDB_SWEEP_ID" in os.environ:
        wandb.init()
        # Get hyperparameters from config
        hparams_dict = {k: wandb.config.get(k, default) for k, default in vars(HParams()).items()}
        hparams = HParams(**hparams_dict)
    # CASE 2: Not part of a sweep
    else:
        # Use default hyperparameters
        hparams = HParams()
        hparams_dict = vars(hparams)
        wandb.init(project="csc413", config=vars(hparams))

    # Load data
    dset_dicts = data.load_data(wandb.config.get("seen_digits", hparams.seen_digits))

    # Create directory for current run
    run_dir = os.path.join("checkpoints", "cdc", wandb.run.id)
    os.makedirs(run_dir)

    try:
        # Train disagreement-based classifier
        model = DisagreementClassifier(hparams)
        model = model.to(DEVICE)
        model.fit(
            id_train_data=dset_dicts["id_train_seen"],
            id_val_data=dset_dicts["id_val_seen"],
            id_test_data=dset_dicts["id_val_seen"],
            ood_train_data=dset_dicts["ood_train_seen"],
            save_dir=run_dir,
        )

        # Save hyperparameters in the run directory
        with open(os.path.join(run_dir, "hparams.json"), "w") as f:
            json.dump(hparams_dict, f, indent=4)

    except Exception as error_msg:
        # Remove directory
        shutil.rmtree(run_dir)

        raise error_msg


def perform_sweep():
    """
    Perform hyperparameter sweep
    """
    # Load data
    seen_digits = [0, 3, 5, 6, 8, 9]    # NOTE: Numbers with curves
    # seen_digits = [1, 2, 4, 7]          # NOTE: Numbers without curves
    # seen_digits = tuple(range(5))

    # Parameter sweep configuration
    sweep_configuration = {
        "name": "cdc_sweep",
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "ood_train_acc"},
        "parameters": {
            "seen_digits": {"value": seen_digits},
            "num_classes": {"value": len(seen_digits)},
            "num_epochs": {"values": [5]},
            "lr": {"min": 0.0001, "max": 0.1},
            "momentum": {"min": 0.9, "max": 1.-1e-5},
            "batch_size": {"values": [128, 256]},
            "optimizer": {"values": ["adamw", "sgd"]},

            "disagreement_alpha": {"min": 0.0001, "max": 1.},
            "entropy_q": {"values": [0.25, 0.5, 0.75, 1.]},
            "init": {"value": "ZerO"}
        },
    }

    # Configure sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="cdc_sweep_w_entropy_and_zero_init")

    # Perform parameter sweep
    wandb.agent(sweep_id, function=train, count=20)


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
    # 1. Set up argument parser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--action",
        choices=["train", "perform_sweep", "extract"],
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
    elif ARGS.action == "perform_sweep":
        perform_sweep()
    elif ARGS.action == "extract":
        assert ARGS.run_dir, "If extracting features, please provide `run_dir`!"
        extract(ARGS.run_dir)
