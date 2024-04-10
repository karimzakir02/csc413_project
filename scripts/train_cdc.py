"""
train_cdc.py

Description: Code to train disagreement-based classifier on MNIST
"""

# Standard libraries
import os
from dataclasses import dataclass
from datetime import datetime

# Non-standard libraries
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


################################################################################
#                                   Classes                                    #
################################################################################
@dataclass
class HParams:
    num_classes: int = 5
    num_epochs: int = 2
    lr: float = 0.005
    momentum: float = 0.9       # NOTE: Only applies if SGD is used
    batch_size: int = 64
    optimizer: str = "adamw"    # one of (sgd, adam, adamw)

    disagreement_alpha: float = 0.5


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

        # Train model
        for epoch in range(1, self.hparams.num_epochs+1):
            for iter_idx, (x, y) in enumerate(id_train_dl):
                loss = self.erm_loss(self.first_model(x), y)

                # Perform backprop
                self.first_opt.zero_grad()
                loss.backward()
                self.first_opt.step()

                # print(f"Epoch {epoch} Iter {iter_idx} | \tTrain Loss: {loss.item():.4f}")

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

        # Create sampler from OOD data
        ood_sampler = dataloader_to_sampler(ood_train_dl)

        print("""
################################################################################
#                            (Disagreement) Model 2                            #
################################################################################""")

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
                    first_probs = torch.softmax(first_model_logits, dim=1)
                    entropy = -(first_probs * torch.log(first_probs + 1e-7)).sum(dim=1)
                    min_entropy, avg_entropy, max_entropy = min(entropy), entropy.mean(), max(entropy)
                    quarter_entropy = np.quantile(entropy, q=0.25)

                    # Filter OOD data for above average entropy
                    # TODO: Further investigate why selecting OOD data with low entropy is useful
                    ood_x = ood_x[entropy < quarter_entropy]

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

        # Log metrics
        wandb.log({
            "model": "second",
            "epoch": epoch,
            "id_train_acc": id_train_acc,
            "id_val_acc": id_val_acc,
            "id_test_acc": id_test_acc,
            "ood_train_acc": ood_train_acc,
        })


    def train(self, id_train_data, id_val_data, id_test_data, ood_train_data, save_dir=None):
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


if __name__ == "__main__":
    # Load data
    seen_digits = [0, 3, 5, 6, 8, 9]    # NOTE: Numbers with curves
    # seen_digits = [1, 2, 4, 7]          # NOTE: Numbers without curves
    # seen_digits = tuple(range(5))
    dset_dicts = data.load_data(seen_digits)

    # Create hyperparameters
    hparams = HParams(num_classes=len(seen_digits))

    ############################################################################
    #                             Train Model                                  #
    ############################################################################
    # Track configuration and metrics for current training run
    config = {"seen_digits": seen_digits}
    config.update(vars(hparams))
    wandb.init(
        project="csc413",
        config=config
    )

    # Create directory for current run
    run_dir = os.path.join("checkpoints", "cdc", CURR_DATETIME_STR)
    os.makedirs(run_dir)

    try:
        # Train disagreement-based classifier
        model = DisagreementClassifier(hparams)
        model = model.to(DEVICE)
        model.train(
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

    ############################################################################
    #                            OOD Evaluation                                #
    ############################################################################
    # Extract features on OOD data
    ood_test_unseen_feats = model.extract_disagreement_features(dset_dicts["ood_test_unseen"])

    # Store features
    np.savez(os.path.join(run_dir, "ood_test_unseen_feats.npz"), embeds=ood_test_unseen_feats)
