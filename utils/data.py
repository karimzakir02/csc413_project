"""
data.py

Description: Used to create colored MNIST datasets.
"""

# Standard libraries
import math
import random
import os
import warnings

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


################################################################################
#                                  Constants                                   #
################################################################################
warnings.simplefilter("ignore")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seed for data generation
SEED = 20240413

# Assets directory
DIR_ASSETS = "assets"

################################################################################
#                               Helper Functions                               #
################################################################################
def set_random_seed(seed):
    """
    Set random seed

    Parameters
    ----------
    seed : int
        Random seed
    """
    # Set the temporary seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def color_digits_fixed(X, Y):
    """
    Color digits with fixed coloring (based on the image label)

    Parameters
    ----------
    X : torch.Tensor
        MNIST images
    Y : torch.Tensor
        Image labels (0 to 9)

    Returns
    -------
    torch.Tensor
        Colored MNIST images
    """
    res = []
    for x, y in zip(X, Y):
        x = x / 255
        mask = x.view(28,28) > 0.1
        img = x.repeat(3, 1, 1)
        if y == 0:
            img[0][mask] *= 0.5
        elif y == 1:
            img[1][mask] *= 0.5
        elif y == 2:
            img[2][mask] *= 0.5
        elif y == 3:
            img[0][mask] *= 0.2
            img[1][mask] *= 0.2
        elif y == 4:
            img[0][mask] *= 0.1
            img[2][mask] *= 0.1
        elif y == 5:
            img[1][mask] *= 0.6
            img[2][mask] *= 0.
        elif y == 6:
            img[1][mask] *= 0.3
            img[2][mask] *= 0.2
        elif y == 7:
            img[0][mask] *= 0.
            img[2][mask] *= 0.6
        elif y == 8:
            img[0][mask] *= 0.5
            img[1][mask] *= 0.2
        else:
            pass
        res.append(img.clip(0,1))
    res = torch.stack(res)
    return res


def color_digits_randomly(X, Y):
    """
    Color digits randomly (ignoring label).

    Parameters
    ----------
    X : torch.Tensor
        MNIST images
    Y : torch.Tensor
        IGNORED Image labels (0 to 9)

    Returns
    -------
    torch.Tensor
        MNIST images colored randomly
    """
    res = []
    for x, y in zip(X, Y):
        x = x / 255
        mask = x.view(28,28) > 0.1
        img = x.repeat(3, 1, 1)
        color_idx = random.randint(0, 9)
        if color_idx == 9:
            img[0][mask] *= 0.5
        elif color_idx == 8:
            img[1][mask] *= 0.5
        elif color_idx == 7:
            img[2][mask] *= 0.5
        elif color_idx == 6:
            img[0][mask] *= 0.2
            img[1][mask] *= 0.2
        elif color_idx == 5:
            img[0][mask] *= 0.1
            img[2][mask] *= 0.1
        elif color_idx == 4:
            img[1][mask] *= 0.6
            img[2][mask] *= 0.
        elif color_idx == 3:
            img[1][mask] *= 0.3
            img[2][mask] *= 0.2
        elif color_idx == 2:
            img[0][mask] *= 0.
            img[2][mask] *= 0.6
        elif color_idx == 1:
            img[0][mask] *= 0.5
            img[1][mask] *= 0.2
        else:
            pass
        res.append(img.clip(0,1))
    res = torch.stack(res)

    return res


def split_seen_and_unseen_digits(X, Y, seen_digits=tuple(range(5))):
    """
    Split data into seen and unseen digits.

    Parameters
    ----------
    X : torch.Tensor
        MNIST images
    Y : torch.Tensor
        Image labels (0 to 9)
    seen_digits : tuple, optional
        Digits seen during training and used by OOD data, by default 0 to 4.
        Other digits are considered "unseen" and used in testing.
    """
    # Filter training set for only seen images
    mask_seen_digits = torch.isin(Y, torch.Tensor(seen_digits))
    X_seen, X_unseen = X[mask_seen_digits], X[~mask_seen_digits]
    Y_seen, Y_unseen = Y[mask_seen_digits], Y[~mask_seen_digits]

    # Map old labels to new labels
    # NOTE: To account for non-contiguous labels
    relabel_seen = {digit: idx for idx, digit in enumerate(seen_digits)}
    unseen_digits = [digit for digit in range(10) if digit not in seen_digits]
    relabel_unseen = {digit: idx for idx, digit in enumerate(unseen_digits)}
    for old_digit, new_digit in relabel_seen.items():
        Y_seen[Y_seen == old_digit] = new_digit
    for old_digit, new_digit in relabel_unseen.items():
        Y_unseen[Y_unseen == old_digit] = new_digit

    assert len(X_seen) == len(Y_seen)
    assert len(X_unseen) == len(Y_unseen)

    return (X_seen, Y_seen), (X_unseen, Y_unseen)


def shuffle_data(X, Y):
    """
    Shuffle data.

    Parameters
    ----------
    X : torch.Tensor
        MNIST image
    Y : torch.Tensor
        Digit labels

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        Shuffled data
    """
    # Shuffle training and test data
    rand_perm = torch.randperm(len(X))
    X = X[rand_perm]
    Y = Y[rand_perm]

    return X, Y


def split_train_val(X, Y, val_size=0.25, shuffle=True):
    """
    Split data into training and validation splits.

    Parameters
    ----------
    X : torch.Tensor
        Images
    Y : torch.Tensor
        Labels
    val_size : float, optional
        Proportion to leave for validation, by default 0.25
    shuffle : bool, optional
        If True, shuffle data before splitting, by default True

    Returns
    -------
    tuple of ((imgs, labels), (imgs, labels))
        First sub-tuple is for training, and second sub-tuple is for validation.
    """
    # If specified, shuffle data first
    if shuffle:
        X, Y = shuffle_data(X, Y)

    # Split into training and validation sets
    val_idx = int(len(X) * val_size)
    X_val, Y_val = X[:val_idx], Y[:val_idx]
    X_train, Y_train = X[val_idx:], Y[val_idx:]

    return (X_train, Y_train), (X_val, Y_val)


def send_to_device(X, Y, device=DEVICE):
    """
    Prepare data to send to device.
    """
    return (X.to(device).float(), Y.to(device))


def get_class_stats(Y, digits):
    """
    Get stats on classes.

    Parameters
    ----------
    Y : torch.Tensor
        Labels
    digits : list
        Digits

    Returns
    -------
    dict
        Mapping of digit to counts.
    """
    # Get label counts
    uniq_vals, counts = torch.unique(Y, return_counts=True)

    # Decode label to digit
    label_to_digit = {idx: digit for idx, digit in enumerate(digits)}

    # Get count for each digit
    class_stats = {}
    for i, val in enumerate(uniq_vals):
        digit = label_to_digit[val.item()]
        class_stats[digit] = counts[i].item()

    return class_stats


def load_data(seen_digits=tuple(range(5)), device=DEVICE, seed=SEED, plot=False):
    """
    Load MNIST data. If seed specified, sets random seed.

    Note
    ----
    Unseen digits are kept for exploratory analysis

    Parameters
    ----------
    seen_digits : tuple, optional
        Digits seen during training and used by OOD data, by default 0 to 4.
        Other digits are considered "unseen" and used in testing.
    device : torch.device, optional
        Device to send data to, by default DEVICE
    seed : int, optional
        Random seed
    plot : bool, optional
        If True, make gridplot for each dataset.

    Returns
    -------
    dict of (str to torch.Dataset)
        "id_train_seen"  : In-Distribution Training Set   (with Seen Digits)
        "id_val_seen"    : In-Distribution Validation Set (with Seen Digits)
        "ood_train_seen" : Out-Of-Distribution Train Set  (with Seen Digits)
        "ood_val_seen"   : Out-Of-Distribution Val. Set   (with Seen Digits)
        "ood_test_seen"  : Out-Of-Distribution Test Set   (with Seen Digits)
        "ood_test_unseen": Out-Of-Distribution Test Set   (with Unseen Digits)
    """
    # Set random seed, if specified
    if seed is not None:
        set_random_seed(seed)

    # Download data
    train_set = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True)
    test_set = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True)

    # Split training and test set into seen and unseen digits
    (X_train_seen, Y_train_seen), (X_train_unseen, Y_train_unseen) = split_seen_and_unseen_digits(
        train_set.train_data, train_set.train_labels, seen_digits=seen_digits
    )
    (X_test_seen, Y_test_seen), (X_test_unseen, Y_test_unseen) = split_seen_and_unseen_digits(
        test_set.test_data, test_set.test_labels, seen_digits=seen_digits
    )

    # Change image shapes
    X_train_seen = X_train_seen.view(-1, 1, 28, 28)
    X_train_unseen = X_train_unseen.view(-1, 1, 28, 28)
    X_test_seen = X_test_seen.view(-1, 1, 28, 28)
    X_test_unseen = X_test_unseen.view(-1, 1, 28, 28)

    # NOTE: Combine unseen digits data from training and test set to form big
    #       OOD test set (unseen digits)
    X_ood_test_unseen = torch.concat([X_train_unseen, X_test_unseen])
    Y_ood_test_unseen = torch.concat([Y_train_unseen, Y_test_unseen])

    # Set aside 20% of the training data for OOD test set (covariate shift)
    (X_train_seen, Y_train_seen), (X_ood_test_seen, Y_ood_test_seen) = split_train_val(
        X_train_seen, Y_train_seen,
        val_size=0.2,
        shuffle=True,
    )
    # NOTE: OOD test set (covariate shift) = MNIST original test set + 20% of training data
    X_ood_test_seen = torch.concat([X_test_seen, X_ood_test_seen])
    Y_ood_test_seen = torch.concat([Y_test_seen, Y_ood_test_seen])

    # Split (half) training & testing data into ID and OOD (covariate shift)
    (X_id_train_seen, Y_id_train_seen), (X_ood_train_seen, Y_ood_train_seen) = split_train_val(
        X_train_seen, Y_train_seen,
        val_size=0.5,
        shuffle=True,
    )

    # Split training data into training and validation sets
    # NOTE: 40% (32-8) is used for ID train-val set (fixed colors)
    (X_id_train_seen, Y_id_train_seen), (X_id_val_seen, Y_id_val_seen) = split_train_val(
        X_id_train_seen, Y_id_train_seen,
        val_size=0.2,
        shuffle=True,
    )
    # NOTE: 40% (32-8) is used for covariate OOD train-val set (random colors)
    (X_ood_train_seen, Y_ood_train_seen), (X_ood_val_seen, Y_ood_val_seen) = split_train_val(
        X_ood_train_seen, Y_ood_train_seen,
        val_size=0.2,
        shuffle=True,
    )

    # Color data
    # NOTE: ID train/val/test set   (fixed colors)
    X_id_train_seen = color_digits_fixed(X_id_train_seen, Y_id_train_seen)
    X_id_val_seen = color_digits_fixed(X_id_val_seen, Y_id_val_seen)

    # NOTE: OOD train/val/test set  (seen digits with random colors)
    X_ood_train_seen = color_digits_randomly(X_ood_train_seen, Y_ood_train_seen)
    X_ood_val_seen = color_digits_randomly(X_ood_val_seen, Y_ood_val_seen)
    X_ood_test_seen = color_digits_randomly(X_ood_test_seen, Y_ood_test_seen)

    # NOTE: OOD test set            (unseen digits with random colors)
    X_ood_test_unseen = color_digits_randomly(X_ood_test_unseen, Y_ood_test_unseen)

    # Log dataset details
    unseen_digits = [digit for digit in range(10) if digit not in seen_digits]
    print(f"Size of ID Training Set     (Seen Digits): {len(Y_id_train_seen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_id_train_seen, seen_digits)}")
    print(f"Size of ID Validation Set   (Seen Digits): {len(Y_id_val_seen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_id_val_seen, seen_digits)}")
    print("")
    print(f"Size of OOD Training Set    (Seen Digits): {len(Y_ood_train_seen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_ood_train_seen, seen_digits)}")
    print(f"Size of OOD Validation Set  (Seen Digits): {len(Y_ood_val_seen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_ood_val_seen, seen_digits)}")
    print(f"Size of OOD Test Set        (Seen Digits): {len(Y_ood_test_seen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_ood_test_seen, seen_digits)}")
    print("")
    print(f"Size of OOD Test Set        (Unseen Digits): {len(Y_ood_test_unseen)}")
    print(f"\tClass Sizes: {get_class_stats(Y_ood_test_unseen, unseen_digits)}")

    # If specified, plot each type of image distribution
    if plot:
        plot_datasets = [
            ("ID Data", "id_seen", X_id_train_seen),
            ("OOD Data (Covariate Shift)", "ood_seen", X_ood_train_seen),
            ("OOD Data (Covariate Shift + Label Shift)", "ood_unseen", X_ood_test_unseen)
        ]
        for fig_title, save_fname, X in plot_datasets:
            gridplot_images(
                X[:64], save_dir=DIR_ASSETS, save_fname=save_fname,
                # fig_title=fig_title,
            )

    # Send to device
    X_id_train_seen, Y_id_train_seen = send_to_device(X_id_train_seen, Y_id_train_seen, device)
    X_id_val_seen, Y_id_val_seen = send_to_device(X_id_val_seen, Y_id_val_seen, device)
    X_ood_train_seen, Y_ood_train_seen = send_to_device(X_ood_train_seen, Y_ood_train_seen, device)
    X_ood_val_seen, Y_ood_val_seen = send_to_device(X_ood_val_seen, Y_ood_val_seen, device)
    X_ood_test_seen, Y_ood_test_seen = send_to_device(X_ood_test_seen, Y_ood_test_seen, device)
    X_ood_test_unseen, Y_ood_test_unseen = send_to_device(X_ood_test_unseen, Y_ood_test_unseen, device)

    # Convert into Dataset objects
    id_train_seen_dataset = torch.utils.data.TensorDataset(X_id_train_seen, Y_id_train_seen)
    id_val_seen_dataset = torch.utils.data.TensorDataset(X_id_val_seen, Y_id_val_seen)
    ood_train_seen_dataset = torch.utils.data.TensorDataset(X_ood_train_seen, Y_ood_train_seen)
    ood_val_seen_dataset = torch.utils.data.TensorDataset(X_ood_val_seen, Y_ood_val_seen)
    ood_test_seen_dataset = torch.utils.data.TensorDataset(X_ood_test_seen, Y_ood_test_seen)
    ood_test_unseen_dataset = torch.utils.data.TensorDataset(X_ood_test_unseen, Y_ood_test_unseen)

    datasets = {
        "id_train_seen": id_train_seen_dataset,
        "id_val_seen": id_val_seen_dataset,
        "ood_train_seen": ood_train_seen_dataset,
        "ood_val_seen": ood_val_seen_dataset,
        "ood_test_seen": ood_test_seen_dataset,
        "ood_test_unseen": ood_test_unseen_dataset,
    }

    return datasets


def gridplot_images(X,
                    fig_title=None,
                    save_dir=None,
                    save_fname="classes_gridplot.png"):
    """
    Plot grid of images
    """
    num_imgs = len(X)
    nrows = math.floor(math.sqrt(num_imgs))

    # Take subset of images to make square
    if len(X) > nrows ** 2:
        X = X[:nrows ** 2]

    # Create grid of images
    grid_img = torchvision.utils.make_grid(X, nrow=nrows, padding=0)

    # Create a new figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(grid_img.permute(1, 2, 0))
    if fig_title is not None:
        plt.title(fig_title, fontsize=33)
    plt.tight_layout()

    # Early exit, if not saving
    if not save_dir:
        return

    # Create directory, if not already exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f"{save_fname}.png"))


if __name__ == "__main__":
    # Verify seed setting
    seen_digits = (0, 3, 5, 6, 8, 9)
    data_0 = load_data(seen_digits=seen_digits, seed=0, plot=True)
    data_1 = load_data(seen_digits=seen_digits, seed=0)

    # Check labels match up for training set
    train_dataset_0 = data_0["id_train_seen"]
    train_dataset_1 = data_1["id_train_seen"]

    assert len(train_dataset_0) == len(train_dataset_1)
    for i in range(len(train_dataset_0)):
        assert train_dataset_0[i][1] == train_dataset_1[i][1], f"Labels don't match for index {i}"
