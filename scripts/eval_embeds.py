"""
eval_embeds.py

Description: Used to evaluate embeddings on OOD data (unseen digits)
"""

# Standard libraries
import argparse
import json
import os

# Non-standard libraries
import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from scipy import stats
from sklearn import metrics as skmetrics
from sklearn.cluster import KMeans

import sys

sys.path.append(os.path.abspath("."))

# Custom libraries
from utils import data


################################################################################
#                                  Constants                                   #
################################################################################
# Random seed
SEED = 0

# Directories
DIR_CKPT = "checkpoints"
DIR_RESULTS = "results"

# Mapping of model to checkpoint subdirectory
MODEL_TO_SUBDIR = {
    # 0. Randomly Initialize Baseline
    "random_baseline": os.path.join("baseline", "enyn0xwy"),

    # 0.1 ZerO Initialized Baseline
    "zero_baseline": os.path.join("baseline", "y0bjpqz0"),

    # 1. Baseline (trained on ID data with seen digits)
    "id_baseline": os.path.join("baseline", "b57lgtqj"),

    # 2. Baseline (trained on OOD data (cov. shift) with seen digits)
    "ood_baseline": os.path.join("baseline", "sfacff20"),

    # 3. Disagreement Model (trained on ID and unlabeled OOD data with seen digits)
    "cdc": os.path.join("cdc", "he671o82"),

    # 4. Self-Supervised Model (trained on OOD data with seen digits)
    "ssl_byol": os.path.join("ssl_byol_encoder"),
}


################################################################################
#                                  Functions                                   #
################################################################################
def load_embeds(model_name, seen=False):
    """
    Load embeddings given model name

    Parameters
    ----------
    model_name : str
        Name of model
    seen : bool, optional
        If True, load embeddings for OOD (seen digits with random colors).
        Otherwise, load embeddings for OOD (unseen digits with random colors).

    Returns
    -------
    np.array
        Embedding for each sample in the OOD test set (seen/unseen digits)
    """
    embed_dir = os.path.join(DIR_CKPT, MODEL_TO_SUBDIR[model_name])
    embed_path = os.path.join(embed_dir, f"ood_test_{'seen' if seen else 'unseen'}_feats.npz")
    assert os.path.exists(embed_path), "OOD unseen digits embeddings don't exist!"

    # Load embeddings
    with np.load(embed_path) as data:
        embeds = data["embeds"]

    return embeds


def plot_2d(embeds, labels, save_dir=None):
    """
    Create 2D UMAP plot of embeddings.

    Parameters
    ----------
    embeds : np.array
        (N, D) array of high-dimensional embeddings
    labels : list
        Labels for each of the N samples
    save_dir : str, optional
        If provided, save figure to directory, by default None
    """
    # Create a UMAP instance
    reducer = umap.UMAP()

    # Fit the model to your data and transform it to 2D
    embeddings_2d = reducer.fit_transform(embeds)

    # Set the style of the plot to be more aesthetically pleasing
    sns.set_theme(style='white', context='paper', rc={'figure.figsize':(14,10)})

    # Create a scatter plot
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette='colorblind', s=50, alpha=0.7)

    # Remove the axis
    plt.gca().set_axis_off()

    # Fix legends
    plt.legend(title='Unseen Digits', fontsize='medium', title_fontsize='x-large')

    # Add a title with a larger font size
    plt.title("2D UMAP Embeddings", fontsize=24)

    # Remove the grid
    plt.grid(False)

    # If not provided, skip saving
    if not os.path.exists(save_dir):
        return plt.gca()

    # Save the figure in high resolution
    save_path = os.path.join(save_dir, f'umap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gca()


def compute_cluster_metrics(embeds, labels, method="kmeans"):
    """
    Cluster high-dimensional embeddings and compute accuracy with labels.

    Parameters
    ----------
    embeds : np.array
        (N, D) array of high-dimensional embeddings
    labels : list
        Labels for each of the N samples
    method : str
        Clustering methods, by default kmeans

    Returns
    -------
    dict
        Dictionary containing metrics from K-Means clustering
    """
    np.unique(labels)

    # Define the number of clusters
    n_clusters = len(np.unique(labels))

    # Create a model instance
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
    else:
        raise NotImplementedError(f"Clustering method `{method}` not implemented!")

    # Fit the model to your data
    model.fit(embeds)

    # Predict the labels of the embeddings
    preds = model.predict(embeds)

    # Compute metrics
    metrics = {
        "adjusted_rand_score": float(skmetrics.adjusted_rand_score(labels, preds)),
        "adjusted_mutual_info": float(skmetrics.adjusted_mutual_info_score(labels, preds)),
        "homogeneity": float(skmetrics.homogeneity_score(labels, preds)),
        "completeness": float(skmetrics.completeness_score(labels, preds)),
        "silhouette": float(skmetrics.silhouette_score(embeds, preds, metric='euclidean')),
    }
    print(metrics)
    return metrics


def compute_knn_accuracies(embeds, labels, k_s=(1, 3, 5, 7)):
    """
    Compute kNN accuracy given embeddings.

    Parameters
    ----------
    embeds : np.array
        (N, D) array of high-dimensional embeddings
    labels : list
        Labels for each of the N samples
    k_s : int
        Number of neighbors to do k-NN, by default (1, 3, 5, 7)

    Returns
    -------
    dict
        k-NN accuracy for various k's provided
    """
    print("Peforming kNN...")
    N = len(labels)
    knn_metrics = {}

    # Build index (on GPU)
    index = faiss.IndexFlatL2(embeds.shape[1])
    # Send to GPU, if possible
    if torch.cuda.is_available():
        print("Moving kNN index to GPU...")
        resource = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resource, 0, index)

    # Fit nearest neighbors model
    index.add(embeds)

    # Find greatest k+1 closest neighbors for each point
    max_k = max(k_s)
    _, neighbors = index.search(embeds, k=max_k+1)

    # Remove the first dimension, since closest neighbor will be the same point
    neighbors = neighbors[:, 1:]

    # Broadcast 1d labels array (to be indexable)
    labels_2d = np.broadcast_to(labels, (N, N))

    # For each of k, compute the accuracy of k neighbors
    for k in k_s:
        print(f"Peforming {k}-NN...")
        # Get the majority label among the first k neighbors
        curr_neighbors = neighbors[:, :k]

        # Create boolean mask for label indices
        mask = np.zeros((N, N), dtype=bool)
        np.put_along_axis(mask, curr_neighbors, values=True, axis=1)

        # Get majority label among neighbors
        curr_neighbor_labels = labels_2d[mask].reshape(N, k)
        preds, _ = stats.mode(curr_neighbor_labels, axis=1)

        # Compute accuracy
        knn_metrics[f"{k}-nn_accuracy"] = round((preds == labels).mean(), 4)

    print("Peforming k-NN...Done")
    return knn_metrics


def main(model_name, seen_digits=(0, 3, 5, 6, 8, 9), seen=False):
    """
    Perform evaluation on OOD test set (seen or unseen digits) with random
    colors.

    Parameters
    ----------
    model_name : str
        Name of model
    seen_digits : tuple, optional
        List of digits seen during training, by default (0, 3, 5, 6, 8, 9)
    seen : bool, optional
        If True, evaluate OOD test set with seen digits. Otherwise, evaluate
        OOD test set with unseen digits, by default False
    """
    seen_key = "ood_test_seen" if seen else "ood_test_unseen"

    # Load OOD test data (unseen digits)
    ood_test_dataset = data.load_data(seen_digits, torch.device("cpu"))[seen_key]
    labels = [y.item() for _, y in ood_test_dataset]

    # Map labels to seen/unseen digits
    # NOTE: Labels were previously encoded
    test_digits = list(seen_digits)
    if not seen:
        test_digits = [digit for digit in range(10) if digit not in seen_digits]
    decode_label = {idx: digit for idx, digit in enumerate(test_digits)}
    decoded_labels = [decode_label[label] for label in labels]

    # Load model embeddings
    embeds = load_embeds(model_name, seen=seen)

    # Create directory to save results
    save_dir = os.path.join(DIR_RESULTS, model_name, seen_key)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Plot 2D UMAP
    plot_2d(embeds, decoded_labels, save_dir)

    # 2. Perform k-NN on embeddings
    metrics = {}
    metrics.update(compute_knn_accuracies(embeds, labels))

    # 3. Cluster embeddings
    metrics.update(compute_cluster_metrics(embeds, labels, method="kmeans"))

    # Save metrics
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    # Set up parser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--model_name", type=str, required=True,
        choices=list(MODEL_TO_SUBDIR.keys()))
    PARSER.add_argument(
        "--seen_digits", nargs="+", type=int,
        default=(0, 3, 5, 6, 8, 9),
    )
    PARSER.add_argument(
        "--seen", action="store_true",
    )

    # Parse arguments
    ARGS = PARSER.parse_args()

    # Call main
    main(ARGS.model_name, ARGS.seen_digits, ARGS.seen)
