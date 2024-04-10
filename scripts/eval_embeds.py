"""
eval_embeds.py

Description: Used to evaluate embeddings on OOD data (unseen digits)
"""

# Standard libraries
import argparse
import json
import os

# Non-standard libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skmetrics
from sklearn.cluster import KMeans
import seaborn as sns
import torch
import umap

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
MODEL_TO_EMBEDS = {
    "cdc": "cdc/2024-04-10_00-03-35/ood_test_unseen_feats.npz"
}


################################################################################
#                                  Functions                                   #
################################################################################
def load_embeds(model_name):
    """
    Load embeddings given model name

    Parameters
    ----------
    model_name : str
        Name of model

    Returns
    -------
    np.array
        Embedding for each sample in the OOD unseen digits set
    """
    embed_path = os.path.join(DIR_CKPT, MODEL_TO_EMBEDS[model_name])
    assert os.path.exists(embed_path), "OOD unseen digits embeddings don't exist!"

    # Load embeddings
    with open(embed_path, "r") as f:
        embeds = np.load(f)

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
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral', s=50)

    # Remove the axis
    plt.gca().set_axis_off()

    # Add a colorbar
    cbar = plt.colorbar(boundaries=np.arange(len(np.unique(labels))+1)-0.5)
    cbar.set_ticks(np.arange(len(np.unique(labels))))
    cbar.set_ticklabels(np.unique(labels))

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
        "adjusted_rand_score": skmetrics.adjusted_rand_score(labels, preds),
        "adjusted_mutual_info": skmetrics.adjusted_mutual_info_score(labels, preds),
        "homogeneity": skmetrics.homogeneity_score(labels, preds),
        "completeness": skmetrics.completeness_score(labels, preds),
        "silhouette": skmetrics.silhouette_score(embeds, preds, metric='euclidean'),
    }
    print(metrics)


def main(model_name, seen_digits=(0, 3, 5, 6, 8, 9)):
    # Load OOD test data (unseen digits)
    ood_test_unseen_dataset = data.load_data(seen_digits, torch.device("cpu"))["ood_test_unseen"]
    labels = [y for _, y in ood_test_unseen_dataset]

    # Load model embeddings
    embeds = load_embeds(model_name)

    # Create directory to save results
    save_dir = os.path.join(DIR_RESULTS, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Plot 2D UMAP
    plot_2d(embeds, labels, save_dir)

    # 2. Cluster embeddings
    cluster_metrics = compute_cluster_metrics(embeds, labels, method="kmeans")

    with open(os.path.join(save_dir, "cluster_metrics.json"), "w") as f:
        json.dump(cluster_metrics, f, indent=4)


if __name__ == "__main__":
    # Set up parser
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--model_name", type=str, required=True,
        choices=list(MODEL_TO_EMBEDS.keys()))
    PARSER.add_argument(
        "--seen_digits", nargs="+",
        default=(0, 3, 5, 6, 8, 9),
    )

    # Parse arguments
    ARGS = PARSER.parse_args()

    # Call main
    main(ARGS.model_name, ARGS.seen_digits)
