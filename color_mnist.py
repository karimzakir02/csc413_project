import numpy as np
from PIL import Image
import torch
import torchvision

import os
import random


COLOR_ARR_LST = np.array([
    [1, 0, 0],  # red
    [0, 1, 0],  # lime
    [0, 0, 1],  # blue
    [1, 165/255, 0],  # orange
    [128/255, 0, 128/255],  # purple
    [1, 1, 0],  # yellow
    [1, 192/255, 203/255],  # pink
    [0, 128/255, 128/255],  # teal
    [1, 1, 1],  # white
    [0, 128/255, 0]  # green
])


COLOR_NAME_LST = [
    "red",
    "lime",
    "blue",
    "orange",
    "purple",
    "yellow",
    "pink",
    "teal",
    "white",
    "green"
]


class CMNIST(torch.utils.data.Dataset):

    def __init__(self, root="data", training=True, exclude_digits=None):
        super().__init__()
        path = os.path.join(root, "CMNIST", 
                            "training" if training else "validation")
        
        init_dataset = torchvision.datasets.ImageFolder(path,
                                                   transform=torchvision.transforms.ToTensor())
    
        if exclude_digits is None:
            exclude_digits = []

        subset_idx = (~np.isin(init_dataset.targets, exclude_digits)).nonzero()[0]
        self.dataset = torch.utils.data.Subset(init_dataset, subset_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def color_image_by_label(img, label):
    return (np.array(img.convert("RGB")) * COLOR_ARR_LST[label]).astype(np.uint8)


def color_image_random(img):
    color_idx = random.randrange(len(COLOR_ARR_LST))
    color_name = COLOR_NAME_LST[color_idx]
    return ((np.array(img.convert("RGB")) * COLOR_ARR_LST[color_idx]).astype(np.uint8), 
            color_name)


def create_dataset(seed=413):
    random.seed(seed)

    if not os.path.isdir("data"):
        os.mkdir("data")

    MNIST_train = torchvision.datasets.MNIST("./data", download=True)
    MNIST_val = torchvision.datasets.MNIST("./data", train=False, download=True)

    cmnist_path = os.path.join("data", "CMNIST")
    if not os.path.isdir(cmnist_path):
        os.mkdir(cmnist_path)

    training_path = os.path.join(cmnist_path, "training")
    if not os.path.isdir(training_path):
        os.mkdir(training_path)
        for i in range(10):
            os.mkdir(os.path.join(training_path, str(i)))

    val_path = os.path.join(cmnist_path, "validation")
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        for i in range(10):
            os.mkdir(os.path.join(val_path, str(i)))

    counter = 0
    for img, label in MNIST_train:
        colored_img = color_image_by_label(img, label)
        im = Image.fromarray(colored_img)
        img_path = os.path.join(training_path, str(label),
                                f"{label}_{counter}.jpeg")

        im.save(img_path)
        counter += 1

    counter = 0
    for img, label in MNIST_val:

        colored_img, color = color_image_random(img)
        im = Image.fromarray(colored_img)
        img_path = os.path.join(val_path, str(label),
                                f"{color}_{counter}.jpeg")

        im.save(img_path)
        counter += 1


if __name__=="__main__":
    create_dataset()
