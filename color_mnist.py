import numpy as np
from PIL import Image
import torch
import torchvision

import os


COLOR_ARR_LST = np.array([
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 165/255, 0],  # orange
    [128/255, 0, 128/255],  # purple
    [1, 1, 0],  # yellow
    [1, 192/255, 203/255],  # pink
    [0, 128/255, 128/255],  # teal
    [1, 1, 1]  # white
])


COLOR_NAME_LST = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "yellow",
    "pink",
    "teal",
    "white"
]


def color_image(img, label):
    return (np.array(img.convert("RGB")) * COLOR_ARR_LST[label]).astype(np.uint8)


def create_dataset():
    # TODO: need to change colors for the validation set
    MNIST_train = torchvision.datasets.MNIST("./data", download=True)
    MNIST_val = torchvision.datasets.MNIST("./data", train=False, download=True)

    cmnist_path = os.path.join("data", "CMNIST")
    if not os.path.isdir(cmnist_path):
        os.mkdir(cmnist_path)

    training_path = os.path.join(cmnist_path, "training")
    if not os.path.isdir(training_path):
        os.mkdir(training_path)
        for i in range(9):
            os.mkdir(os.path.join(training_path, str(i)))

    val_path = os.path.join(cmnist_path, "validation")
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        for i in range(9):
            os.mkdir(os.path.join(val_path, str(i)))

    counter = 0
    for img, label in MNIST_train:
        if label == 9:  # skip 9 for now
            continue

        colored_img = color_image(img, label)
        im = Image.fromarray(colored_img)
        img_path = os.path.join(training_path, str(label),
                                f"{label}_{counter}.jpeg")
        
        im.save(img_path)
        counter += 1

    for img, label in MNIST_val:
        if label == 9:
            continue
        colored_img = color_image(img, label)
        im = Image.fromarray(colored_img)
        img_path = os.path.join(val_path, str(label),
                                f"{label}_{counter}.jpeg")
        
        im.save(img_path)
        counter += 1


if __name__=="__main__":
    create_dataset()
