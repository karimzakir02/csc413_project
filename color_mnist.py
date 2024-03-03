import numpy as np
import torch
import torchvision


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
    return np.array(img.convert("RGB")) * COLOR_ARR_LST[label] / 255


def main():
    MNIST = torchvision.datasets.MNIST("./data", download=True)
    X, y = MNIST[0]
    if y != 9:  # not doing this for 9, excluded from the dataset
        colored_img = color_image(X, y)


if __name__=="__main__":
    main()
