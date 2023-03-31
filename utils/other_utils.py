import matplotlib.pyplot as plt
from numpy import ndarray
import json
import numpy as np
from numpy import ndarray
from matplotlib.image import AxesImage


def read_json(file_path: str = "config.json") -> dict:
    """
    Reads a json file and returns the content as a dictionary.

    Parameters
    :arg file_path: Path to the json file

    Returns
    :return: Dictionary containing the content of the json file

    """

    with open(file_path, "r") as f:
        content = json.load(f)
    return content


def assert_config(config: dict) -> None:
    """
    Asserts that the config is correct before setting up the trainer.

    Parameters
    :arg config: Dictionary containing the configuration of the trainer

    """
    assert config["batch_size"] > 0, "Batch size must be greater than 0"
    assert (
        config["train_val_ratio"] > 0 and config["train_val_ratio"] < 1
    ), "Train val ratio must be between 0 and 1"
    assert config["epochs"] > 0, "Epochs must be greater than 0"
    assert config["lr"] > 0, "Learning rate must be greater than 0"
    assert (
        isinstance(config["early_stopping"], int) and config["early_stopping"] > 0
    ) or config["early_stopping"] is None, "Early stopping must be an integer or None"
    assert isinstance(config["fine_tune"], bool), "Fine tune must be a boolean"
    assert isinstance(
        config["mixed_precision"], bool
    ), "Mixed precision must be a boolean"
    assert (isinstance(config["Nit"], int) and config["Nit"] > 0) or config[
        "Nit"
    ] is None, "Nit must be an integer or None"
    assert isinstance(
        config["best_models_dir"], str
    ), "Best models dir must be a string"
    assert (
        isinstance(config["train_from_checkpoint"], str)
        or config["train_from_checkpoint"] is None
    ), "Train from checkpoint must be a string or None"


def get_patches_legend(x: AxesImage, label: ndarray):
    """
    Returns a list of patches to be used as a legend in the plot.

    Parameters
    :arg x: AxesImage object to get the colormap from
    :arg label: Label array

    Returns
    :return: List of patches to be used as a legend in the plot

    """
    labels = {1: "Anterior", 2: "Posterior"}
    colors = x.cmap(x.norm(np.unique(label)))
    patches = [
        plt.plot(
            [],
            [],
            marker="o",
            ms=10,
            ls="",
            mec=None,
            color=colors[i],
            label=f"{labels[i]}",
        )[0]
        for i in range(len(colors))[1:]
    ]
    return patches


def plot_image_label(
    batch_image: ndarray, batch_label: ndarray, slice_idx: int, legend: bool = False
) -> None:
    """
    Plots the image and the label of a batch, and the image with the label overlaid.

    Parameters
    :arg batch_image: Batch of images
    :arg batch_label: Batch of labels
    :arg slice_idx: Index of the slice to plot

    """

    image = batch_image[1, 0, :, :, slice_idx]
    label = batch_label[1, 0, :, :, slice_idx]

    # image plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(image, cmap="gray")

    # label plot
    plt.subplot(1, 3, 2)
    plt.title("label")
    x = plt.imshow(label, cmap="gray", interpolation="none")

    if legend:
        # create legend for labels
        patches = get_patches_legend(x, label)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # image + label plot
    plt.subplot(1, 3, 3)
    plt.title("image + label")
    plt.imshow(image, cmap="gray")
    x = plt.imshow(label, alpha=0.5, cmap="jet", interpolation="none")

    if legend:
        patches = get_patches_legend(x, label)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.tight_layout()
    plt.show()
