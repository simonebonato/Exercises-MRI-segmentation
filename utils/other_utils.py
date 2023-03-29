import matplotlib.pyplot as plt
from numpy import ndarray


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


def plot_image_label(
    batch_image: ndarray, batch_label: ndarray, slice_idx: int
) -> None:
    """
    Plots the image and the label of a batch, and the image with the label overlaid.

    Parameters
    :arg batch_image: Batch of images
    :arg batch_label: Batch of labels
    :arg slice_idx: Index of the slice to plot

    """

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(batch_image[0, 0, :, :, slice_idx], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(batch_label[0, 0, :, :, slice_idx], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("image + label")
    plt.imshow(batch_image[0, 0, :, :, slice_idx], cmap="gray")
    plt.imshow(batch_label[0, 0, :, :, slice_idx], alpha=0.5)
    plt.show()
