import matplotlib.pyplot as plt
from numpy import ndarray


def plot_image_label(batch_image: ndarray, batch_label: ndarray, slice_idx: int):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(batch_image[0, 0, :, :, slice_idx], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(batch_label[0, 0, :, :, slice_idx], cmap="gray")
    plt.show()
