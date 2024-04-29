import matplotlib.pyplot as plt
import numpy as np
from data import data_augmentation

def plot_pictures(data):
    plt.figure(figsize=(10, 10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(data.class_names[int(labels[i])])
            plt.axis("off")
    plt.show()

def visualise_augmentation(data):
    plt.figure(figsize=(10, 10))
    for images, _ in data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
    plt.show()