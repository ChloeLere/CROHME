import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def plot_pictures(data):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Sample images and labels from our dataset")
    plt.axis("off")
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(data.class_names[int(labels[i])])
    plt.show()

def visualise_augmentation(data):
    plt.figure(figsize=(10, 10))
    plt.title(f"data augmentation of the character '3'.")
    plt.axis("off")
    for images, labels in data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
    plt.show()

def clean_corrupted(dir_name):
    num_skipped = 0
    for folder_name in os.listdir(dir_name):
        folder_path = os.path.join(dir_name, folder_name)
        if os.path.isdir(folder_path):
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = b"JFIF" in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath)
    print(f"Deleted {num_skipped} images for {dir_name}.")

def png_to_jpg(dir_path):
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
        for file in os.listdir(folder_path):
            fpath = os.path.join(folder_path, file)
            if fpath.endswith('.png'): 
                try:
                    im = Image.open(fpath).convert("RGB")
                    im.save(fpath[:-4] + '.jpg', quality=95, optimize=True) 
                    os.remove(fpath)
                except Exception as e:
                    print("error file", fpath)


def create_dataset():
    image_size = (128, 128)
    batch_size = 64

    train, val = keras.utils.image_dataset_from_directory(
        "../handwritten-mathematical-expressions/finaltrain",
        validation_split=0.2,
        subset="both",
        seed=28,
        image_size=image_size,
        batch_size=batch_size,
    )
    test = keras.utils.image_dataset_from_directory(
        "../handwritten-mathematical-expressions/finaltest",
        seed=None,
        image_size=image_size,
        batch_size=batch_size,
    )
    return (train, val, test)

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def clean_data_files():
    png_to_jpg("../handwritten-mathematical-expressions/finaltrain")
    png_to_jpg("../handwritten-mathematical-expressions/finaltest")
    clean_corrupted("../handwritten-mathematical-expressions/finaltrain")
    clean_corrupted("../handwritten-mathematical-expressions/finaltest")

def get_data():
    clean_data_files()
    train, val, test = create_dataset()
    class_names_test = test.class_names
    #plot_pictures(train)

    train = train.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train = train.prefetch(tf_data.AUTOTUNE)
    val = val.prefetch(tf_data.AUTOTUNE)

    #visualise_augmentation(train)
    return train, val, test, class_names_test

def format_data_for_svc(train, test):
    # Flatten images for SVM input
    X_train, y_train = next(iter(train))
    X_train_flat = np.array([x.flatten() for x in X_train.numpy()])
    y_train = np.array(y_train)

    X_test, y_test = next(iter(test))
    X_test_flat = np.array([x.flatten() for x in X_test.numpy()])

    return X_train_flat, y_train, X_test_flat, y_test
    
