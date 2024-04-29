import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image 

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
                    # Delete corrupted image
                    os.remove(fpath)
    print(f"Deleted {num_skipped} images for {dir_name}.")

def png_to_jpg(dir_path):

    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
        for file in os.listdir(folder_path):
            fpath = os.path.join(folder_path, file)
            if fpath.endswith('.png'): 
                im = Image.open(fpath).convert("RGB")
                im.save(fpath[:-4] + '.jpg', quality=95, optimize=True) 
                os.remove(fpath)

def create_dataset():
    image_size = (180, 180)
    batch_size = 128

    train, val = keras.utils.image_dataset_from_directory(
        "../handwritten-mathematical-expressions/finaltrain",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    test = keras.utils.image_dataset_from_directory(
        "../handwritten-mathematical-expressions/finaltest",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return (train, val, test)

def clean_data_files():
    png_to_jpg("../handwritten-mathematical-expressions/finaltrain")
    png_to_jpg("../handwritten-mathematical-expressions/finaltest")
    clean_corrupted("../handwritten-mathematical-expressions/finaltrain")
    clean_corrupted("../handwritten-mathematical-expressions/finaltest")

def get_data():
    clean_data_files()
    train, val, test = create_dataset()