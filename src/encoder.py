import numpy as np
import pandas as pdfrom 
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from tensorflow import keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def build_model(num_class, input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data formatx = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)  # now the representation is (32, 32, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.summary()

    return autoencoder

def compile_model(autoencoder, train, val, nb_epoch=25, learning_rate=0.0001):
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.fit(
        train,
        epochs=nb_epoch,
        validation_data=val,
    )
