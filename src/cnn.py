import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import keras
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

class Cnn:
    def __init__(self, num_class=84, input_shape=(128, 128, 3)):
        self.num_class = num_class
        self.input_shape = input_shape
        self.model = self.build_model()
        
    def get_model(self):
        return self.model
    
    def build_model_simple(self):
        num_filters = 8
        filter_size = 3
        pool_size = 2

        model = Sequential([
            Conv2D(num_filters, filter_size, input_shape=(128, 128, 3)),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(self.num_class, activation='softmax'),
        ])
        return model

    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # Entry block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])
            previous_block_activation = x

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        if self.num_class == 2:
            units = 1
        else:
            units = self.num_class

        x = layers.Dropout(0.25)(x)
        outputs = layers.Dense(units, activation=None)(x)
        return keras.Model(inputs, outputs)

    def build_model_bad(self):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.num_class, activation='softmax'))
        return model


    def compile_model(self, train, val, nb_epoch=25, learning_rate=0.0001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
        )
        self.model.fit(
            train,
            epochs=nb_epoch,
            validation_data=val,
        )
    
    def predict(self, test):
        all_predictions = []
        all_actual_labels = []

        # Iterate through the test dataset
        for images, labels in test:
            # Perform prediction on the current batch
            predictions = self.model.predict(images)
            
            # Convert predictions to classes
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Store predicted classes and actual labels
            all_predictions.extend(predicted_classes)
            all_actual_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_actual_labels = np.array(all_actual_labels)

        return all_predictions, all_actual_labels
    
    def grid_search(self, grid, train, val, test):
        best_score = -1
        best_params = None
        best_model = None
        best_history = None

        for lr in grid['learning_rate']:
            for epoch in grid['epoch']:
                model = self.build_model()
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
                )
                history = model.fit(
                    train,
                    epochs=epoch,
                    validation_data=val,
                )
                loss, accuracy = model.evaluate(test)
                print(f"========================Test with a learning rate : {lr} and a epoch of {epoch}. The accuracy is {accuracy} and the loss {loss}========================")
                
                if accuracy != 1 and accuracy > best_score:
                    best_score = accuracy
                    best_params = {"learning_rate": lr, "epoch": epoch}
                    best_model = model
                    best_history = history

        
        print("Best score:", best_score)
        print("Best params: ", best_params)
        self.model = best_model
        
        return best_score, best_params, best_model, best_history

    def display_history(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history.get('acc')
        val_accuracy = history.history.get('val_acc')

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        if accuracy and val_accuracy:
            plt.plot(accuracy, label='Training Accuracy')
            plt.plot(val_accuracy, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.savefig("grid_search_results_cnn6.png")
        plt.show()
        plt.close()
