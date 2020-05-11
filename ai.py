import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from abc import ABC, abstractmethod
from skimage import feature
import numpy as np


# Abstract class for other neural networks.
class NeuralNetwork(ABC):

    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self, input_dim):
        pass

    @abstractmethod
    def transform_input(self, input):
        pass

    def run(self, inputs):
        final_inputs = self.transform_inputs(inputs)
        return self.model.predict(final_inputs)

    def train(self, inputs, targets, patience, batch_size, val_split):
        self.build(inputs[0].shape)
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience)
        return self.model.fit(inputs, targets, epochs=5000, callbacks=[early_stopping], validation_split=val_split, batch_size=batch_size, verbose=2)

    def test(self, inputs, targets):
        outputs = self.model.predict(inputs)
        trues = (np.argmax(targets, 1) == np.argmax(outputs, 1)).sum()
        return trues, len(inputs) - trues

    def summary(self):
        self.model.summary()

    def get_default_filename(self):
        return type(self).__name__.lower() + ".h5"

    def save(self, filename=None):
        self.model.save(filename if filename else self.get_default_filename())

    def load(self, filename=None):
        self.model = load_model(filename if filename else self.get_default_filename())


class FFNN(NeuralNetwork):

    def __str__(self):
        return "FFNN"

    def build(self, input_shape):
        self.model = Sequential([
            Dense(8, input_dim=input_shape[0], activation=tf.nn.tanh),
            Dropout(0.4),
            Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), loss="mse")

    def transform_input(self, input):
        return feature.hog(input, orientations=9, pixels_per_cell=(2, 2))


class CNN(NeuralNetwork):

    def __str__(self):
        return "CNN"

    def build(self, input_shape):
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=input_shape),
            MaxPooling2D(),
            Flatten(),
            Dense(32, activation=tf.nn.relu),
            Dropout(0.5),
            Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss="mse")

    def transform_input(self, input):
        return input