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
    def build(self, input_dim, rate):
        pass

    @abstractmethod
    def transform_input(self, input):
        pass

    def run(self, inputs):
        final_inputs = self.transform_input(inputs)
        return self.model.predict(final_inputs)

    def train(self, inputs, targets, patience, rate, batch_size, val_split):
        self.build(inputs[0].shape, rate)
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


class GeneratorNeuralNetwork(NeuralNetwork, ABC):

    def transform_input(self, input):
        return input

    def train(self, train_set, val_set, patience, rate, batch_size):
        imgs, targets = next(train_set)
        train_set.reset()
        self.build(imgs[0].shape, rate)
        early_stopping = EarlyStopping(monitor="val_loss", patience=patience)
        return self.model.fit(train_set, epochs=5000, callbacks=[early_stopping], validation_data=val_set, batch_size=batch_size, verbose=2)

    def test(self, test_set):
        outputs = np.argmax(self.model.predict(test_set), 1)
        trues = (outputs == test_set.labels).sum()
        return trues, test_set.n - trues


class FFNN(NeuralNetwork):

    def __str__(self):
        return "FFNN"

    def build(self, input_shape, rate):
        self.model = Sequential([
            Dense(4, input_dim=input_shape[0], activation=tf.nn.tanh),
            Dropout(0.4),
            Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=rate), loss="mse")

    def transform_input(self, input):
        return feature.hog(input, orientations=9, pixels_per_cell=(2, 2))


class CNN(GeneratorNeuralNetwork):

    def __str__(self):
        return "CNN"

    def build(self, input_shape, rate):
        self.model = Sequential([
            Conv2D(16, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=input_shape),
            MaxPooling2D(),
            Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation=tf.nn.relu),
            Dropout(0.3),
            Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=rate), loss="mse")