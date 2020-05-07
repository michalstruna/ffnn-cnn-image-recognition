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
    def transform_inputs(self, inputs):
        pass

    def run(self, inputs):
        final_inputs = self.transform_inputs(inputs)
        return self.model.predict(final_inputs)

    def train(self, inputs, targets, epochs, batch, val):
        final_inputs = self.transform_inputs(inputs)
        self.build(final_inputs[0].shape)

        early_stopping = EarlyStopping(monitor="val_loss", patience=100)

        return self.model.fit(final_inputs, targets, epochs=epochs, callbacks=[early_stopping], batch_size=batch, validation_split=val, verbose=2)

    def test(self, inputs, targets):
        outputs = np.round(self.run(inputs))
        trues, falses = 0, 0

        for i in range(len(outputs)):
            if np.array_equal(outputs[i], targets[i]):
                trues += 1
            else:
                falses += 1

        return trues, falses

    def summary(self):
        self.model.summary()

    def get_default_filename(self):
        return type(self).__name__.lower() + ".h5"

    def save(self, filename=None):
        self.model.save(filename if filename else self.get_default_filename())

    def load(self, filename=None):
        self.model = load_model(filename if filename else self.get_default_filename())


# Feedforward neural network.
class FFNN(NeuralNetwork):

    def __str__(self):
        return "FFNN"

    def build(self, input_shape):
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=input_shape[0], activation=tf.nn.tanh))
        self.model.add(Dense(2, activation=tf.nn.softmax))
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mse")

    def transform_inputs(self, inputs):
        hog_size = self.hog(inputs[0]).shape[0]

        result = np.zeros((inputs.shape[0], hog_size))

        for i in range(len(inputs)):
            result[i, :] = self.hog(inputs[i])

        return result

    def hog(self, img):
        return feature.hog(img, orientations=9, pixels_per_cell=(2, 2))


# Convolution neural network.
class CNN(NeuralNetwork):

    def __str__(self):
        return "CNN"

    def build2(self, input_shape):
        self.model = Sequential([
            Conv2D(filters=8, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=input_shape, padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(8, activation=tf.nn.tanh),
            Dense(2, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001), loss="mse")

    def transform_inputs(self, inputs):
        return inputs