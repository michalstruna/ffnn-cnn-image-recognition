from abc import ABC


# Abstract class for other neural networks.
class NeuralNetwork(ABC):
    pass


# Feedforward neural network.
class FFNN(NeuralNetwork):

    def __str__(self):
        return "Feedforward neural network (FFNN)"


# Convolutional neural network.
class CNN(NeuralNetwork):

    def __str__(self):
        return "Convolution neural network (CNN)"
