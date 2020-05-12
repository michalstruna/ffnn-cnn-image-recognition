from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ai import FFNN, CNN


def read_args():
    parser = ArgumentParser()

    parser.add_argument("--run", "-r", type=str, metavar="FILE",
                        help="Decides whether human head is in picture or not.")

    parser.add_argument("--use", "-u", type=str.upper, choices=("FFNN", "CNN"),
                        help="Specify neural network used for selected operation. If not specified, both of FFNN and "
                             "CNN will be used.")

    parser.add_argument("--dir", "-d", type=str,
                        help="Specify dirs for testing or training. If not specified, positive/test and negative/test "
                             "will be used for test and positive/train and negative/train will be used for training.")

    parser.add_argument("--test", "-t", action="store_true",
                        help="Test selected neural network (--use) with all images in selected dir (--dir).")

    parser.add_argument("--train", "-tr", type=float, nargs=4, metavar=("PATIENCE", "RATE", "BATCH_SIZE", "VALIDATION"),
                        help="Train selected neural network (--use) with all images in selected dir (--dir).")

    return parser.parse_args()


# Depends on program argument --use return list of networks that will be used for selected operation.
def get_networks(use):
    if not use: # If --use is not specified, return both of CNN and FFNN.
        return [CNN(), FFNN()]
    else:
        return [FFNN()] if use == "FFNN" else [CNN()]
    

def prepare_input_set(path_dir, train=False, transform_input=None, val_split=0.15):
    if transform_input:
        if train:
            generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(path_dir, batch_size=1, target_size=(51, 51))
            return get_data_from_generator(generator, transform_input, generator.n)
        else:
            generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(path_dir, target_size=(51, 51), batch_size=1, shuffle=False)
            inp, out = get_data_from_generator(generator, transform_input, generator.n)
            return inp, out
    else:
        if train:
            generator = ImageDataGenerator(validation_split=val_split, rescale=1./255, rotation_range=45, height_shift_range=0.1, brightness_range=(0.5, 1), shear_range=0.1, zoom_range=0.1, channel_shift_range=0.3)
            train_set = generator.flow_from_directory(path_dir, batch_size=1, target_size=(51, 51), subset="training", color_mode="grayscale")
            val_set = generator.flow_from_directory(path_dir, batch_size=1, target_size=(51, 51), subset="validation", color_mode="grayscale")
            return train_set, val_set
        else:
            return ImageDataGenerator(rescale=1. / 255).flow_from_directory(path_dir, target_size=(51, 51), batch_size=1, shuffle=False, color_mode="grayscale")


def get_data_from_generator(generator, transform, count):
    imgs, targets = next(generator)
    generator.reset()

    inputs_shape = list(transform(imgs[0]).shape)
    inputs_shape.insert(0, count)
    inputs_shape = tuple(inputs_shape)

    targets_shape = list(targets.shape)
    targets_shape[0] = count
    targets_shape = tuple(targets_shape)

    inputs, targets = np.zeros(inputs_shape), np.zeros(targets_shape)

    for i in range(inputs_shape[0]):
        imgs, outputs = next(generator)
        inputs[i, :], targets[i, :] = transform(imgs[0, :]), outputs[0, :]

    return inputs, targets


def col(value, char=" ", width=11):
    return str(value).ljust(width, char)


class Console:

    def __init__(self):
        self.output = ""
        self.row_i = 0

    def f(self, value, length=0):
        return ("{:." + str(length) + "e}").format(value).replace("-0", "-") if value < 0.1 else np.trunc(value * 100) / 100

    def print_run(self, network, result):
        if self.row_i == 0:
            self.row("", "Výsledek", "Zaokrouhl.", "Závěr")

        f_result = "[" + str(self.f(result[0][0])) + " " + str(self.f(result[0][1])) + "]"

        rounded = np.round(result[0])

        self.row(network, f_result, rounded.astype(int), "Je hlava" if rounded[0] > rounded[1] else "Není hlava")

    def print_test(self, network, trues, falses):
        all = trues + falses
        success = round(100 * trues / all, 2)

        if self.row_i == 0:
            self.row("", "Správně", "Špatně", "Celkem", "Úspěšnost")

        self.row(network, trues, falses, all, str(success) + " %")

    def print_training(self, network, history, time):
        if self.row_i == 0:
            self.row("", "loss", "loss_val", "Doba")

        f_time = str(round(time / 60)) + " m " + str(round(time % 60)) + " s"
        loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        self.row(network, self.f(loss, 2), self.f(val_loss, 2), f_time)

        plt.plot(history.history["loss"], label=str(network) + " loss")
        plt.plot(history.history["val_loss"], label=str(network) + " val_loss")
        plt.ylabel("Chyba")
        plt.xlabel("Epocha")
        plt.legend()
        plt.grid(True)

    def show(self):
        print(self.output)
        self.output = ""
        self.row_i = 0
        plt.show()

    def row(self, *cells):
        if self.row_i == 0:
            self.output += "-+-"

            for cell in cells:
                self.output += col("", "-") + "-+-"

        self.output += "\n | "

        for cell in cells:
            self.output += col(cell) + " | "

        self.output += "\n-+-"

        for cell in cells:
            self.output += col("", "-") + "-+-"

        self.row_i += 1
