from argparse import ArgumentParser
import os
from matplotlib import pyplot as plt
import numpy as np

from ai import FFNN, CNN


def read_args():
    parser = ArgumentParser()

    parser.add_argument("--run", "-r", type=str, metavar="FILE",
                        help="Decides whether human head is in picture or not.")

    parser.add_argument("--use", "-u", type=str.upper, choices=("FFNN", "CNN"),
                        help="Specify neural network used for selected operation. If not specified, both of FFNN and "
                             "CNN will be used.")

    parser.add_argument("--dir", "-d", type=str, nargs=2, metavar=("POS", "NEG"),
                        help="Specify dirs for testing or training. If not specified, positive/test and negative/test "
                             "will be used for test and positive/train and negative/train will be used for training.")

    parser.add_argument("--test", "-t", action="store_true",
                        help="Test selected neural network (--use)  with all images in selected dirs (--dir).")

    parser.add_argument("--train", "-tr", type=float, nargs=3, metavar=("EPOCHS", "BATCH_SIZE", "VALIDATION"),
                        help="Train selected neural network (--use) with all images in selected dirs (--dir).")

    return parser.parse_args()


# Depends on program argument --use return list of networks that will be used for selected operation.
def get_networks(use, neurons):
    if not use:
        return [CNN(), FFNN()]
    else:
        return [FFNN()] if use == "FFNN" else [CNN()]


def print_network_type(network):
    print("\n" + "=" * (len(str(network)) + 42))
    print("=" * 20, network, "=" * 20)
    print("=" * (len(str(network)) + 42) + "\n")


def read_img(path):
    image = plt.imread(path)
    return image

def show_img(img):
    plt.imshow(img)
    plt.show()


def create_output(value, size):
    """
    Parameters:
        value: int - Order of output value that should have value 1.
        size: int - Count of output values.

    Returns:
        np.array - Array with only zeros and one 1 at position value, e. g. [0, 1, 0, 0] for value = 1 and size = 4.
    """
    output = np.zeros(size)
    output[value] = 1
    return output


def prepare_input_set(dirs, transform=False):
    """
    Prepare input set for testing or training.

    Parameters:
        dirs: string[] - Images will have same output as dir index. E. g. images from dirs[0] will have output 0.
        transform: boolean - All images will be rotated/inverted, so input set will be bigger.

    Returns:
        np.array, np.array - First array contains images of shape (size_x, size_y, 3), second array goals, e. g. [0, 1].
    """
    dir_images = []
    images_count = 0

    for i in range(len(dirs)):
        dir_images.append([])
        file_names = os.listdir(dirs[i])

        for j in range(len(file_names)):
            file_path = os.path.join(dirs[i], file_names[j])
            img = read_img(file_path)

            dir_images[i].append(img)
            images_count += 1

    (size_y, size_x, depth) = dir_images[0][0].shape

    inputs = np.zeros((images_count, size_y, size_x, depth))
    outputs = np.zeros((images_count, 2))

    index = 0

    for i in range(len(dir_images)):
        for j in range(len(dir_images[i])):
            inputs[index, :, :, :] = dir_images[i][j]
            outputs[index, :] = create_output(i, len(dir_images))
            index += 1

    return inputs, outputs


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
