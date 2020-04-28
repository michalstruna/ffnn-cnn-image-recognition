from argparse import ArgumentParser

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

    parser.add_argument("--train", "-tr", type=int, nargs=3, metavar=("EPOCHS", "BATCH_SIZE", "VALIDATION"),
                        help="Train selected neural network (--use) with all images in selected dirs (--dir).")

    return parser.parse_args()


def get_networks(use):
    if not use:
        return [FFNN(), CNN()]
    else:
        return [FFNN()] if use == "FFNN" else [CNN()]
