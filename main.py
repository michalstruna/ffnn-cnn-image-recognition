#!/usr/bin/python3

import init
import numpy as np
import time
import os

import io_utils
from ai import GeneratorNeuralNetwork

console = io_utils.Console()
args = io_utils.read_args()
networks = io_utils.get_networks(args.use)

for network in networks:
    is_generator = isinstance(network, GeneratorNeuralNetwork)

    if args.run:
        network.load()
        img = io_utils.read_img(args.run, grayscale=is_generator)
        result = network.run(img)
        console.print_run(network, result)

    elif args.test:
        dir = args.dir if args.dir else os.path.join("img", "test")
        network.load()

        if is_generator:
            test_set = io_utils.prepare_input_set(dir)
            trues, falses = network.test(test_set)
        else:
            inputs, targets = io_utils.prepare_input_set(dir, transform_input=network.transform_input)
            trues, falses = network.test(inputs, targets)

        console.print_test(network, trues, falses)

    elif args.train:
        dir = args.dir if args.dir else os.path.join("img", "train")

        start = time.time()

        if is_generator:
            train_set, val_set = io_utils.prepare_input_set(dir, train=True, val_split=float(args.train[3]))
            history = network.train(train_set, val_set, int(args.train[0]), float(args.train[1]), int(args.train[2]))
        else:
            inputs, targets = io_utils.prepare_input_set(dir, train=True, transform_input=network.transform_input)
            history = network.train(inputs, targets, int(args.train[0]), float(args.train[1]), int(args.train[2]), float(args.train[3]))

        end = time.time()

        network.save()
        console.print_training(network, history, end - start)
    else:
        print("\n\n" + str(network) + "\n")
        network.load()
        network.summary()
        pass

console.show()
